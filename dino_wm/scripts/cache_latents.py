"""
Phase 0.2 - cache frozen DINOv2 latents for the multi-color dataset.

Encodes the START frame (t=0) and the GOAL frame (block at the named target) of
every episode through the *frozen* DINOv2 ViT-S/14, reproducing the world model's
exact preprocessing:

    frame uint8 -> /255 -> default_transform(224)  [Resize, CenterCrop, Normalize(0.5,0.5)]
               -> Resize(encoder_image_size=196)   [VWorldModel.encoder_transform]
               -> DinoV2Encoder.forward            -> (196, 384) patch tokens

Writes <data_path>/latents/<split>/{start_latents.pth, goal_latents.pth}, each
(N, 196, 384). These feed PushTMultiColorLatentGoalDataset (the bridge `g` view).

GPU run (vast.ai):
    cd dino_wm
    DATASET_DIR=/data python scripts/cache_latents.py --data_path $DATASET_DIR/pusht_multicolor
Mac smoke test (cpu, slow, downloads ~85MB DINOv2 once):
    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python scripts/cache_latents.py \
        --data_path data/pusht_multicolor_smoke --device cpu --batch 8
"""
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from torchvision import transforms

# allow `python scripts/cache_latents.py` from the repo root
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from datasets.img_transforms import default_transform
from datasets.pusht_multicolor_dset import PushTMultiColorDataset
from models.dino import DinoV2Encoder


def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # MPS often lacks dinov2 ops; cpu is the safe Mac default


@torch.no_grad()
def encode_frames(frames, encoder, enc_resize, device, batch):
    """frames: (N, 3, 224, 224) already default_transform'd -> (N, 196, 384)."""
    out = []
    for i in range(0, len(frames), batch):
        chunk = enc_resize(frames[i:i + batch].to(device))
        out.append(encoder.forward(chunk).cpu())
    return torch.cat(out, dim=0)


def load_goal_frame(data_path, idx, tfm):
    img = imageio.imread(Path(data_path) / "goal_obses" / f"episode_{idx:06d}.png")  # HWC uint8
    t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    return tfm(t.unsqueeze(0))[0]  # (3,224,224)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(os.environ.get("DATASET_DIR", "data"), "pusht_multicolor"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device={device}")
    encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(device).eval()
    # match VWorldModel.encoder_transform: Resize((img//16)*patch_size)
    enc_image_size = (args.img_size // 16) * encoder.patch_size
    enc_resize = transforms.Resize(enc_image_size)
    print(f"encoder input {enc_image_size}x{enc_image_size} -> {196} tokens x {encoder.emb_dim}")
    tfm = default_transform(args.img_size)

    for split in args.splits:
        split_path = Path(args.data_path) / split
        if not split_path.exists():
            print(f"skip {split} (missing)")
            continue
        dset = PushTMultiColorDataset(data_path=str(split_path), transform=tfm,
                                      normalize_action=True, with_velocity=True)
        n = len(dset)
        start = torch.stack([dset.get_frames(i, [0])[0]["visual"][0] for i in range(n)])  # (N,3,224,224)
        goal = torch.stack([load_goal_frame(split_path, i, tfm) for i in range(n)])        # (N,3,224,224)

        z_start = encode_frames(start, encoder, enc_resize, device, args.batch)
        z_goal = encode_frames(goal, encoder, enc_resize, device, args.batch)

        out_dir = Path(args.data_path) / "latents" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(z_start, out_dir / "start_latents.pth")
        torch.save(z_goal, out_dir / "goal_latents.pth")
        print(f"[{split}] cached start {tuple(z_start.shape)} goal {tuple(z_goal.shape)} -> {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
