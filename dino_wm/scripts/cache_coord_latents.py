"""
Phase-1 (clean-scene pivot): cache frozen DINOv2 latents for the single-T COORDINATE
dataset produced by scripts/gen_pusht_coord.py.

Encodes BOTH the start frame (start_obses/episode_*.png) and the teleport goal frame
(goal_obses/episode_*.png) through the *frozen* DINOv2 ViT-S/14, reproducing the world
model's exact preprocessing (byte-for-byte with VWorldModel.encode_obs and with the
multicolor cache_latents.py so z_start/z_goal live in the SAME x_norm_patchtokens space
as the dynamics rollout latents):

    PNG uint8 -> /255 -> default_transform(224) [Resize,CenterCrop,Normalize(0.5,0.5)]
              -> Resize(encoder_image_size=(224//16)*14=196) [encoder_transform]
              -> DinoV2Encoder.forward -> (196, 384) patch tokens.

Writes <data_path>/latents/<split>/{start_latents.pth, goal_latents.pth} (N,196,384)
and copies specs.pth/start_poses.pth alongside so datasets/pusht_coord_dset.py can serve
(z_start, z_goal, spec) without re-reading the split dir.

GPU run (vast.ai 4090):
    cd dino_wm && DATASET_DIR=/workspace/data /workspace/envs/dino_wm/bin/python \
        scripts/cache_coord_latents.py --data_path $DATASET_DIR/pusht_coord
Mac smoke (cpu):
    SDL_VIDEODRIVER=dummy /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python \
        scripts/cache_coord_latents.py --data_path data/pusht_coord_smoke --device cpu --batch 8
"""
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from torchvision import transforms

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from datasets.img_transforms import default_transform
from models.dino import DinoV2Encoder


def pick_device(arg):
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_png(path, tfm):
    img = imageio.imread(path)  # HWC uint8
    t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    return tfm(t.unsqueeze(0))[0]  # (3,224,224)


@torch.no_grad()
def encode_frames(frames, encoder, enc_resize, device, batch):
    out = []
    for i in range(0, len(frames), batch):
        chunk = enc_resize(frames[i:i + batch].to(device))
        out.append(encoder.forward(chunk).cpu())
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.path.join(os.environ.get("DATASET_DIR", "data"), "pusht_coord"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device={device}")
    encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(device).eval()
    enc_image_size = (args.img_size // 16) * encoder.patch_size  # -> 196
    enc_resize = transforms.Resize(enc_image_size)
    print(f"encoder input {enc_image_size}x{enc_image_size} -> 196 tokens x {encoder.emb_dim}")
    tfm = default_transform(args.img_size)

    for split in args.splits:
        split_path = Path(args.data_path) / split
        if not (split_path / "specs.pth").exists():
            print(f"skip {split} (missing {split_path}/specs.pth)")
            continue
        n = sum(1 for _ in (split_path / "start_obses").glob("episode_*.png"))
        start = torch.stack([load_png(split_path / "start_obses" / f"episode_{i:06d}.png", tfm)
                             for i in range(n)])
        goal = torch.stack([load_png(split_path / "goal_obses" / f"episode_{i:06d}.png", tfm)
                            for i in range(n)])

        z_start = encode_frames(start, encoder, enc_resize, device, args.batch)
        z_goal = encode_frames(goal, encoder, enc_resize, device, args.batch)

        out_dir = Path(args.data_path) / "latents" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(z_start, out_dir / "start_latents.pth")
        torch.save(z_goal, out_dir / "goal_latents.pth")
        # copy the spec + start pose alongside the latents (loader convenience)
        torch.save(torch.load(split_path / "specs.pth"), out_dir / "specs.pth")
        torch.save(torch.load(split_path / "start_poses.pth"), out_dir / "start_poses.pth")
        print(f"[{split}] cached start {tuple(z_start.shape)} goal {tuple(z_goal.shape)} -> {out_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
