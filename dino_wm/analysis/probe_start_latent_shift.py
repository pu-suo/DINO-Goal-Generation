"""Quantify the mp4-compression train/deploy shift on g's z_start (review finding P1).

g's cached training start latents come from mp4v-compressed video frames
(cache_latents.py -> PushTMultiColorDataset.get_frames -> obses/*.mp4), while
plan-time z_start is encoded from the CLEAN live env render. This probe re-renders
frame 0 exactly the way the generator produced it (PushTMultiColorEnv.set_layout +
reset_to_state, scripts/gen_pusht_multicolor.generate_one) and compares

    enc(clean render)   vs   cached start latent  (= enc(mp4 frame 0))

as per-episode full-grid cosine and per-patch L2. Interpretation: cos >= ~0.99 and
per-patch L2 well under the tf-1-step model error (~14-17) -> the shift is noise,
no action; materially below that -> re-cache start latents from clean renders and
retrain g.

Run (box; GPU light):
  python analysis/probe_start_latent_shift.py --data_path $DATASET_DIR/pusht_multicolor \
    --latent_dir $DATASET_DIR/pusht_multicolor/latents --n 50
"""
import argparse
import os
import sys

import numpy as np
import torch
from torchvision import transforms

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.img_transforms import default_transform
from datasets.pusht_multicolor_dset import PushTMultiColorDataset, label_to_layout
from env.pusht.pusht_multicolor_env import PushTMultiColorEnv
from models.dino import DinoV2Encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--latent_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--render_size", type=int, default=224)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens").to(device).eval()
    enc_resize = transforms.Resize((224 // 16) * encoder.patch_size)
    tfm = default_transform(224)

    dset = PushTMultiColorDataset(data_path=os.path.join(args.data_path, args.split),
                                  transform=tfm, normalize_action=False, with_velocity=True)
    cached = torch.load(os.path.join(args.latent_dir, args.split, "start_latents.pth"))

    env = PushTMultiColorEnv(render_size=args.render_size, with_velocity=True,
                             n_targets=4, n_bins=3)
    cos_all, l2_all = [], []
    for i in range(min(args.n, len(dset))):
        label = dset.labels[i]
        layout = label_to_layout(label)
        env.set_layout(layout)
        env.seed(int(label["seed"]))
        env.reset_to_state = np.asarray(label["init_state"], dtype=np.float64).copy()
        obs, _ = env.reset()
        frame = torch.from_numpy(np.asarray(obs["visual"])).permute(2, 0, 1).float() / 255.0
        x = tfm(frame.unsqueeze(0))
        with torch.no_grad():
            z_clean = encoder.forward(enc_resize(x.to(device))).cpu()[0]   # (196,384)
        z_mp4 = cached[i]
        cos_all.append(float(torch.nn.functional.cosine_similarity(
            z_clean.reshape(1, -1), z_mp4.reshape(1, -1))))
        l2_all.append(float(torch.linalg.norm(z_clean - z_mp4, dim=-1).mean()))

    cos_all, l2_all = np.array(cos_all), np.array(l2_all)
    print(f"[start-shift] n={len(cos_all)} | full-grid cos mean {cos_all.mean():.4f} "
          f"min {cos_all.min():.4f} | per-patch L2 mean {l2_all.mean():.2f} "
          f"max {l2_all.max():.2f}")
    print("  reference scales: tf-1-step model error ~14-17 per patch; "
          "g changed-cos gate 0.90")
    verdict = "NEGLIGIBLE (no action)" if cos_all.min() > 0.99 and l2_all.mean() < 5 else \
              "MATERIAL -> re-cache start latents from clean renders + retrain g"
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
