"""Local CPU smoke for analysis/eval_bridge_stage1.py -- validates every code path runs on the
smoke latents with a synthetic g checkpoint + synthetic pose decoder + dummy text (no GPU/net).
Checks the report has all fidelity + grounding + ablation fields (not the values, which are
meaningless for an untrained g / random decoder).

    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python analysis/_smoke_eval_bridge.py
"""
import json
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from models.bridge import BridgeG, N_PATCHES, DIM


def make_g_ckpt(path):
    g = BridgeG(dim=DIM, depth=2, heads=6, d_text=DIM)
    torch.save({"state_dict": g.state_dict(),
                "config": {"dim": DIM, "depth": 2, "heads": 6, "d_text": DIM},
                "tau": 20.0, "text_model": None}, path)


def make_decoder(path):
    D = N_PATCHES * DIM
    torch.save({"mu": torch.zeros(D), "sd": torch.ones(D), "W": torch.randn(D, 4) * 0.01,
                "ymu": torch.tensor([256.0, 256.0, 1.0, 0.0]), "dilation": 0}, path)


def main():
    tmp = tempfile.mkdtemp(prefix="eval_bridge_smoke_")
    gck, dec, out = (os.path.join(tmp, x) for x in ("g_best.pth", "dec.pt", "out"))
    make_g_ckpt(gck)
    make_decoder(dec)
    cmd = [sys.executable, "analysis/eval_bridge_stage1.py", "--ckpt", gck,
           "--latent_dir", "data/pusht_multicolor_smoke/latents",
           "--data_path", "data/pusht_multicolor_smoke", "--split", "test", "--train_split", "train",
           "--pose_decoder", dec, "--dummy_text", "--device", "cpu", "--out", out]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(r.stdout[-3200:])
    if r.returncode != 0:
        print("STDERR:\n", r.stderr[-3200:])
        raise SystemExit("eval_bridge_stage1.py exited non-zero")

    rep = json.load(open(os.path.join(out, "stage1_test.json")))
    for k in ("g_changed_cos", "g_changed_cos_macro", "n_zero_changed", "identity_changed_cos",
              "retrieval_changed_cos", "g_full_cos", "g_full_l2"):
        assert k in rep["fidelity"], f"missing fidelity.{k}"
    for k in ("decoder_trustworthy", "transfer_within_gate", "named_within_gate",
              "swapped_follows_swapped", "swapped_moved_to_swapped", "named_pos_mae_px",
              "agnostic_pos_mae_px"):
        assert k in rep["grounding"], f"missing grounding.{k}"
    print("\nSMOKE OK: Stage-1 eval runs end-to-end (fidelity + retrieval + pose-grounding + "
          "swapped-text + instruction-agnostic floor all wired; report written).")


if __name__ == "__main__":
    main()
