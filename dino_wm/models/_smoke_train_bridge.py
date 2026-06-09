"""Local CPU smoke for train_bridge.py -- runs the full g training loop on the cached smoke
latents with the dummy text encoder (no transformers/GPU), asserting it trains end to end:
data loads, text table builds, tau estimates, train loss drops (overfits 6 samples), and the
checkpoint + history are written.

    /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python models/_smoke_train_bridge.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    out = tempfile.mkdtemp(prefix="g_train_smoke_")
    # batch_size=6 == all train -> 1 step/epoch, so use enough epochs to actually overfit.
    cmd = [sys.executable, "train_bridge.py",
           "--latent_dir", "data/pusht_multicolor_smoke/latents",
           "--data_path", "data/pusht_multicolor_smoke",
           "--out", out, "--epochs", "600", "--lr", "1e-3", "--depth", "2", "--batch_size", "6",
           "--dummy_text", "--device", "cpu", "--save_every", "600", "--seed", "0"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print("STDERR:\n", r.stderr[-2500:])
        raise SystemExit("train_bridge.py exited non-zero")

    assert os.path.exists(os.path.join(out, "g_best.pth")), "no g_best.pth checkpoint"
    hist = json.load(open(os.path.join(out, "train_history.json")))
    l0, lf = hist[0]["train_loss"], hist[-1]["train_loss"]
    print(f"\nSMOKE CHECK: train loss {l0:.3f} -> {lf:.3f} over {len(hist)} epochs "
          f"| final changed-cos {hist[-1]['val_changed_cos']:.3f}")
    # WIRING check: the model must be able to FIT the train set (loss -> ~0). val-cosine / val-loss
    # are meaningless here (dummy random text + 6 samples = memorization, no grounding to generalize);
    # the real Stage-1 gate (changed-cos >= 0.90) is checked on the box with real MiniLM + full data.
    assert lf < l0 * 0.05, f"train loss should overfit 6 samples to ~0 ({l0:.3f} -> {lf:.3f})"
    print("SMOKE OK: g training loop runs end-to-end and FITS the data "
          "(data + text table + tau + weighted-L2 loss + ckpt/history all wired).")


if __name__ == "__main__":
    main()
