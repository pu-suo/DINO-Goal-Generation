"""Local smoke for analysis/fit_multicolor_pose_decoder.py on the smoke goal latents (wiring only;
6 train samples can't give accurate pose -- just checks it fits, evals, and saves a usable decoder)."""
import os, subprocess, sys, tempfile
import torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = tempfile.mkdtemp(prefix="mc_dec_smoke_")
dec = os.path.join(out, "mc_dec.pt")
r = subprocess.run([sys.executable, "analysis/fit_multicolor_pose_decoder.py",
                    "--latent_dir", "data/pusht_multicolor_smoke/latents",
                    "--data_path", "data/pusht_multicolor_smoke", "--device", "cpu", "--out", dec],
                   cwd=REPO, capture_output=True, text=True)
print(r.stdout[-1500:])
if r.returncode != 0:
    print("STDERR:\n", r.stderr[-2000:]); raise SystemExit("fit exited non-zero")
ck = torch.load(dec)
assert tuple(ck["W"].shape) == (196 * 384, 4), ck["W"].shape
assert ck["mu"].numel() == 196 * 384 and ck["ymu"].numel() == 4
assert ck["pose_param"] == ["x_px", "y_px", "cos", "sin"] and "metrics" in ck
print("SMOKE OK: multicolor pose-decoder fit runs + saves a decoder in linear_decoder.pt format.")
