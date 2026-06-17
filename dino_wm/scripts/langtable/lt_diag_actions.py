"""Diagnostic: oracle action scale vs the CEM sampling scale (H.0b scatter root-cause check)."""
import numpy as np

tr = dict(np.load("/workspace/lt_cache_3k/train.npz", allow_pickle=True))
fs = int(tr["frameskip"])
A = [tr["actions"][i, :int(tr["seq_lengths"][i])] for i in range(len(tr["seq_lengths"]))]
A = np.concatenate(A, 0)            # (M, fs*2) per model-step
env = A.reshape(-1, 2)             # (M*fs, 2) per env-action
mag = np.linalg.norm(env, axis=1)
ac = np.abs(env)
print("ORACLE per-ENV-action (the in-distribution action the dynamics was trained on):")
print("  mean|comp|=%.4f  std/comp=%s" % (ac.mean(), env.std(0)))
print("  |comp| percentiles: p50=%.4f p90=%.4f p99=%.4f max=%.4f" %
      (np.percentile(ac, 50), np.percentile(ac, 90), np.percentile(ac, 99), ac.max()))
print("  magnitude: mean=%.4f p90=%.4f p99=%.4f max=%.4f" %
      (mag.mean(), np.percentile(mag, 90), np.percentile(mag, 99), mag.max()))
print("CEM currently samples N(0, sigma=0.06), clamp +/-0.10 per env-action component.")
print("  frac of oracle |comp| exceeding 0.06: %.3f ; exceeding 0.10: %.3f" %
      (np.mean(ac > 0.06), np.mean(ac > 0.10)))
print("  => if oracle |comp| is mostly << 0.06, the CEM explores actions the dynamics never saw")
print("     (OOD) -> exploitation + violent shoves -> scene scatter -> R OOD -> render-drift.")
