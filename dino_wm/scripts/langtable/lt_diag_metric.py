"""Lever B evidence: is 0.05u the right relational success threshold, or does block geometry justify
a different (validated) contact distance? Anti-gaming: derive from geometry + oracle, never from the
disappointing eval number.

Corpus side (this file, dino_wm env): oracle env-success rate; final & min ||A-B|| distributions for
oracle SUCCESS vs FAIL episodes; the empirical closest-approach (contact) distance between named pairs.
"""
import numpy as np

def load(p):
    return dict(np.load(p, allow_pickle=True))

tr = load("/workspace/lt_cache_3k/train.npz"); va = load("/workspace/lt_cache_3k/val.npz")
blocks = [str(b) for b in tr["blocks"]]; bidx = {b: i for i, b in enumerate(blocks)}

def stats(c, name):
    S = c["seq_lengths"]; succ = c["success"].astype(bool)
    finalAB, minAB = [], []
    for e in range(len(S)):
        n = int(S[e])
        if n < 2:
            continue
        ai = bidx[str(c["start_block"][e])]; bi = bidx[str(c["target_block"][e])]
        bxy = c["block_xy"][e, :n]
        d = np.linalg.norm(bxy[:, ai] - bxy[:, bi], axis=-1)
        finalAB.append(d[-1]); minAB.append(d.min())
    finalAB = np.array(finalAB); minAB = np.array(minAB)
    print(f"\n[{name}] n={len(succ)}  ORACLE env-success rate (block2block ||A-B||<0.05) = {succ.mean():.3f}")
    print(f"  final ||A-B|| (last frame): mean={finalAB.mean():.4f} median={np.median(finalAB):.4f} "
          f"p10={np.percentile(finalAB,10):.4f} min={finalAB.min():.4f}")
    print(f"  MIN ||A-B|| over episode (closest approach = contact): mean={minAB.mean():.4f} "
          f"median={np.median(minAB):.4f} p10={np.percentile(minAB,10):.4f} MIN={minAB.min():.4f}")
    for thr in [0.04, 0.05, 0.06, 0.07, 0.08]:
        print(f"    oracle frac with min||A-B|| < {thr:.2f}: {np.mean(minAB<thr):.3f}   "
              f"final < {thr:.2f}: {np.mean(finalAB<thr):.3f}")
    return finalAB, minAB

stats(tr, "train"); stats(va, "val")

# global closest-approach between ANY two blocks (physical contact floor) across the corpus
allmin = 1e9
for c in (tr, va):
    S = c["seq_lengths"]
    for e in range(len(S)):
        n = int(S[e])
        bxy = c["block_xy"][e, :n]                      # (n,8,2)
        for i in range(8):
            for j in range(i + 1, 8):
                dij = np.linalg.norm(bxy[:, i] - bxy[:, j], axis=-1).min()
                allmin = min(allmin, dij)
print(f"\nGLOBAL closest any-two-block center distance ever (physical contact floor) = {allmin:.4f}u")
print("INTERPRETATION: if oracle env-success >=0.95 and contact floor <=0.05 -> 0.05u is achievable at")
print("contact and IS the task's ground-truth relational metric (block2block reward). within-0.07 = a")
print("GAP (near, not touching); counting it would diverge from the env success def = inflation.")
