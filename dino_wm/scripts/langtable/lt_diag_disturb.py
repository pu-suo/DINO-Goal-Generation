"""E.1 baseline: how much do NON-target blocks move in the oracle (human-demo proxy) trajectories?
The closed-loop anti-bulldoze target is 'non-target displacement ~ human-demo incidental level'."""
import numpy as np

va = dict(np.load("/workspace/lt_cache_3k/val.npz", allow_pickle=True))
blocks = [str(b) for b in va["blocks"]]; bidx = {b: i for i, b in enumerate(blocks)}
disp_all = []
for e in range(len(va["seq_lengths"])):
    S = int(va["seq_lengths"][e])
    if S < 2:
        continue
    ai = bidx[str(va["start_block"][e])]; bi = bidx[str(va["target_block"][e])]
    bxy = va["block_xy"][e, :S]                      # (S,8,2)
    others = [k for k in range(len(blocks)) if k not in (ai, bi)]
    d = np.linalg.norm(bxy[-1, others] - bxy[0, others], axis=-1)   # per non-target block net displacement
    disp_all.append(d.mean())
disp_all = np.array(disp_all)
print(f"ORACLE (human-demo proxy) non-target block displacement over {len(disp_all)} val trajs:")
print(f"  mean={disp_all.mean():.4f}u  median={np.median(disp_all):.4f}u  "
      f"p90={np.percentile(disp_all,90):.4f}u  max={disp_all.max():.4f}u")
print(f"  => closed-loop H.3 disturb should be ~this. (H.3 smoke measured mean 0.007u.)")
