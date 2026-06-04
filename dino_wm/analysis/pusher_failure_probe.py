"""Study WHICH fake-pusher evals fail and WHY -- to improve procedural pusher placement.

Reads a stock-pusht plan run's `plan_targets.pkl` (which stores the REAL goal state
incl. the real pusher, plus the fabricated `obs_g` goal image) and, per eval, measures
how far our procedural pusher guess is from the real recorded pusher, plus the goal's
difficulty (block translation / rotation). Cross-referenced with the per-eval success
mask(s), it isolates the failures that are SPECIFICALLY a pusher-faking problem (succeed
with the real pusher, fail with the fake) from failures that are just hard goals.

Run ON THE BOX where the pkl lives, e.g.:
  python analysis/pusher_failure_probe.py plan_outputs/20260604212201_pusht_gH5 \
      --seed 99 \
      --success_contact "T,F,F,F,T,T,F,T,T,T" \
      --success_real    "T,T,F,T,T,T,F,T,T,T" \
      --save_frames

Masks accept "T,F,.." / "1 0 .." / a pasted "[ True False .. ]" array. If you omit
masks it just reports the geometry; one mask -> failure geometry; two masks -> the
paired real-vs-contact bucketing (the useful one).
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.pusht.multicolor_common import (  # noqa: E402
    contact_pusher_pose, tee_world_polygon, angle_diff)
import shapely.geometry as sg  # noqa: E402

PUSHER_R = 15.0


def parse_mask(s, n):
    """Accept 'T,F,..' / '1 0 ..' / a pasted 'array([ True, False, ..])'."""
    if s is None:
        return None
    vals = []
    for t in re.findall(r"[A-Za-z]+|[01]", s):
        tl = t.lower()
        if tl in ("true", "t", "1"):
            vals.append(True)
        elif tl in ("false", "f", "0"):
            vals.append(False)
        # ignore stray words like 'array', 'dtype', 'bool'
    if len(vals) != n:
        raise ValueError(f"mask has {len(vals)} entries, expected {n}: {s!r}")
    return np.array(vals, dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run dir or path to plan_targets.pkl")
    ap.add_argument("--seed", type=int, default=99, help="eval_seed[i] = seed*i + 1")
    ap.add_argument("--success_contact", default=None)
    ap.add_argument("--success_real", default=None)
    ap.add_argument("--save_frames", action="store_true",
                    help="dump obs_0/obs_g PNGs for the studied failures")
    args = ap.parse_args()

    pkl = args.run if args.run.endswith(".pkl") else os.path.join(args.run, "plan_targets.pkl")
    run_dir = os.path.dirname(os.path.abspath(pkl))
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    s0 = np.asarray(d["state_0"], dtype=np.float64)   # (b, 7) [ax,ay,bx,by,th,vx,vy]
    sg_ = np.asarray(d["state_g"], dtype=np.float64)  # REAL goal (real pusher at [0:2])
    n = len(s0)
    m_c = parse_mask(args.success_contact, n)
    m_r = parse_mask(args.success_real, n)

    rows = []
    for i in range(n):
        block0, blockg, thg, th0 = s0[i, 2:4], sg_[i, 2:4], sg_[i, 4], s0[i, 4]
        real_p = sg_[i, 0:2]
        contact_p = contact_pusher_pose(block0, np.r_[blockg, thg])
        trans = float(np.linalg.norm(blockg - block0))
        rot_deg = float(np.degrees(angle_diff(thg, th0)))
        if contact_p is None:           # ~no translation -> we keep the real pusher
            gap, overlap = 0.0, False
        else:
            gap = float(np.linalg.norm(real_p - contact_p))
            T = tee_world_polygon(np.r_[blockg, thg])
            overlap = T.distance(sg.Point(contact_p)) < PUSHER_R
        rows.append(dict(i=i, seed=args.seed * i + 1, trans=trans, rot=rot_deg,
                         gap=gap, overlap=overlap, real_p=real_p, contact_p=contact_p))

    # ---- per-eval table -------------------------------------------------------
    hdr = f"{'idx':>3} {'seed':>5} {'trans':>6} {'rot°':>6} {'gap':>6} {'ovl':>3}"
    if m_c is not None: hdr += f" {'cont':>4}"
    if m_r is not None: hdr += f" {'real':>4}"
    print(hdr)
    for r in rows:
        line = f"{r['i']:>3} {r['seed']:>5} {r['trans']:>6.1f} {r['rot']:>6.1f} {r['gap']:>6.1f} {str(r['overlap'])[0]:>3}"
        if m_c is not None: line += f" {('OK' if m_c[r['i']] else 'X'):>4}"
        if m_r is not None: line += f" {('OK' if m_r[r['i']] else 'X'):>4}"
        print(line)

    def stat(name, idx):
        if len(idx) == 0:
            print(f"  {name:<16} n=0"); return
        tr = np.array([rows[i]["trans"] for i in idx])
        ro = np.array([rows[i]["rot"] for i in idx])
        gp = np.array([rows[i]["gap"] for i in idx])
        print(f"  {name:<16} n={len(idx):<2}  trans={tr.mean():6.1f}  rot={ro.mean():5.1f}  gap_real_vs_contact={gp.mean():6.1f}")

    # ---- the useful part: paired real-vs-contact bucketing --------------------
    if m_c is not None and m_r is not None:
        both_ok   = [i for i in range(n) if m_r[i] and m_c[i]]
        both_fail = [i for i in range(n) if not m_r[i] and not m_c[i]]
        fake_fail = [i for i in range(n) if m_r[i] and not m_c[i]]   # <- faking broke it
        fake_help = [i for i in range(n) if not m_r[i] and m_c[i]]
        print("\n=== paired buckets (real vs contact) ===")
        stat("both_ok", both_ok)
        stat("both_fail(hard)", both_fail)
        stat("FAKING_FAIL", fake_fail)
        stat("fake_help", fake_help)
        print(f"\nFAILURES SPECIFIC TO FAKING (real OK, contact X) -> study these: "
              f"{[rows[i]['seed'] for i in fake_fail]}")
        print("If FAKING_FAIL has high gap_real_vs_contact or high rot, the contact "
              "heuristic is the lever; if it's ~empty, faking is not the bottleneck.")
        study = fake_fail
    elif m_c is not None:
        fails = [i for i in range(n) if not m_c[i]]
        print("\n=== contact failures ===")
        stat("contact_fail", fails)
        stat("contact_ok", [i for i in range(n) if m_c[i]])
        study = fails
    else:
        study = []

    # ---- optional: dump goal/start frames for the studied failures ------------
    if args.save_frames and study:
        out = os.path.join(run_dir, "failure_frames")
        os.makedirs(out, exist_ok=True)
        try:
            from PIL import Image
            def save(arr, path):
                a = np.asarray(arr)
                a = np.squeeze(a)
                if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
                    a = np.transpose(a, (1, 2, 0))   # CHW -> HWC
                if a.max() <= 1.0: a = a * 255.0
                Image.fromarray(a.astype(np.uint8)).save(path)
            for i in study:
                save(d["obs_0"]["visual"][i], os.path.join(out, f"eval{i}_seed{rows[i]['seed']}_start.png"))
                save(d["obs_g"]["visual"][i], os.path.join(out, f"eval{i}_seed{rows[i]['seed']}_goal.png"))
            print(f"\nSaved start/goal frames for {len(study)} evals to {out}")
        except Exception as e:
            print(f"\n[save_frames skipped] {e}")


if __name__ == "__main__":
    main()
