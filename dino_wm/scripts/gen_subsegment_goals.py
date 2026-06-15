"""Option B Part 3a: generate bounded sub-segment goals with rotation commands.

Goals = (start_state, goal_state) bounded sub-segments of REAL pusht_noise
trajectories (reachable-within-horizon by construction; Part A), labelled with the
ROTATION-ONLY command (Part 1). RAW clean scene -- NO rigid transform (deferred);
clean_retrain (block-TF 7.06) already covers this distribution. Held-out by
TRAJECTORY (disjoint pools, traj_id %% test_mod == 0 -> test) and stratified across
the 5 rotation buckets so every command is represented.

Saves start_states/goal_states (N,5), rot_buckets (N,2 = sign,band), command_text,
traj_ids (leakage audit). The oracle/g render clean frames via env.prepare, as
plan_rigid does -- no frames stored here.

Box:
  DATASET_DIR=/workspace/data /workspace/envs/dino_wm/bin/python \
    scripts/gen_subsegment_goals.py --n_train 8000 --n_test 1000 \
    --out $DATASET_DIR/pusht_subseg
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.subsegment_extract import load_split, extract_candidates, dual_bound_mask
from datasets.rotation_command import (
    rotation_bucket, rotation_command_text, signed_drot_deg, all_buckets, bucket_name)


def stratified_by_bucket(buckets_per_cand, cand_idx, n, rng):
    """Round-robin across the 5 rotation buckets so each command is represented."""
    by_b = {b: [] for b in all_buckets()}
    for k in cand_idx:
        by_b[buckets_per_cand[k]].append(k)
    for b in by_b:
        rng.shuffle(by_b[b])
    keys = [b for b in all_buckets() if by_b[b]]
    out, bi = [], 0
    while len(out) < n and any(by_b[k] for k in keys):
        k = keys[bi % len(keys)]
        if by_b[k]:
            out.append(by_b[k].pop())
        bi += 1
    return np.array(out[:n], dtype=np.int64)


def build(data, traj_mask, dials, n, rng):
    c = extract_candidates(data, dials["h"], dials["stride"])
    keep = dual_bound_mask(c, dials["D_max"], dials["R_max"], dials["D_min"], dials["R_min"])
    keep &= traj_mask[c["traj"]]
    idx = np.where(keep)[0]
    # bucket per surviving candidate (from signed Drot deg)
    drot_deg = np.degrees(c["drot"])
    bpc = {int(k): rotation_bucket(drot_deg[k]) for k in idx}
    sel = stratified_by_bucket(bpc, list(idx), n, rng)
    rec = {"start": [], "goal": [], "bucket": [], "text": [], "traj": []}
    for k in sel:
        t, i, j = int(c["traj"][k]), int(c["i"][k]), int(c["j"][k])
        s5 = data["states5"][t, i].astype(np.float32)
        g5 = data["states5"][t, j].astype(np.float32)
        d = signed_drot_deg(s5[4], g5[4])
        rec["start"].append(s5); rec["goal"].append(g5)
        rec["bucket"].append(rotation_bucket(d)); rec["text"].append(rotation_command_text(d))
        rec["traj"].append(t)
    return rec, sel, c


def save_split(out, split, rec):
    d = Path(out) / split
    d.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(np.array(rec["start"]), dtype=torch.float32), d / "start_states.pth")
    torch.save(torch.tensor(np.array(rec["goal"]), dtype=torch.float32), d / "goal_states.pth")
    torch.save(torch.tensor(np.array(rec["bucket"]), dtype=torch.int64), d / "rot_buckets.pth")
    torch.save(torch.tensor(np.array(rec["traj"]), dtype=torch.int64), d / "traj_ids.pth")
    pickle.dump(rec["text"], open(d / "command_text.pkl", "wb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATASET_DIR", ".") + "/pusht_noise")
    ap.add_argument("--src_split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--h", type=int, default=16); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--D_max", type=float, default=50.0); ap.add_argument("--R_max", type=float, default=12.0)
    ap.add_argument("--D_min", type=float, default=15.0); ap.add_argument("--R_min", type=float, default=5.0)
    ap.add_argument("--test_mod", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    dials = dict(h=args.h, stride=args.stride, D_max=args.D_max, R_max=args.R_max,
                 D_min=args.D_min, R_min=args.R_min)

    data = load_split(args.data_path, args.src_split)
    N = len(data["seqlen"])
    tid = np.arange(N)
    test_mask = (tid % args.test_mod == 0); train_mask = ~test_mask
    print(f"[split] {train_mask.sum()} train-pool / {test_mask.sum()} held-out trajs (disjoint)")

    tr, _, _ = build(data, train_mask, dials, args.n_train, rng)
    te, _, _ = build(data, test_mask, dials, args.n_test, rng)
    # leakage guard: no shared trajectory across splits
    shared = set(tr["traj"]) & set(te["traj"])
    assert not shared, f"LEAK: {len(shared)} trajectories shared across train/test"
    save_split(args.out, "train", tr); save_split(args.out, "test", te)

    for nm, rec in (("train", tr), ("test", te)):
        cnt = Counter(tuple(b) for b in rec["bucket"])
        dist = ", ".join(f"{bucket_name(b)}:{cnt.get(b,0)}" for b in all_buckets())
        print(f"[{nm}] {len(rec['start'])} goals | buckets: {dist}")
    json.dump({"dials": dials, "test_mod": args.test_mod, "seed": args.seed,
               "n_train": len(tr["start"]), "n_test": len(te["start"])},
              open(Path(args.out) / "manifest.json", "w"), indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
