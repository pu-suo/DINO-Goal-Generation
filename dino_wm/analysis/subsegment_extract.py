"""Part A (spec-invariant): bounded sub-segment extraction + dual-bound filter +
CPU replay reachability self-test + decorrelation/leak logging.

WHY: the previous goal generator used the transformed END of the FULL real
trajectory -> goals needed up to 292px / 179deg, far beyond the ~50-action
planner horizon (oracle SR 0.194). Fix: goals = BOUNDED sub-segments
(frame_i, frame_{i+h}) of a real trajectory, small enough in block displacement
AND rotation to be reachable WITHIN horizon -- and, being sliding windows, far
more numerous than one-goal-per-trajectory.

This module is language/transform-agnostic (Parts B/C/E add language, metric,
rigid transform, render). It works on the RAW pusht_noise states+actions.

The replay self-test (A3) is the cheap, GPU-free pre-gate: replay a sub-segment's
OWN recorded actions in the env from its start state and confirm the block lands
at the recorded end pose. That proves a reachable path EXISTS within horizon. It
does NOT prove the CEM planner FINDS it -- that is the later GPU oracle gate.
Necessary, not sufficient; free.

Replay fidelity caveat (measured, not assumed): env._set_state restores the
agent velocity but NOT the block velocity (block vel -> 0). So a mid-trajectory
reset -- exactly what the planner does when it is handed frame_i -- can lose the
block's momentum. We measure two replay variants to isolate this:
  full   : replay from frame 0 (true rest start) -> validates env config +
           action indexing + determinism (should be ~0 error).
  subseg : replay from frame_i via _set_state -> the planner-faithful number,
           includes any block-velocity-loss effect.

Box (4090, CPU):
  DATASET_DIR=/workspace/data /workspace/envs/dino_wm/bin/python \
    analysis/subsegment_extract.py --split train --sweep_h --n_replay 400 \
    --D_max 50 --R_max 12 --h 16 --out analysis/out/subseg_train.json
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ACTION_SCALE = 100.0  # env multiplies the passed action by this; data is raw -> divide


# --- geometry ----------------------------------------------------------------
def wrap_pi(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def load_split(data_path, split):
    """Load raw pusht_noise tensors for one split.
    Returns dict with states5 (N,T,5), vel (N,T,2), rel_actions (N,T,2), seqlen (N,)."""
    d = Path(data_path) / split
    states5 = torch.load(d / "states.pth").float().numpy()          # (N,T,5)
    vel = torch.load(d / "velocities.pth").float().numpy()          # (N,T,2)
    rel = torch.load(d / "rel_actions.pth").float().numpy()         # (N,T,2) raw
    with open(d / "seq_lengths.pkl", "rb") as f:
        seqlen = np.asarray(pickle.load(f), dtype=np.int64)         # (N,)
    n = min(len(states5), len(seqlen), len(vel), len(rel))
    return dict(states5=states5[:n], vel=vel[:n], rel=rel[:n], seqlen=seqlen[:n])


def extract_candidates(data, h, stride=1):
    """All (traj, i, j=i+h) windows with both frames inside the valid length.
    Returns a structured dict of numpy arrays (vectorised, no per-frame python)."""
    states5, seqlen = data["states5"], data["seqlen"]
    traj_ids, ii = [], []
    for t in range(len(seqlen)):
        L = int(seqlen[t])
        last_start = L - 1 - h          # need i and i+h both < L
        if last_start < 0:
            continue
        starts = np.arange(0, last_start + 1, stride)
        traj_ids.append(np.full(len(starts), t, dtype=np.int64))
        ii.append(starts)
    if not traj_ids:
        return dict(traj=np.array([], int), i=np.array([], int), j=np.array([], int),
                    dp=np.zeros((0, 2)), dp_mag=np.array([]), drot=np.array([]))
    traj = np.concatenate(traj_ids)
    i = np.concatenate(ii)
    j = i + h
    b0 = states5[traj, i, 2:4]
    b1 = states5[traj, j, 2:4]
    dp = b1 - b0                                            # (M,2) block displacement
    dp_mag = np.linalg.norm(dp, axis=1)
    drot = wrap_pi(states5[traj, j, 4] - states5[traj, i, 4])
    return dict(traj=traj, i=i, j=j, dp=dp, dp_mag=dp_mag, drot=drot)


def dual_bound_mask(cand, D_max, R_max_deg, D_min=0.0, R_min_deg=0.0):
    """Keep candidates within the reachable UPPER bound (|Dp|<D_max AND |Drot|<R_max)
    AND above a MEANINGFUL-motion LOWER bound (|Dp|>=D_min OR |Drot|>=R_min) -- the
    lower bound drops degenerate 'stay-put' windows (pusher not in contact) so the
    language command is non-trivial. D_min=R_min=0 -> upper-bound-only (old behavior)."""
    rot_deg = np.abs(np.degrees(cand["drot"]))
    upper = (cand["dp_mag"] < D_max) & (rot_deg < R_max_deg)
    lower = (cand["dp_mag"] >= D_min) | (rot_deg >= R_min_deg)
    return upper & lower


# --- buckets (provisional; Part B formalises naming) -------------------------
def bucketize(dp, drot, n_dir=8, mag_edges=(0, 25, 50), rot_edges_deg=(0, 5, 12)):
    """Coarse qualitative buckets: 8-way direction of Dp, magnitude band, rotation
    sign + coarse magnitude band. Returns integer codes (dir, mag, rsign, rmag)."""
    ang = np.arctan2(dp[:, 1], dp[:, 0])                    # state-coord direction
    # 8-way sectors CENTERED on the compass directions: bin 0 = +x, bin 2 = +y,
    # bin 4 = -x, bin 6 = -y (each sector spans +/-22.5deg about its center).
    dir_bin = np.round(ang / (2 * np.pi / n_dir)).astype(int) % n_dir
    mag = np.linalg.norm(dp, axis=1)
    mag_bin = np.clip(np.digitize(mag, mag_edges[1:]), 0, len(mag_edges) - 1)
    rsign = np.sign(drot).astype(int)                       # -1,0,+1
    rmag_bin = np.clip(np.digitize(np.abs(np.degrees(drot)), rot_edges_deg[1:]),
                       0, len(rot_edges_deg) - 1)
    return dir_bin, mag_bin, rsign, rmag_bin


def circ_corr(a, b):
    """Circular correlation coefficient (Jammalamadaka-SenGupta) between angle arrays."""
    a = np.asarray(a); b = np.asarray(b)
    am = np.angle(np.mean(np.exp(1j * a)))
    bm = np.angle(np.mean(np.exp(1j * b)))
    num = np.sum(np.sin(a - am) * np.sin(b - bm))
    den = np.sqrt(np.sum(np.sin(a - am) ** 2) * np.sum(np.sin(b - bm) ** 2))
    return float(num / den) if den > 0 else float("nan")


# --- replay (A3) -------------------------------------------------------------
def make_replay_env():
    """PushT env matching the PLANNING env (legacy=False, relative=True,
    action_scale=100, with_velocity, clean scene), with rendering stubbed out so
    physics-only replay is cheap. Render output only feeds the (ignored) visual."""
    from env.pusht.pusht_wrapper import PushTWrapper
    env = PushTWrapper(with_velocity=True, with_target=False)
    dummy = np.zeros((4, 4, 3), dtype=np.uint8)
    env._render_frame = lambda *a, **k: dummy              # skip rasterisation
    return env


def replay_end_pose(env, state7_start, actions_raw, seed=0):
    """Set env to state7_start, step through actions (raw rel_actions, scaled back),
    return final block pose [x, y, angle]. No rendering cost."""
    env.seed(seed)
    env.reset_to_state = np.asarray(state7_start, dtype=np.float64)
    env.reset()
    for a in actions_raw:
        env.step(np.asarray(a) / ACTION_SCALE)
    return np.array([env.block.position[0], env.block.position[1],
                     float(env.block.angle) % (2 * np.pi)])


def block_pose_err(pred_xyang, true_xyang):
    """(position L2 px, angle err deg) between two [x,y,angle] poses."""
    dpos = float(np.linalg.norm(pred_xyang[:2] - true_xyang[:2]))
    dang = float(np.degrees(np.abs(wrap_pi(pred_xyang[2] - true_xyang[2]))))
    return dpos, dang


def state7(data, t, k):
    """7-d env state [agent_xy, block_xy, angle, agent_vel_xy] for traj t frame k."""
    return np.concatenate([data["states5"][t, k], data["vel"][t, k]])


def replay_subsegment(env, data, t, i, j, variant="subseg"):
    """Replay a sub-segment and return (pos_err_px, ang_err_deg) vs recorded end.
    variant 'subseg': start at frame i via _set_state (planner-faithful).
    variant 'full'  : start at frame 0 (true rest) and replay 0..j (config check)."""
    true_end = np.array([data["states5"][t, j, 2], data["states5"][t, j, 3],
                         data["states5"][t, j, 4] % (2 * np.pi)])
    if variant == "full":
        acts = data["rel"][t, 0:j]
        pred = replay_end_pose(env, state7(data, t, 0), acts)
    else:
        acts = data["rel"][t, i:j]
        pred = replay_end_pose(env, state7(data, t, i), acts)
    return block_pose_err(pred, true_end)


# --- reporting ---------------------------------------------------------------
def pct(x, ps=(50, 90, 95, 99, 100)):
    x = np.asarray(x)
    return {f"p{p}": round(float(np.percentile(x, p)), 2) for p in ps} if len(x) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=os.environ.get("DATASET_DIR", ".") + "/pusht_noise")
    ap.add_argument("--split", default="train")
    ap.add_argument("--h", type=int, default=16, help="window length (raw frames)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--D_max", type=float, default=50.0, help="max block displacement px")
    ap.add_argument("--R_max", type=float, default=12.0, help="max |block rotation| deg")
    ap.add_argument("--D_min", type=float, default=0.0, help="min block displacement px (meaningful-motion lower bound)")
    ap.add_argument("--R_min", type=float, default=0.0, help="min |block rotation| deg (OR-combined with D_min)")
    ap.add_argument("--sweep_h", action="store_true", help="report Dp/Drot dists across h")
    ap.add_argument("--n_replay", type=int, default=400, help="sub-segments to replay-test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    print(f"[load] {args.data_path}/{args.split}")
    data = load_split(args.data_path, args.split)
    N = len(data["seqlen"])
    print(f"[load] {N} trajs | seqlen p50={np.percentile(data['seqlen'],50):.0f} "
          f"min={data['seqlen'].min()} max={data['seqlen'].max()}")
    report = {"split": args.split, "n_traj": int(N), "args": vars(args)}

    # --- A1/A2: candidate yield + Dp/Drot distribution, sweeping h -----------
    h_list = [5, 8, 12, 16, 24, 32] if args.sweep_h else [args.h]
    report["sweep"] = {}
    for h in h_list:
        c = extract_candidates(data, h, args.stride)
        m = dual_bound_mask(c, args.D_max, args.R_max, args.D_min, args.R_min)
        report["sweep"][h] = {
            "candidates": int(len(c["dp_mag"])),
            "dp_px": pct(c["dp_mag"]),
            "drot_deg": pct(np.abs(np.degrees(c["drot"]))),
            "survivors_dualbound": int(m.sum()),
            "survivor_frac": round(float(m.mean()), 4) if len(m) else 0.0,
        }
        print(f"[h={h:2d}] cand={len(c['dp_mag']):>8d}  "
              f"dp p50/p90={report['sweep'][h]['dp_px'].get('p50')}/{report['sweep'][h]['dp_px'].get('p90')}  "
              f"drot p50/p90={report['sweep'][h]['drot_deg'].get('p50')}/{report['sweep'][h]['drot_deg'].get('p90')}  "
              f"survivors={int(m.sum())} ({100*m.mean():.1f}%)")

    # --- chosen h: survivors, buckets, replay, leak --------------------------
    c = extract_candidates(data, args.h, args.stride)
    m = dual_bound_mask(c, args.D_max, args.R_max, args.D_min, args.R_min)
    sel = np.where(m)[0]
    print(f"\n[chosen h={args.h}] survivors={len(sel)} "
          f"(D in [{args.D_min},{args.D_max}]px, |R| in [{args.R_min},{args.R_max}]deg "
          f"upper-AND lower-OR)")
    report["chosen"] = {"h": args.h, "D_max": args.D_max, "R_max": args.R_max,
                        "D_min": args.D_min, "R_min": args.R_min,
                        "survivors": int(len(sel)),
                        "survivor_dp_px": pct(c["dp_mag"][sel]),
                        "survivor_drot_deg": pct(np.abs(np.degrees(c["drot"][sel])))}
    print(f"[chosen] survivor dp p50/p90={report['chosen']['survivor_dp_px'].get('p50')}/"
          f"{report['chosen']['survivor_dp_px'].get('p90')}px  "
          f"drot p50/p90={report['chosen']['survivor_drot_deg'].get('p50')}/"
          f"{report['chosen']['survivor_drot_deg'].get('p90')}deg")

    # A4: within-bucket spread of continuous (Dp, Drot) among survivors
    dp_s, drot_s = c["dp"][sel], c["drot"][sel]
    dbin, mbin, rsign, rmbin = bucketize(dp_s, drot_s)
    codes = (dbin * 100 + mbin * 10 + (rsign + 1) * 3 + rmbin)
    spreads_dp, spreads_rot, bucket_sizes = [], [], []
    for code in np.unique(codes):
        idx = np.where(codes == code)[0]
        if len(idx) < 5:
            continue
        bucket_sizes.append(len(idx))
        # spread = std of the continuous displacement magnitude & rotation in-bucket
        spreads_dp.append(float(np.std(c["dp_mag"][sel][idx])))
        spreads_rot.append(float(np.degrees(np.std(drot_s[idx]))))
    report["chosen"]["n_buckets_ge5"] = int(len(bucket_sizes))
    report["chosen"]["within_bucket_dp_std_px"] = pct(spreads_dp) if spreads_dp else {}
    report["chosen"]["within_bucket_rot_std_deg"] = pct(spreads_rot) if spreads_rot else {}
    report["chosen"]["bucket_size"] = pct(bucket_sizes) if bucket_sizes else {}
    print(f"[bucket] {len(bucket_sizes)} buckets(>=5)  "
          f"within-bucket dp-std p50/p90={report['chosen']['within_bucket_dp_std_px'].get('p50')}/"
          f"{report['chosen']['within_bucket_dp_std_px'].get('p90')}px  "
          f"rot-std p50/p90={report['chosen']['within_bucket_rot_std_deg'].get('p50')}/"
          f"{report['chosen']['within_bucket_rot_std_deg'].get('p90')}deg")

    # A4: pusher-start-direction vs goal-direction leak correlation
    t_s, i_s = c["traj"][sel], c["i"][sel]
    goal_dir = np.arctan2(dp_s[:, 1], dp_s[:, 0])
    vel_i = data["vel"][t_s, i_s]
    vel_dir = np.arctan2(vel_i[:, 1], vel_i[:, 0])
    act_i = data["rel"][t_s, i_s]
    act_dir = np.arctan2(act_i[:, 1], act_i[:, 0])
    moving = np.linalg.norm(vel_i, axis=1) > 1e-3
    # VISUALLY-AVAILABLE leak channel: g sees a STATIC start frame, so it cannot read
    # velocity directly -- the readable cue is the pusher->block contact geometry (which
    # side the pusher sits on). Push direction ~ (block - agent). This is the honest
    # proxy for what g could exploit; the Part-D ablation tests the true leak.
    ab = data["states5"][t_s, i_s, 2:4] - data["states5"][t_s, i_s, 0:2]   # block - agent
    contact_dir = np.arctan2(ab[:, 1], ab[:, 0])
    pb_dist = np.linalg.norm(ab, axis=1)
    in_contact = pb_dist < 40.0     # pusher radius 15 + block extent; loose contact gate
    report["chosen"]["leak"] = {
        "circ_corr_contact_dir_vs_goal_dir": circ_corr(contact_dir, goal_dir),
        "circ_corr_contact_in_contact": circ_corr(contact_dir[in_contact], goal_dir[in_contact]) if in_contact.any() else float("nan"),
        "mean_cos_contact_goal": float(np.mean(np.cos(contact_dir - goal_dir))),
        "frac_in_contact_at_start": round(float(in_contact.mean()), 3),
        "circ_corr_vel_dir_vs_goal_dir": circ_corr(vel_dir[moving], goal_dir[moving]),
        "circ_corr_action_dir_vs_goal_dir": circ_corr(act_dir, goal_dir),
        "mean_cos_action_goal": float(np.mean(np.cos(act_dir - goal_dir))),
        "frac_pusher_moving_at_start": round(float(moving.mean()), 3),
    }
    lk = report["chosen"]["leak"]
    print(f"[leak] VISUAL contact-dir->goal: circ-corr={lk['circ_corr_contact_dir_vs_goal_dir']:.3f} "
          f"(in-contact {lk['circ_corr_contact_in_contact']:.3f}, {100*lk['frac_in_contact_at_start']:.0f}% in contact) "
          f"mean-cos={lk['mean_cos_contact_goal']:.3f}")
    print(f"[leak] (proxies) vel-dir->goal circ-corr={lk['circ_corr_vel_dir_vs_goal_dir']:.3f}  "
          f"action-dir->goal circ-corr={lk['circ_corr_action_dir_vs_goal_dir']:.3f}")

    # A3: replay self-test on a random sample of survivors (n_replay)
    if len(sel) and args.n_replay > 0:
        env = make_replay_env()
        k = min(args.n_replay, len(sel))
        samp = rng.choice(sel, size=k, replace=False)
        errs_sub, errs_full = [], []
        for s in samp:
            t, i, j = int(c["traj"][s]), int(c["i"][s]), int(c["j"][s])
            errs_sub.append(replay_subsegment(env, data, t, i, j, "subseg"))
            errs_full.append(replay_subsegment(env, data, t, i, j, "full"))
        errs_sub = np.array(errs_sub); errs_full = np.array(errs_full)
        # tolerance: a sub-segment "replays" if the block lands within the success gate
        TOL_POS, TOL_ANG = 20.0, np.degrees(np.pi / 9)     # stock pose gate
        for name, e in (("subseg", errs_sub), ("full", errs_full)):
            ok = (e[:, 0] < TOL_POS) & (e[:, 1] < TOL_ANG)
            report.setdefault("replay", {})[name] = {
                "n": int(len(e)),
                "pos_err_px": pct(e[:, 0]),
                "ang_err_deg": pct(e[:, 1]),
                "pass_frac_at_gate": round(float(ok.mean()), 4),
                "fail_frac": round(float(1 - ok.mean()), 4),
            }
            print(f"[replay:{name}] n={len(e)}  pos p50/p90={pct(e[:,0]).get('p50')}/{pct(e[:,0]).get('p90')}px  "
                  f"ang p50/p90={pct(e[:,1]).get('p50')}/{pct(e[:,1]).get('p90')}deg  "
                  f"pass@gate={100*ok.mean():.1f}%")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2, default=str)
        print(f"\n[out] -> {args.out}")
    return report


if __name__ == "__main__":
    main()
