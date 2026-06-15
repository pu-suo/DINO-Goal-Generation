"""Guard tests for Part A (subsegment_extract). Every guard gets a should-PASS and
a should-FAIL case. Pure-logic tests run anywhere; replay-fidelity tests need the
pusht_noise data + env (4090).

  /workspace/envs/dino_wm/bin/python analysis/test_subsegment.py            # logic only
  DATASET_DIR=/workspace/data ... analysis/test_subsegment.py --with_env    # + replay
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.subsegment_extract import (
    wrap_pi, extract_candidates, dual_bound_mask, bucketize, circ_corr,
    load_split, make_replay_env, replay_subsegment, state7, replay_end_pose)

P = 0


def ok(name, cond):
    global P
    assert cond, f"FAIL: {name}"
    P += 1
    print(f"  ok  {name}")


# --- pure-logic guards -------------------------------------------------------
def test_wrap_pi():
    ok("wrap +350deg -> -10deg", abs(np.degrees(wrap_pi(np.radians(350))) + 10) < 1e-6)
    ok("wrap -190deg -> +170deg", abs(np.degrees(wrap_pi(np.radians(-190))) - 170) < 1e-6)
    ok("wrap +5deg unchanged", abs(np.degrees(wrap_pi(np.radians(5))) - 5) < 1e-6)


def _toy(seqlen, T=40):
    """Synthetic data: block walks +1px/frame in x and +1deg/frame, agent/vel zero."""
    N = len(seqlen)
    states5 = np.zeros((N, T, 5))
    for t in range(N):
        L = seqlen[t]
        states5[t, :L, 2] = np.arange(L)              # block x = frame index
        states5[t, :L, 4] = np.radians(np.arange(L))  # block angle = frame deg
    vel = np.zeros((N, T, 2))
    rel = np.zeros((N, T, 2))
    return dict(states5=states5, vel=vel, rel=rel, seqlen=np.array(seqlen))


def test_extractor_bounds():
    data = _toy([30, 10, 5])
    h = 8
    c = extract_candidates(data, h, stride=1)
    # window bounds: every j < L of its trajectory  (SHOULD-PASS)
    Ls = data["seqlen"][c["traj"]]
    ok("all i>=0", (c["i"] >= 0).all())
    ok("all j=i+h", (c["j"] == c["i"] + h).all())
    ok("all j < L (no out-of-traj window)", (c["j"] < Ls).all())
    # counts: traj L=30 -> starts 0..(30-1-8)=21 -> 22; L=10 -> 0..1 ->2; L=5 -> none
    ok("candidate count exact (22+2+0)", len(c["i"]) == 24)
    # SHOULD-FAIL guard: a window longer than every traj yields zero candidates
    c2 = extract_candidates(data, h=40, stride=1)
    ok("h>maxL -> 0 candidates", len(c2["i"]) == 0)


def test_extractor_delta():
    data = _toy([30])
    c = extract_candidates(data, h=10, stride=10)   # starts 0,10 (last_start=19)
    # block walks +1px/frame, +1deg/frame -> over h=10: dp_x=10, drot=10deg
    ok("dp magnitude == h px", np.allclose(c["dp_mag"], 10.0))
    ok("dp is +x only", np.allclose(c["dp"][:, 1], 0.0))
    ok("drot == h deg", np.allclose(np.degrees(c["drot"]), 10.0))


def test_dual_bound():
    # construct candidates straddling the bound
    cand = dict(dp_mag=np.array([10., 49., 51., 30.]),
                drot=np.radians(np.array([5., 5., 5., 15.])))
    m = dual_bound_mask(cand, D_max=50.0, R_max_deg=12.0)
    ok("keep small disp+rot", m[0] == True)            # noqa: E712
    ok("keep disp just under D_max", m[1] == True)      # noqa: E712
    ok("reject disp over D_max", m[2] == False)         # noqa: E712 (SHOULD-FAIL)
    ok("reject rot over R_max", m[3] == False)          # noqa: E712 (SHOULD-FAIL)
    # lower bound (meaningful motion = enough disp OR enough rot)
    cand2 = dict(dp_mag=np.array([2., 2., 20., 2.]),
                 drot=np.radians(np.array([1., 8., 1., 1.])))
    m2 = dual_bound_mask(cand2, D_max=50.0, R_max_deg=12.0, D_min=15.0, R_min_deg=5.0)
    ok("reject tiny disp AND tiny rot (degenerate)", m2[0] == False)   # noqa: E712 SHOULD-FAIL
    ok("keep tiny disp but enough rot", m2[1] == True)                 # noqa: E712
    ok("keep enough disp, tiny rot", m2[2] == True)                    # noqa: E712
    ok("reject stay-put even within upper bound", m2[3] == False)      # noqa: E712 SHOULD-FAIL


def test_bucketize():
    dp = np.array([[10., 0.], [0., 10.], [-10., 0.]])          # E, N(+y), W
    drot = np.radians(np.array([8., -3., 0.]))
    dbin, mbin, rsign, rmbin = bucketize(dp, drot, n_dir=8,
                                         mag_edges=(0, 25, 50), rot_edges_deg=(0, 5, 12))
    ok("+x -> dir bin 0", dbin[0] == 0)
    ok("+y -> dir bin 2 (90deg)", dbin[1] == 2)
    ok("-x -> dir bin 4 (180deg)", dbin[2] == 4)
    ok("rot +8deg -> sign +1", rsign[0] == 1)
    ok("rot -3deg -> sign -1", rsign[1] == -1)
    ok("rot 0 -> sign 0", rsign[2] == 0)
    ok("|rot|=8deg in band 1 (>5)", rmbin[0] == 1)
    ok("|rot|=3deg in band 0 (<5)", rmbin[1] == 0)
    ok("mag 10px in band 0 (<25)", mbin[0] == 0)


def test_circ_corr():
    rng = np.random.RandomState(0)
    a = rng.uniform(-np.pi, np.pi, 500)
    ok("circ_corr(a,a)==1", abs(circ_corr(a, a) - 1.0) < 1e-6)
    b = rng.uniform(-np.pi, np.pi, 500)                # independent
    ok("circ_corr indep ~0", abs(circ_corr(a, b)) < 0.2)


# --- replay-fidelity guards (need env + real data) ---------------------------
def test_replay(data_path):
    print("\n[env] replay-fidelity guards on real pusht_noise:")
    data = load_split(data_path, "train")
    env = make_replay_env()
    # pick a few real trajectories long enough for h=16
    h = 16
    cand = extract_candidates(data, h, stride=50)
    rng = np.random.RandomState(0)
    samp = rng.choice(len(cand["i"]), size=min(40, len(cand["i"])), replace=False)
    full_pos, sub_pos = [], []
    for s in samp:
        t, i, j = int(cand["traj"][s]), int(cand["i"][s]), int(cand["j"][s])
        full_pos.append(replay_subsegment(env, data, t, i, j, "full")[0])
        sub_pos.append(replay_subsegment(env, data, t, i, j, "subseg")[0])
    full_pos, sub_pos = np.array(full_pos), np.array(sub_pos)
    print(f"  full-from-0 pos err: p50={np.percentile(full_pos,50):.2f} p90={np.percentile(full_pos,90):.2f}px")
    print(f"  subseg-from-i pos err: p50={np.percentile(sub_pos,50):.2f} p90={np.percentile(sub_pos,90):.2f}px")
    # SHOULD-PASS: full replay from true rest start reproduces recorded states tightly
    # (this validates env config + action indexing + determinism). Threshold generous
    # to absorb float/pymunk noise but tight enough to catch a wrong config.
    ok("full-from-0 replay reproduces states (p90<5px)", np.percentile(full_pos, 90) < 5.0)
    # SHOULD-FAIL: replaying ZERO actions must NOT reach a MOVED goal (sanity that the
    # test can detect non-reachability -- else 'reachable' is meaningless). Pick a
    # candidate where the block genuinely moves >20px, else the guard is vacuous.
    moves = cand["dp_mag"][samp]
    pick = samp[int(np.argmax(moves))]
    t, i, j = int(cand["traj"][pick]), int(cand["i"][pick]), int(cand["j"][pick])
    true_end = data["states5"][t, j, 2:4]
    zero_pred = replay_end_pose(env, state7(data, t, i), np.zeros((j - i, 2)))
    moved = float(np.linalg.norm(data["states5"][t, j, 2:4] - data["states5"][t, i, 2:4]))
    zero_err = float(np.linalg.norm(zero_pred[:2] - true_end))
    print(f"  zero-action (moving seg): block moved {moved:.1f}px in data, zero-action err {zero_err:.1f}px")
    ok("a moving sub-segment exists in sample (>20px)", moved > 20.0)
    ok("zero-action replay does NOT reach the moved goal", zero_err > 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_env", action="store_true")
    ap.add_argument("--data_path", default=os.environ.get("DATASET_DIR", ".") + "/pusht_noise")
    args = ap.parse_args()
    print("== pure-logic guards ==")
    test_wrap_pi()
    test_extractor_bounds()
    test_extractor_delta()
    test_dual_bound()
    test_bucketize()
    test_circ_corr()
    if args.with_env:
        test_replay(args.data_path)
    print(f"\nALL {P} guard assertions PASSED")


if __name__ == "__main__":
    main()
