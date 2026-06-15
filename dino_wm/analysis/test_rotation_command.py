"""Guard tests for Option B Part 1 (rotation command) + Part 2 (success metric).
Every guard has a should-PASS and a should-FAIL case. Pure logic -- runs anywhere.
"""
import os, sys, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.rotation_command import (
    rotation_bucket, rotation_command_text, rotation_in_bucket, signed_drot_deg,
    all_buckets, bucket_name)
from metrics.rotation_goal_success import rotation_goal_success, command_from_states

P = 0
def ok(name, cond):
    global P
    assert cond, f"FAIL: {name}"
    P += 1
    print(f"  ok  {name}")


def st(ax, ay, bx, by, ang_deg):
    return np.array([ax, ay, bx, by, np.radians(ang_deg), 0.0, 0.0])


# --- Part 1: rotation command ------------------------------------------------
def test_buckets():
    ok("0deg -> (0,0) none", rotation_bucket(0) == (0, 0))
    ok("2.9deg -> none (below 3)", rotation_bucket(2.9) == (0, 0))
    ok("+5deg -> (+1,1) slight CW", rotation_bucket(5) == (1, 1))
    ok("-5deg -> (-1,1) slight CCW", rotation_bucket(-5) == (-1, 1))
    ok("+10deg -> (+1,2) moderate CW", rotation_bucket(10) == (1, 2))
    ok("-9deg -> (-1,2) moderate CCW", rotation_bucket(-9) == (-1, 2))
    ok("band edge 8deg -> moderate (>=8)", rotation_bucket(8) == (1, 2))
    ok("all_buckets has 5", len(all_buckets()) == 5)


def test_command_text():
    ok("none -> no-rotate phrase", rotation_command_text(1.0) == "without rotating it")
    ok("+6 -> slight clockwise", rotation_command_text(6) == "rotating it slight clockwise")
    ok("-10 -> moderate counterclockwise",
       rotation_command_text(-10) == "rotating it moderate counterclockwise")
    # SHOULD-FAIL guard: the command must carry NO numeric angle (banned).
    for d in (-11, -6, 0, 4, 9):
        ok(f"no digits in text for drot={d}", re.search(r"\d", rotation_command_text(d)) is None)


def test_command_matches_drot():
    # emitted command's bucket must equal the segment's actual signed Drot bucket
    for sa, ga in [(0, 6), (10, 4), (-5, -16), (30, 33)]:
        d = signed_drot_deg(np.radians(sa), np.radians(ga))
        ok(f"text bucket == drot bucket ({sa}->{ga})",
           rotation_bucket(d) == rotation_bucket(signed_drot_deg(np.radians(sa), np.radians(ga))))


# --- Part 2: success metric --------------------------------------------------
def test_self_consistency():
    start = st(100, 100, 200, 200, 10)
    goal = st(150, 150, 230, 250, 16)        # block moved + rotated +6deg (slight CW)
    cmd = command_from_states(start, goal)
    ok("command_from_states == slight CW", cmd == (1, 1))
    r = rotation_goal_success(goal[2:4], start[4], goal, cmd)
    ok("goal scores success under its own command", r["success"] is True)
    ok("  pos_ok at goal", r["pos_ok"] and r["pos_dist"] < 1e-6)
    ok("  rot_ok at goal", r["rot_ok"])


def test_in_out_bucket():
    start = st(100, 100, 200, 200, 10)
    goal = st(150, 150, 230, 250, 18)        # +8 -> moderate CW (1,2)
    cmd = command_from_states(start, goal)
    ok("goal +8 -> moderate CW", cmd == (1, 2))
    # SHOULD-PASS: cur rotated +9 (in moderate band) at goal pos
    cur_in = st(150, 150, 230, 250, 19)
    ok("in-band rotation passes rot_ok", rotation_goal_success(goal[2:4], start[4], cur_in, cmd)["rot_ok"])
    # SHOULD-FAIL: cur rotated +5 (slight band, wrong band)
    cur_band = st(150, 150, 230, 250, 15)
    ok("wrong-band rotation fails rot_ok", not rotation_goal_success(goal[2:4], start[4], cur_band, cmd)["rot_ok"])
    # SHOULD-FAIL: cur rotated -9 (right band, WRONG sign)
    cur_sign = st(150, 150, 230, 250, 1)
    ok("wrong-sign rotation fails rot_ok", not rotation_goal_success(goal[2:4], start[4], cur_sign, cmd)["rot_ok"])


def test_pos_rot_separable():
    """Spec Part 2c: position-pass / rotation-fail must be DETECTABLE (to read the
    grounding ablation), and vice versa."""
    start = st(100, 100, 200, 200, 10)
    goal = st(150, 150, 230, 250, 17)        # +7 slight CW (1,1)
    cmd = command_from_states(start, goal)
    # position-pass, rotation-FAIL (right place, wrong angle -- the interpretable fail)
    cur_pf = st(150, 150, 231, 251, 10)      # at goal pos, but no rotation (none)
    r1 = rotation_goal_success(goal[2:4], start[4], cur_pf, cmd)
    ok("pos-pass/rot-fail: pos_ok True", r1["pos_ok"])
    ok("pos-pass/rot-fail: rot_ok False", not r1["rot_ok"])
    ok("pos-pass/rot-fail: success False", not r1["success"])
    # position-FAIL, rotation-pass
    cur_rf = st(150, 150, 300, 320, 17)      # rotated right, but block far from goal pos
    r2 = rotation_goal_success(goal[2:4], start[4], cur_rf, cmd)
    ok("pos-fail/rot-pass: pos_ok False", not r2["pos_ok"])
    ok("pos-fail/rot-pass: rot_ok True", r2["rot_ok"])
    ok("pos-fail/rot-pass: success False", not r2["success"])


def test_shared_bucket_def():
    """Command (Part 1) and metric (Part 2) must use the SAME partition."""
    for sa, ga in [(0, 6), (20, 12), (-10, -25), (0, 1)]:
        s, g = st(0, 0, 0, 0, sa), st(0, 0, 0, 0, ga)
        cmd = command_from_states(s, g)
        d = signed_drot_deg(np.radians(sa), np.radians(ga))
        ok(f"metric & command share bucket ({sa}->{ga})", cmd == rotation_bucket(d))
        # and rotation_in_bucket agrees with rotation_bucket
        ok(f"in_bucket consistent ({sa}->{ga})", rotation_in_bucket(d, cmd))


def main():
    print("== Part 1: rotation command ==")
    test_buckets(); test_command_text(); test_command_matches_drot()
    print("== Part 2: success metric ==")
    test_self_consistency(); test_in_out_bucket(); test_pos_rot_separable(); test_shared_bucket_def()
    print(f"\nALL {P} guard assertions PASSED")


if __name__ == "__main__":
    main()
