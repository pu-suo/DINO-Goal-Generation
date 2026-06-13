"""Part-1 guard tests (non-render): each guard gets a SHOULD-FAIL and a SHOULD-PASS
case. Run locally (numpy/shapely, no pygame):

  /Users/Tom/miniforge3/envs/dino_wm_dev/bin/python analysis/test_rigid_guards.py
"""
import os, sys, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.rigid_goal_common import (
    rigid_matrix, assert_rigid, effective_matrix, apply_se2,
    traj_uses_wall, near_wall, traj_path_valid, assert_relative_geometry_preserved,
    make_language, relative_rotation_deg, block_extent, WALL_LO, WALL_HI, FRAME_CENTER,
)
from env.pusht.multicolor_common import tee_world_polygon, TEE_SCALE
from metrics.regional_success import block_cell

DEV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_devdata", "pusht_noise_val")
states = torch.load(os.path.join(DEV, "states.pth")).double().numpy()
seq = pickle.load(open(os.path.join(DEV, "seq_lengths.pkl"), "rb"))
print(f"loaded {len(seq)} real val trajs, states {states.shape}\n")

PASS = "PASS"; FAIL = "FAIL"
results = []
def check(name, cond):
    results.append((name, bool(cond)))
    print(f"  [{PASS if cond else FAIL}] {name}")

# ===== GUARD 1.2 -- rigid transform ONLY =====================================
print("GUARD 1.2  rigid transform (orthonormal, det+1, no scale/shear; shape preserved)")
rng = np.random.RandomState(0)
th = rng.uniform(-np.pi, np.pi); t = rng.uniform(-50, 50, 2)
try:
    assert_rigid(effective_matrix(th, t)); ok = True
except ValueError: ok = False
check("should-PASS: a real rotation+translation is accepted as rigid", ok)

M_scale = rigid_matrix(th, t).copy(); M_scale[:2, :2] *= 1.5
try: assert_rigid(M_scale); rej = False
except ValueError: rej = True
check("should-FAIL: a 1.5x SCALED matrix is rejected", rej)

M_shear = rigid_matrix(0.0, t).copy(); M_shear[0, 1] = 0.4  # shear
try: assert_rigid(M_shear); rej = False
except ValueError: rej = True
check("should-FAIL: a SHEARED matrix is rejected", rej)

# shape (area) + pusher<->block distance preserved by apply_se2
row = states[0, 0].copy()
row_tf = apply_se2(row, th, t)
a0 = tee_world_polygon((row[2], row[3], row[4]), scale=TEE_SCALE).area
a1 = tee_world_polygon((row_tf[2], row_tf[3], row_tf[4]), scale=TEE_SCALE).area
check(f"should-PASS: block AREA unchanged after transform ({a0:.1f}->{a1:.1f}, d={abs(a0-a1):.2e})",
      abs(a0 - a1) < 1e-6)
d0 = np.linalg.norm(row[0:2] - row[2:4]); d1 = np.linalg.norm(row_tf[0:2] - row_tf[2:4])
check(f"should-PASS: pusher<->block distance unchanged ({d0:.2f}->{d1:.2f})", abs(d0 - d1) < 1e-9)

# ===== GUARD 1.1 -- wall-use exclusion (original trajectory) =================
print("\nGUARD 1.1  wall-use exclusion (original rollout)")
# constructed should-FAIL: a frame with the block jammed against the right wall
bad = states[0, :1].copy(); bad[0, 2] = WALL_HI - 20; bad[0, 3] = 256; bad[0, 4] = 0.0
check("should-FAIL(excluded): block at right wall -> traj_uses_wall True",
      traj_uses_wall(bad, 1, margin=10.0))
# constructed should-PASS: a frame with block + pusher safely centered
good = states[0, :1].copy(); good[0, 0:2] = [256, 256]; good[0, 2:4] = [256, 300]; good[0, 4] = 0.0
check("should-PASS(kept): centered block+pusher -> traj_uses_wall False",
      not traj_uses_wall(good, 1, margin=10.0))
# how many of the REAL val trajs are wall-free at margin 10?
nfree = sum(not traj_uses_wall(states[i], seq[i], margin=10.0) for i in range(len(seq)))
print(f"   (real val: {nfree}/{len(seq)} trajectories are wall-free at margin=10px)")

# ===== GUARD 1.4 -- full-path validity of TRANSFORMED trajectory =============
print("\nGUARD 1.4  full-path validity (every frame, transformed)")
# pick a wall-free real traj as the base
base_i = next(i for i in range(len(seq)) if not traj_uses_wall(states[i], seq[i], 10.0))
base = states[base_i]; L = seq[base_i]
# should-FAIL: a huge translation swings the path out of bounds
big = apply_se2(base, 0.3, np.array([400.0, 400.0]))
okb, bad_t, reason = traj_path_valid(big, L)
check(f"should-FAIL: +400px translation -> path invalid (frame {bad_t}, {reason})", not okb)
# should-PASS: a small in-bounds transform keeps every frame valid
small = apply_se2(base, 0.2, np.array([10.0, -10.0]))
oks, _, _ = traj_path_valid(small, L)
check("should-PASS: small transform -> every frame valid", oks)
# relative geometry preserved on the transformed path
try:
    assert_relative_geometry_preserved(base, small, L); ok = True
except ValueError: ok = False
check("should-PASS: pusher<->block distance preserved at every frame (no impossible overlap)", ok)

# ===== GUARD 1.7 -- language matches the transformed goal pose ===============
print("\nGUARD 1.7  language <-> transformed goal pose")
th2, t2 = 1.1, np.array([-30.0, 40.0])
s0 = apply_se2(base[0], th2, t2); sT = apply_se2(base[L - 1], th2, t2)
lang = make_language(s0, sT)
print(f"   text: {lang['text']!r}")
check("should-PASS: language region == block_cell(goal)", lang["region_cell"] == block_cell(sT[2:4]))
check("should-PASS: language rotation == relative_rotation(start,goal)",
      abs(lang["rel_rot_deg"] - relative_rotation_deg(s0, sT)) < 1e-9)
# relative rotation PRESERVED from the original (reachability-by-construction)
rel_orig = relative_rotation_deg(base[0], base[L - 1])
check(f"should-PASS: rel-rotation preserved by transform (orig {rel_orig:.1f} == tf {lang['rel_rot_deg']:.1f})",
      abs(rel_orig - lang["rel_rot_deg"]) < 1e-6)
# should-FAIL: a WRONG language (rolled region) must NOT match the goal
wrong_cell = (block_cell(sT[2:4])[0] ^ 1, block_cell(sT[2:4])[1])  # flip a bit -> different cell
check("should-FAIL(mismatch detected): a wrong region != goal cell", wrong_cell != block_cell(sT[2:4]))

print("\n" + "=" * 60)
npass = sum(v for _, v in results)
print(f"GUARD TESTS: {npass}/{len(results)} passed")
sys.exit(0 if npass == len(results) else 1)
