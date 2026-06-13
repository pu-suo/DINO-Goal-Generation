"""Rigid-transform goal construction for LANGUAGE-conditioned PushT (Part 1).

Core idea: take a REAL pusht_noise trajectory (reachable by construction -- it
physically happened), apply ONE rigid SE(2) transform (rotation + translation,
NO scale/shear) to the object poses at EVERY timestep, and re-render on the
FIXED scene. Rigid motion preserves the trajectory's relative start->goal motion,
so the transformed goal is reachable with the SAME difficulty as the original
rollout, while its absolute pose is randomized.

This module holds the geometry + the TESTED guards (the failure modes the task
enumerates). Rendering / encoding live in the generator script; everything here
is numpy/shapely and unit-testable without pygame.

KEY INVARIANT (why the guards are cheap): a rigid transform preserves ALL
*relative* geometry -- block shape, pusher<->block contact, the start->goal
relative motion. So the ONLY things a transform can break are ABSOLUTE:
out-of-bounds and wall-collision. Pusher<->block "impossible overlap" cannot
arise from a rigid transform of a valid trajectory (asserted as a sanity check).

Scene constants come from PushTEnv._setup / add_tee / add_circle (verified):
  walls: 4 segments forming the box [5, 506]^2           (pusht_env.py:765-771)
  block: scale-30 T (add_tee hardcodes scale=30)          (pusht_env.py:817-856)
  pusher: kinematic circle, radius 15                      (pusht_env.py:774,798)
State row = [agent_x, agent_y, block_x, block_y, block_angle] in sim-512.
"""
import numpy as np
import shapely.geometry as sg

from env.pusht.multicolor_common import tee_world_vertices, tee_world_polygon, TEE_SCALE
from metrics.regional_success import block_cell, region_name, angle_diff  # SHARED partition

# --- scene constants (sim-512) ----------------------------------------------
WALL_LO, WALL_HI = 5.0, 506.0      # wall-box segment coords (pusht_env.py:765-771)
PUSHER_R = 15.0                    # add_circle radius (pusht_env.py:774)
FRAME_CENTER = np.array([256.0, 256.0])


# --- SE(2) -------------------------------------------------------------------
def rigid_matrix(theta, t):
    """3x3 homogeneous SE(2): rotation by theta (rad) about origin, then +t."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, t[0]], [s, c, t[1]], [0.0, 0.0, 1.0]])


def assert_rigid(M, tol=1e-9):
    """Raise unless M[:2,:2] is orthonormal with det +1 (no scale, no shear)."""
    R = np.asarray(M)[:2, :2]
    orth_err = float(np.abs(R @ R.T - np.eye(2)).max())
    det = float(np.linalg.det(R))
    if orth_err > tol or abs(det - 1.0) > tol:
        raise ValueError(f"not a rigid SE(2): orth_err={orth_err:.2e}, det={det:.8f}")
    return True


def effective_matrix(theta, t, center=FRAME_CENTER):
    """The SE(2) actually applied by apply_se2 (rotate about `center`, then +t),
    as a single 3x3 so callers can assert_rigid it."""
    center = np.asarray(center, float)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    t_eff = center - R @ center + np.asarray(t, float)
    return rigid_matrix(theta, t_eff)


def apply_se2(states, theta, t, center=FRAME_CENTER):
    """Apply SE(2) to a (T,5) (or (5,)) state array. Positions (agent xy, block xy)
    rotate about `center` then translate by t; block angle += theta; pusher has no
    angle. Returns the transformed states (same shape)."""
    s_in = np.asarray(states, dtype=np.float64)
    single = s_in.ndim == 1
    S = np.atleast_2d(s_in).copy()
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    center = np.asarray(center, float)

    def tf(p):
        return (p - center) @ R.T + center + np.asarray(t, float)

    S[:, 0:2] = tf(S[:, 0:2])           # agent (pusher)
    S[:, 2:4] = tf(S[:, 2:4])           # block origin
    S[:, 4] = S[:, 4] + theta           # block angle
    return S[0] if single else S


# --- geometry helpers --------------------------------------------------------
def block_vertices(pose):
    """(8,2) world vertices of the scale-30 T at pose=(x,y,theta)."""
    v1, v2 = tee_world_vertices((float(pose[0]), float(pose[1]), float(pose[2])), scale=TEE_SCALE)
    return np.concatenate([v1, v2], axis=0)


def block_extent(pose):
    """Axis-aligned (xmin, ymin, xmax, ymax) of the block T. Exact for wall/bounds
    checks: the union-of-two-rects bbox == bbox of its 8 vertices."""
    V = block_vertices(pose)
    return float(V[:, 0].min()), float(V[:, 1].min()), float(V[:, 0].max()), float(V[:, 1].max())


def _state_block_pose(row):
    return (row[2], row[3], row[4])


def near_wall(row, margin):
    """True if the block OR pusher comes within `margin` of any wall (box [5,506])."""
    bx0, by0, bx1, by1 = block_extent(_state_block_pose(row))
    ax, ay = row[0], row[1]
    pe0, pe1 = ax - PUSHER_R, ax + PUSHER_R
    pf0, pf1 = ay - PUSHER_R, ay + PUSHER_R
    lo, hi = WALL_LO + margin, WALL_HI - margin
    block_near = (bx0 < lo) or (by0 < lo) or (bx1 > hi) or (by1 > hi)
    pusher_near = (pe0 < lo) or (pf0 < lo) or (pe1 > hi) or (pf1 > hi)
    return bool(block_near or pusher_near)


# --- GUARD 1.1: wall-use exclusion (original trajectory) ---------------------
def traj_uses_wall(states, seq_len, margin=10.0):
    """Guard 1.1. True if the block/pusher came within `margin` of a wall during
    the ORIGINAL rollout (the wall may have been load-bearing -> exclude)."""
    for t in range(int(seq_len)):
        if near_wall(states[t], margin):
            return True
    return False


# --- GUARD 1.4: full-path validity of the TRANSFORMED trajectory -------------
def block_in_bounds(row, pad=0.0):
    bx0, by0, bx1, by1 = block_extent(_state_block_pose(row))
    return (bx0 >= WALL_LO + pad and by0 >= WALL_LO + pad
            and bx1 <= WALL_HI - pad and by1 <= WALL_HI - pad)


def pusher_in_bounds(row, pad=0.0):
    ax, ay = row[0], row[1]
    return (ax - PUSHER_R >= WALL_LO + pad and ay - PUSHER_R >= WALL_LO + pad
            and ax + PUSHER_R <= WALL_HI - pad and ay + PUSHER_R <= WALL_HI - pad)


def traj_path_valid(states, seq_len, pad=0.0, require_pusher=True):
    """Guard 1.4. Check EVERY frame of the (transformed) path: block + pusher in
    bounds, block not crossing a wall. Returns (ok, first_bad_frame, reason)."""
    for t in range(int(seq_len)):
        if not block_in_bounds(states[t], pad):
            return False, t, "block_out_of_bounds"
        if require_pusher and not pusher_in_bounds(states[t], pad):
            return False, t, "pusher_out_of_bounds"
    return True, -1, ""


def assert_relative_geometry_preserved(orig, transformed, seq_len, tol=1e-6):
    """Sanity: a rigid transform preserves pusher<->block distance at every frame
    (so 'impossible overlap' cannot be introduced). Raises on violation."""
    o, x = np.asarray(orig), np.asarray(transformed)
    do = np.linalg.norm(o[:seq_len, 0:2] - o[:seq_len, 2:4], axis=1)
    dx = np.linalg.norm(x[:seq_len, 0:2] - x[:seq_len, 2:4], axis=1)
    err = float(np.abs(do - dx).max())
    if err > tol:
        raise ValueError(f"rigid transform changed pusher<->block distance by {err:.2e}")
    return True


# --- GUARD 1.7: language from the transformed goal pose ----------------------
_CW_NAMES = {1: "clockwise", -1: "counterclockwise"}


def relative_rotation_deg(start_row, goal_row):
    """Signed relative block rotation start->goal in degrees, wrapped to (-180,180].
    Preserved by the rigid transform (both angles get +theta), so it equals the
    ORIGINAL trajectory's relative rotation -> reachable by construction."""
    d = (float(goal_row[4]) - float(start_row[4]) + np.pi) % (2 * np.pi) - np.pi
    return float(np.degrees(d))


def make_language(start_row, goal_row):
    """Build the (text, region_cell, rel_rot_deg) spec from the TRANSFORMED start
    & goal. Region uses the SHARED block_cell partition; rotation is the signed
    relative rotation. Returns a dict; the metric/region and language can't drift
    because both call block_cell()."""
    goal_xy = (goal_row[2], goal_row[3])
    cell = block_cell(goal_xy)
    rel = relative_rotation_deg(start_row, goal_row)
    sign = 1 if rel >= 0 else -1
    mag = abs(rel)
    if mag < 5:
        rot_phrase = "without rotating it"
    else:
        rot_phrase = f"rotating it about {int(round(mag))} degrees {_CW_NAMES[sign]}"
    text = f"Push the T to the {region_name(cell)} region, {rot_phrase}."
    return {"text": text, "region_cell": cell, "region_name": region_name(cell),
            "rel_rot_deg": rel}
