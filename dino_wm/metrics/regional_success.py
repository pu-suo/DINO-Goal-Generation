"""Regional success metric for LANGUAGE-conditioned PushT goals.

  *** PROPOSED in Part 0, PENDING REVIEW. Not yet wired into the planner. ***

Why the stock metric is wrong for this task
-------------------------------------------
The only success metric in the repo is the env's single-point pose gate
(``PushTWrapper.eval_state``, env/pusht/pusht_wrapper.py:62-81):

    success = (||cur_block_xy - goal_block_xy|| < 20 sim-px)  AND
              (|cur_angle - goal_angle|        < pi/9 = 20 deg)

That is correct when the goal is ONE reachable pose. It is WRONG for broad
language ("upper-right corner, rotated ~45 deg"), where the language names a
REGION of poses, not a point: g, trained on broad language, can only aim at a
*representative* pose in the region, so scoring it against one exact xy with a
20 sim-px (~0.55 of a 36-px DINO patch) tolerance under-counts every correct
in-region placement. (Coverage/IoU is even worse -- it needs sub-patch
precision and previously returned spurious 0.0.)

What this scores instead
------------------------
  position    : the achieved block ORIGIN falls in the SAME grid cell as the
                goal block origin, on a coarse N x N partition of the block
                workspace. Cells are ~100 sim-px ~= 2.7 DINO patches, so this is
                SUB-PATCH-FREE (never needs precision finer than a patch).
  orientation : |angle(cur) - angle(goal)| < ANG_TOL. Because the goal angle ==
                start angle + (the language's RELATIVE rotation), the start angle
                cancels and this IS exactly the relative-rotation check the
                language states. Set ANG_TOL ~= half the language rotation-bucket
                width so the tolerance matches the language granularity.

SHARED PARTITION (the load-bearing invariant)
---------------------------------------------
``REGION_BOUNDS`` / ``REGION_NCELLS`` define the partition ONCE. The language
generator (Part 1.7) MUST import ``block_cell``/``REGION_*`` from here to name
the region, so the metric's region and the language's region can never drift.
Verify in Part 0 review that this is the single source of truth.

Open item (flagged, not blocking the metric's soundness): the row<->"upper"/
"lower" NAME mapping depends on the render's y-up/y-down convention, which must
be confirmed against a real rendered frame in Part 1. Cell EQUALITY -- the
actual success test -- is naming-independent and sound regardless of that.
"""
import numpy as np

# --- the SHARED partition (metric <-> language generator) --------------------
# Block body-origin operating range in sim-512 coords. The env sampler draws
# block xy in [100, 400] (pusht_wrapper.py:30-52); gen_pusht_coord used
# [110, 402]. 3x3 -> 9 natural language regions of ~100 sim-px each.
REGION_BOUNDS = (100.0, 400.0)   # (lo, hi); same range for x and y
REGION_NCELLS = 3                # NxN grid (3 -> 9 regions)
ANG_TOL = np.pi / 9              # 20 deg (== stock gate; ~half a 40-deg bucket)

# Provisional names, col index 0..N-1 = x left->right, row index 0..N-1 = y.
# NOTE: which row is "upper" depends on the render y convention -- confirm in
# Part 1 against an actual frame before trusting these strings. Success uses
# the integer cell, not the name, so this does not affect correctness.
_COL_NAMES = ("left", "center", "right")
_ROW_NAMES = ("top", "middle", "bottom")   # provisional; verify y-up/down


def block_cell(xy, bounds=REGION_BOUNDS, ncells=REGION_NCELLS):
    """Integer (col, row) grid cell of a block origin xy in sim-512 coords.

    Out-of-range points clamp into the nearest edge cell (a block pushed past
    the operating range still belongs to the edge region, not to nowhere).
    """
    lo, hi = bounds
    step = (hi - lo) / ncells
    col = int(np.clip((float(xy[0]) - lo) // step, 0, ncells - 1))
    row = int(np.clip((float(xy[1]) - lo) // step, 0, ncells - 1))
    return (col, row)


def region_name(cell, ncells=REGION_NCELLS):
    """Provisional human name for a cell (only valid for the default 3x3)."""
    col, row = cell
    if ncells == 3:
        return f"{_ROW_NAMES[row]}-{_COL_NAMES[col]}"
    return f"col{col}_row{row}"


def angle_diff(a, b):
    """Minimal absolute circular difference (radians)."""
    d = np.abs(float(a) - float(b)) % (2 * np.pi)
    return float(np.minimum(d, 2 * np.pi - d))


def regional_success(goal_state, cur_state, bounds=REGION_BOUNDS,
                     ncells=REGION_NCELLS, ang_tol=ANG_TOL):
    """Regional pose success for a language goal.

    States are PushT state vectors [agent_x, agent_y, block_x, block_y, angle, ...].
    Position uses the block origin (cols 2:4); the manipulator is ignored (the
    goal-time pusher is not part of a language goal). Orientation uses col 4.

    Returns a dict mirroring env.eval_state plus the regional diagnostics.
    """
    goal_state = np.asarray(goal_state, dtype=np.float64)
    cur_state = np.asarray(cur_state, dtype=np.float64)
    goal_cell = block_cell(goal_state[2:4], bounds, ncells)
    cur_cell = block_cell(cur_state[2:4], bounds, ncells)
    pos_ok = (goal_cell == cur_cell)
    ad = angle_diff(cur_state[4], goal_state[4])
    ang_ok = ad < ang_tol
    return {
        "success": bool(pos_ok and ang_ok),
        "pos_ok": bool(pos_ok),
        "ang_ok": bool(ang_ok),
        "goal_cell": goal_cell,
        "cur_cell": cur_cell,
        "goal_region": region_name(goal_cell, ncells),
        "cur_region": region_name(cur_cell, ncells),
        "ang_diff_deg": float(np.degrees(ad)),
    }
