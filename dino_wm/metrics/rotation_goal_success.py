"""Option B (Part 2): success metric matched to the rotation-command granularity.

Success = BOTH
  position: ||block_final_pos - goal_block_pos|| < pos_tol (20px). Position is a
            SPECIFIC scene-determined target (the sub-segment's real end), so a
            point-tolerance is appropriate -- this is exactly why Option B sidesteps
            the bucket-vs-tolerance problem on position.
  rotation: the achieved relative rotation (block_final vs START) lands in the SAME
            (sign, magnitude-band) bucket as the command -- BUCKET MEMBERSHIP, because
            the rotation command is coarse. NOT a tight angle-to-a-specific-value
            tolerance; NOT coverage/IoU; NOT absolute cells.

pos_ok and rot_ok are returned separately so the grounding ablation can read a
position-pass / rotation-fail (the interpretable "right place, wrong angle").

The rotation bucket definition is imported from datasets.rotation_command -- the same
one Part 1 emits, so command and metric cannot drift.
"""
import numpy as np

from datasets.rotation_command import (
    signed_drot_deg, rotation_bucket, rotation_in_bucket)

POS_TOL = 20.0     # stock pusht position gate (px)


def rotation_goal_success(goal_block_xy, start_angle_rad, cur_state, command_bucket,
                          pos_tol=POS_TOL):
    """goal_block_xy: (2,) scene-determined target block position.
       start_angle_rad: block angle at the START (for achieved relative rotation).
       cur_state: full current state [agent_xy, block_xy, angle, ...].
       command_bucket: (sign, band) the rotation command g was given."""
    cur = np.asarray(cur_state, dtype=float)
    cur_xy, cur_ang = cur[2:4], cur[4]
    pos_ok = bool(np.linalg.norm(cur_xy - np.asarray(goal_block_xy, float)) < pos_tol)
    achieved = signed_drot_deg(start_angle_rad, cur_ang)
    rot_ok = bool(rotation_in_bucket(achieved, command_bucket))
    return {"success": bool(pos_ok and rot_ok), "pos_ok": pos_ok, "rot_ok": rot_ok,
            "achieved_drot_deg": achieved,
            "pos_dist": float(np.linalg.norm(cur_xy - np.asarray(goal_block_xy, float)))}


def command_from_states(start_state, goal_state):
    """Derive the rotation command bucket from a (start, goal) pair -- used by the
    oracle (goal is the real sub-segment end) so its command provably matches the goal."""
    s = np.asarray(start_state, float); g = np.asarray(goal_state, float)
    return rotation_bucket(signed_drot_deg(s[4], g[4]))
