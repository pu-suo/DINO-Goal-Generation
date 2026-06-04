"""
Shared geometry / palette / coverage helpers for the multi-color PushT testbed.

Kept dependency-light (numpy + shapely only) so it can be imported host-side
(samplers, dataset generation, planning workspace) WITHOUT pulling in
pygame/pymunk. The env module imports from here too.

Geometry note: the T-block coverage here is computed analytically from the
block *pose* (x, y, theta), reproducing exactly the vertices and rotation
convention used by `PushTEnv.add_tee` (scale=30) and pymunk's
`body.local_to_world` (v.rotated(theta) = (x*cos - y*sin, x*sin + y*cos)).
This makes our named-target coverage identical in spirit to the env's own
LightGreen-goal coverage, just evaluated against the *named* target's pose and
ignoring the manipulator.
"""

import numpy as np
import shapely.geometry as sg
from shapely.ops import unary_union

# --- T-block geometry (must match PushTEnv.add_tee at scale=30) ---------------
TEE_SCALE = 30
_LENGTH = 4

# Horizontal bar of the T (vertices1 in add_tee) and vertical stem (vertices2).
_TEE_RECT1 = np.array([
    (-_LENGTH * TEE_SCALE / 2, TEE_SCALE),
    (_LENGTH * TEE_SCALE / 2, TEE_SCALE),
    (_LENGTH * TEE_SCALE / 2, 0),
    (-_LENGTH * TEE_SCALE / 2, 0),
], dtype=np.float64)
_TEE_RECT2 = np.array([
    (-TEE_SCALE / 2, TEE_SCALE),
    (-TEE_SCALE / 2, _LENGTH * TEE_SCALE),
    (TEE_SCALE / 2, _LENGTH * TEE_SCALE),
    (TEE_SCALE / 2, TEE_SCALE),
], dtype=np.float64)


# --- Color palette ------------------------------------------------------------
# High-saturation, mutually distinct, and (deliberately) distinct from the
# RoyalBlue pusher (65,105,225), LightSlateGray block, and white background.
# Targets are drawn as hollow T-OUTLINES (not filled), so the only SOLID shape
# in the frame is the gray block (the object to push) -- this keeps block-vs-
# target unambiguous for the model, and at the goal the block sits visibly
# INSIDE the named outline (both stay visible). Saturated colors read clearly as
# thin outlines. Grounding probe (Phase 0.3) validates separability.
DEFAULT_PALETTE = [
    ("red", (220, 30, 30)),
    ("green", (30, 160, 60)),
    ("blue", (30, 90, 230)),
    ("yellow", (235, 200, 25)),
    ("magenta", (200, 40, 180)),
    ("orange", (240, 130, 20)),
    ("cyan", (30, 190, 200)),
    ("purple", (120, 50, 190)),
]


def get_palette(n):
    """Return the first n (name, rgb) entries; error if n exceeds the palette."""
    if n > len(DEFAULT_PALETTE):
        raise ValueError(
            f"n_targets={n} exceeds palette size {len(DEFAULT_PALETTE)}; "
            "add more colors to DEFAULT_PALETTE."
        )
    return DEFAULT_PALETTE[:n]


# --- Pose / coverage math -----------------------------------------------------
def _rotate(verts, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return verts @ R.T


def tee_world_polygon(pose, scale=TEE_SCALE):
    """Shapely polygon for a T-block at pose=(x, y, theta) in sim (512) coords."""
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    if scale != TEE_SCALE:
        f = scale / TEE_SCALE
        rect1, rect2 = _TEE_RECT1 * f, _TEE_RECT2 * f
    else:
        rect1, rect2 = _TEE_RECT1, _TEE_RECT2
    t = np.array([x, y])
    poly1 = sg.Polygon(_rotate(rect1, theta) + t)
    poly2 = sg.Polygon(_rotate(rect2, theta) + t)
    return unary_union([poly1, poly2])


def tee_world_vertices(pose, scale=TEE_SCALE):
    """The two T-rectangles' world vertices (each (4,2)) at pose=(x,y,theta).

    Same convention as tee_world_polygon; used to rasterize the T (block or a
    target outline) into the DINOv2 input image for the grounding/pose probes.
    """
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    f = scale / TEE_SCALE
    t = np.array([x, y])
    return [_rotate(_TEE_RECT1 * f, theta) + t, _rotate(_TEE_RECT2 * f, theta) + t]


def tee_coverage(goal_pose, cur_pose, scale=TEE_SCALE):
    """Fraction of the goal-T area covered by the current-T (the PushT metric)."""
    goal_poly = tee_world_polygon(goal_pose, scale)
    cur_poly = tee_world_polygon(cur_pose, scale)
    goal_area = goal_poly.area
    if goal_area <= 0:
        return 0.0
    return float(goal_poly.intersection(cur_poly).area / goal_area)


def angle_diff(a, b):
    """Minimal absolute circular difference between angles a, b (radians)."""
    d = np.abs(a - b) % (2 * np.pi)
    return float(np.minimum(d, 2 * np.pi - d))


def pusher_patch_mask(agent_xy, img=196, patch=14, sim=512, radius_sim=15, pad=0.6):
    """(P,) manipulator mask for the CEM energy: 0 at the pusher's patches, 1 else.

    `g` (and the oracle) can't know the arm's goal-time position, so those patches
    are dropped from the planning cost. agent_xy is in sim (512) coords.
    """
    grid = img // patch
    ax = agent_xy[0] * img / sim
    ay = agent_xy[1] * img / sim
    r = radius_sim * img / sim + pad * patch
    mask = np.ones(grid * grid, dtype=np.float32)
    for ri in range(grid):
        for ci in range(grid):
            cx, cy = (ci + 0.5) * patch, (ri + 0.5) * patch
            if (cx - ax) ** 2 + (cy - ay) ** 2 <= r * r:
                mask[ri * grid + ci] = 0.0
    return mask


def tee_centroid_offset(scale=TEE_SCALE):
    """(dx, dy) from the T's body origin to its area centroid in local coords.

    Useful for "nearest target" decorrelation checks that should compare to the
    visual center of mass rather than the body origin.
    """
    poly = tee_world_polygon((0.0, 0.0, 0.0), scale)
    c = poly.centroid
    return float(c.x), float(c.y)


def contact_pusher_pose(block_start_xy, block_goal_pose, pusher_r=15.0, gap=2.0,
                        scale=TEE_SCALE):
    """Plausible, *reachable* goal-time pusher position for a fabricated goal.

    Places the pusher on the trailing contact side of the block-at-goal (the side
    it was pushed FROM), just OUTSIDE the rotated T silhouette so the pusher circle
    never overlaps the solid T. Unlike the naive ``behind`` placement
    (``block_origin - offset*unit(d)``), this accounts for the block heading theta
    via the T's support distance along the trailing direction, so it is a valid
    end-of-push state for EVERY pose. (The naive ``behind`` ignores theta and so
    puts the pusher circle inside/overlapping the T for ~62% of uniformly-rotated
    goals -- a physically impossible config that corrupts the oracle goal latent.)

    Args:
        block_start_xy: (2,) block body-origin position at the start.
        block_goal_pose: (3,) block (x, y, theta) at the goal/named target.
        pusher_r: pusher circle radius in sim (512) px (PushTEnv add_circle=15).
        gap: extra clearance (px) between the pusher circle and the T silhouette.
    Returns:
        (2,) world pusher position, or None when the block barely translates
        (no well-defined push direction -> caller should keep the real/start pose).
    """
    start = np.asarray(block_start_xy, dtype=np.float64)
    goal = np.asarray(block_goal_pose, dtype=np.float64)
    d = goal[:2] - start
    n = float(np.linalg.norm(d))
    if n < 1e-3:
        return None
    u = -d / n  # trailing direction (opposite the block's motion)
    verts = np.concatenate(tee_world_vertices(goal, scale), axis=0)  # (8, 2) world
    h = float(np.max((verts - goal[:2]) @ u))  # T support distance from origin along u
    return goal[:2] + (h + pusher_r + gap) * u
