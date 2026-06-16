"""Top-down rendering for Language Table, with a PushT-style end-effector DOT.

Three render modes (frame-mode contract):
  - mode='dot'   : start/rollout frames -- arm linkage HIDDEN, a compact white dot drawn
                   at the 2D end-effector position (physical contact radius). The pusher is
                   visible to DINO (localizable, maskable) but does NOT occlude blocks.
  - mode='clean' : goal frames -- arm hidden, NO dot (pusher-blind by construction).
  - mode='arm'   : DEBUG ONLY -- full xArm6 visible (occludes; not used in the pipeline).

Camera: straight-down NADIR, high mount (cam_z large => near-orthographic, low foreshorten),
framed tight to the workspace. True ortho renders blank under pybullet's headless TINY
renderer, so we use a high-mount narrow-fov perspective (near-ortho). At the table plane the
world->pixel map is affine:
    col = (1 - (y - cy)/H) / 2 * size      row = (1 - (x - cx)/H) / 2 * size
"""
import contextlib
import math

import cv2
import numpy as np
import pybullet

# Workspace (constants.py): X[0.15,0.6], Y[-0.3048,0.3048]
CENTER = (0.375, 0.0)
# half_extent 0.3048 == workspace y-half => frames the table tightly (full y, x has small margin)
DEFAULT_HALF_EXTENT = 0.3048
DEFAULT_CAM_Z = 4.0          # high mount => near-orthographic (low foreshortening)
EE_RADIUS_U = 0.0127         # end-effector cylinder contact radius (cylinder_real.urdf)
DOT_COLOR = (255, 255, 255)  # white: non-block, table-distinct, maskable


def topdown_camera(client, half_extent=DEFAULT_HALF_EXTENT, cam_z=DEFAULT_CAM_Z,
                   center=CENTER, znear=0.01, zfar=20.0):
    eye = [center[0], center[1], cam_z]          # directly above center
    target = [center[0], center[1], 0.0]         # looking straight down (nadir)
    up = [1.0, 0.0, 0.0]                          # world +x -> image up
    viewm = client.computeViewMatrix(eye, target, up)
    fov_deg = math.degrees(2.0 * math.atan(half_extent / cam_z))
    projm = client.computeProjectionMatrixFOV(fov_deg, 1.0, znear, zfar)
    return viewm, projm


def world_to_pixel(x, y, size, half_extent=DEFAULT_HALF_EXTENT, center=CENTER):
    col = (1.0 - (y - center[1]) / half_extent) / 2.0 * size
    row = (1.0 - (x - center[0]) / half_extent) / 2.0 * size
    return col, row


@contextlib.contextmanager
def effector_hidden(env):
    """alpha=0 the arm + end-effector body, restore on exit."""
    client = env.pybullet_client
    robot = env._robot
    bodies = [robot.xarm]
    ee = getattr(robot, "end_effector", None)
    if ee is not None:
        bodies.append(ee)
    snap = []
    for b in bodies:
        for entry in client.getVisualShapeData(b):
            link_index, rgba = entry[1], list(entry[7])
            snap.append((b, link_index, rgba))
            client.changeVisualShape(b, linkIndex=link_index, rgbaColor=list(rgba[:3]) + [0.0])
    try:
        yield
    finally:
        for b, link_index, rgba in snap:
            client.changeVisualShape(b, linkIndex=link_index, rgbaColor=rgba)


def _raw_render(env, size, half_extent, cam_z, center):
    client = env.pybullet_client
    viewm, projm = topdown_camera(client, half_extent, cam_z, center)
    out = client.getCameraImage(width=size, height=size, viewMatrix=viewm,
                                projectionMatrix=projm, renderer=pybullet.ER_TINY_RENDERER)
    return np.ascontiguousarray(np.array(out[2], np.uint8).reshape(size, size, 4)[:, :, :3])


def _mask_border(rgb, size, half_extent, center, dark_thresh=110):
    """Auto-detect the dark table band and fill the lighter lab backdrop (top/bottom rows)
    with the table color, so DINO patches aren't wasted on the high-frequency backdrop.
    Geometry-preserving (no resize/stretch). Per-row median is robust to the minority of
    block pixels in a row."""
    gray = rgb.mean(2)
    rows = np.where(np.median(gray, axis=1) < dark_thresh)[0]  # table rows are dark
    cols = np.where(np.median(gray, axis=0) < dark_thresh)[0]  # table cols are dark
    if len(rows) < size // 4 or len(cols) < size // 4:         # detection failed -> leave as-is
        return rgb
    top, bot = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    table = np.median(rgb[top:bot, left:right].reshape(-1, 3), axis=0).astype(np.uint8)
    rgb[:top] = table
    rgb[bot:] = table
    rgb[:, :left] = table
    rgb[:, right:] = table
    return rgb


def render_topdown(env, size=224, mode="dot", ee_xy=None, half_extent=DEFAULT_HALF_EXTENT,
                   cam_z=DEFAULT_CAM_Z, center=CENTER, mask_border=True):
    """mode in {'dot','clean','arm'}. 'dot' requires ee_xy (2D end-effector world pos)."""
    if mode == "arm":
        return _raw_render(env, size, half_extent, cam_z, center)
    with effector_hidden(env):
        rgb = _raw_render(env, size, half_extent, cam_z, center)
    if mask_border:
        rgb = _mask_border(rgb, size, half_extent, center)
    if mode == "clean":
        return rgb
    if mode == "dot":
        assert ee_xy is not None, "dot mode needs ee_xy"
        col, row = world_to_pixel(ee_xy[0], ee_xy[1], size, half_extent, center)
        r = max(int(round(EE_RADIUS_U * size / (2 * half_extent))), 3)
        cv2.circle(rgb, (int(round(col)), int(round(row))), r, DOT_COLOR, -1, cv2.LINE_AA)
        return rgb
    raise ValueError(mode)
