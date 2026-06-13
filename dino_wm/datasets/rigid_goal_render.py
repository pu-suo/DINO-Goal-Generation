"""Render helpers for rigid-transform goals (Part 1, guards 1.3 & 1.5).

Renders a state on the FIXED PushT scene via the env's own state->image path
(env.prepare -> _set_state -> _render_frame), NOT by warping an image -- so the
walls are redrawn from the fixed space every frame and stay pixel-identical
(guard 1.3 holds by construction; the test confirms it). Green-T removal uses
with_target=False, which fills the goal polygon with White == the (255,255,255)
background (aliased fill, no halo) -> a residual-free clean scene (guard 1.5).

Headless-safe: rgb_array rendering uses a pygame.Surface (no display); we still
set SDL_VIDEODRIVER=dummy so it never tries to open a window.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
from env.pusht.pusht_wrapper import PushTWrapper

LIGHTGREEN = np.array([144, 238, 144])   # pygame.Color("LightGreen")


def make_env(with_target=False, render_size=224, with_velocity=True):
    """PushT env on the canonical scene. with_target=False -> green goal-T removed."""
    return PushTWrapper(with_velocity=with_velocity, with_target=with_target,
                        render_size=render_size)


def render_state(env, state5, seed=0):
    """Render a (5,) [agent_xy, block_xy, angle] state -> (H,W,3) uint8 image.
    Pads zero velocity if the env expects it. Returns (img, realized_state)."""
    s = np.asarray(state5, dtype=np.float64).ravel()
    if getattr(env, "with_velocity", False) and s.shape[0] == 5:
        s = np.concatenate([s, [0.0, 0.0]])
    obs, realized = env.prepare(seed, s)
    return np.asarray(obs["visual"]), realized


def green_pixel_count(img, tol=40):
    """# pixels within L-inf `tol` of LightGreen (goal-T residual detector)."""
    d = np.abs(img.astype(np.int32) - LIGHTGREEN[None, None, :]).max(axis=2)
    return int((d <= tol).sum())


def border_ring(img, w=6):
    """Outer w-pixel ring (walls + background for an interior-block frame)."""
    a = img.copy()
    inner = a[w:-w, w:-w].copy()
    a[w:-w, w:-w] = 0
    return a, inner
