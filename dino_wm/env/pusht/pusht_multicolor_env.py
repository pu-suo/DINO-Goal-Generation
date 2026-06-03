"""
Multi-color PushT environment.

Subclasses the frozen-physics `PushTEnv` and changes ONLY the visual layer +
goal bookkeeping:

* Renders N colored T-target *outlines* (decals): visual-only, added to neither
  the pymunk space nor collision handling -> block/pusher physics are byte-for-
  byte the original. This is what makes shipped-dynamics reuse plausible.
* All N targets are drawn every frame (start included), so the start image is
  identical across instructions for a given physical layout -> text is the only
  disambiguator.
* The *named* (active) target drives `goal_pose`, the env reward (coverage), and
  `eval_state` success. Which target is named is decided by the layout sampler
  (decorrelated from the block), NOT by anything visual.

Nothing here knows about the language model; the instruction string is carried
as a label only.
"""

import numpy as np
import pygame
import pymunk
import cv2

from .pusht_env import PushTEnv, DrawOptions
from .multicolor_common import tee_coverage, angle_diff, get_palette
from . import multicolor_sampler as mcs


class PushTMultiColorEnv(PushTEnv):
    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_size=224,
        reset_to_state=None,
        with_velocity=True,
        with_target=True,          # kept for signature compat; goal is drawn via targets
        n_targets=4,
        outline_thickness=7,       # 512-space px; ~3px at render_size=224
        success_threshold=0.95,    # coverage fraction for named-target success
        pos_range=mcs.DEFAULT_POS_RANGE,
        min_sep=mcs.DEFAULT_MIN_SEP,
        n_bins=3,
    ):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            with_velocity=with_velocity,
            with_target=with_target,
            shape="T",
        )
        self.n_targets = n_targets
        self.outline_thickness = outline_thickness
        self._success_threshold = success_threshold
        self.pos_range = tuple(pos_range)
        self.min_sep = min_sep
        self.n_bins = n_bins

        # layout state (set via set_layout / sample_and_set_layout / update_env)
        self.targets = None              # list of {'color','rgb','pose','bin'}
        self.active_target_idx = 0
        self.active_color = None
        self.instruction = None
        self.template_id = None

    # --- layout management ----------------------------------------------------
    def set_layout(self, layout):
        """Apply a layout dict (from multicolor_sampler.sample_layout)."""
        self.targets = [
            {
                "color": t["color"],
                "rgb": tuple(int(c) for c in t["rgb"]),
                "pose": np.asarray(t["pose"], dtype=np.float64),
                "bin": int(t.get("bin", -1)),
            }
            for t in layout["targets"]
        ]
        self.active_target_idx = int(layout["active_idx"])
        self.active_color = layout.get("active_color", self.targets[self.active_target_idx]["color"])
        self.instruction = layout.get("instruction")
        self.template_id = layout.get("template_id")
        self.goal_pose = self.targets[self.active_target_idx]["pose"].copy()
        return layout

    def sample_and_set_layout(self, seed, **kw):
        layout = mcs.sample_layout(
            seed,
            n_targets=self.n_targets,
            with_velocity=self.with_velocity,
            pos_range=self.pos_range,
            min_sep=self.min_sep,
            n_bins=self.n_bins,
            **kw,
        )
        self.set_layout(layout)
        return layout

    def update_env(self, env_info):
        """Vector-env hook. Carries a full layout dict (preferred) or {'shape':...}."""
        if isinstance(env_info, dict) and "targets" in env_info:
            self.set_layout(env_info)
        elif isinstance(env_info, dict) and "shape" in env_info:
            self.shape = env_info["shape"]

    # --- gym overrides --------------------------------------------------------
    def _setup(self):
        super()._setup()
        # success threshold drives reward clip in PushTEnv.step
        self.success_threshold = self._success_threshold
        if self.targets is not None:
            self.goal_pose = self.targets[self.active_target_idx]["pose"].copy()

    def reset(self):
        # standalone reset (e.g. bare gym.make().reset()): make sure a layout
        # exists and the block starts consistent with it.
        if self.targets is None:
            layout = self.sample_and_set_layout(self._seed if self._seed is not None else 0)
            if self.reset_to_state is None:
                self.reset_to_state = layout["init_state"]
        return super().reset()

    def _get_info(self):
        info = super()._get_info()
        info["active_color"] = self.active_color
        info["active_target_idx"] = self.active_target_idx
        info["instruction"] = self.instruction
        info["template_id"] = self.template_id
        if self.targets is not None:
            info["target_colors"] = [t["color"] for t in self.targets]
            info["target_poses"] = np.stack([t["pose"] for t in self.targets])
        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas
        draw_options = DrawOptions(canvas)

        # Draw N colored target OUTLINES (decals). Visual only.
        if self.targets is not None:
            for tgt in self.targets:
                body = self._get_goal_pose_body(tgt["pose"])
                for shape in self.block.shapes:
                    pts = [
                        pymunk.pygame_util.to_pygame(body.local_to_world(v), draw_options.surface)
                        for v in shape.get_vertices()
                    ]
                    pts += [pts[0]]
                    pygame.draw.polygon(canvas, tgt["rgb"], pts, width=self.outline_thickness)

        # Draw agent (pusher) + block on top.
        self.space.debug_draw(draw_options)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        return img

    # --- named-target success -------------------------------------------------
    def eval_state(self, goal_state, cur_state):
        """Success = coverage of the NAMED target's T-pose by the block.

        goal_state: full state whose block pose is the named target pose.
        cur_state:  achieved full state. Manipulator (agent) is ignored.
        """
        goal_pose = np.asarray(goal_state)[2:5]
        cur_pose = np.asarray(cur_state)[2:5]
        coverage = tee_coverage(goal_pose, cur_pose)
        pos_dist = float(np.linalg.norm(goal_pose[:2] - cur_pose[:2]))
        ang_dist = angle_diff(goal_pose[2], cur_pose[2])
        return {
            "success": bool(coverage >= self._success_threshold),
            "coverage": float(coverage),
            "block_pos_dist": pos_dist,
            "block_angle_dist": ang_dist,
            # reference: the original PushT pose criterion (block-only)
            "success_pose": bool(pos_dist < 20 and ang_dist < np.pi / 9),
            "state_dist": float(np.linalg.norm(np.asarray(goal_state) - np.asarray(cur_state))),
        }
