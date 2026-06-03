import numpy as np

from utils import aggregate_dct
from .pusht_multicolor_env import PushTMultiColorEnv


class PushTMultiColorWrapper(PushTMultiColorEnv):
    """gym entry point for multi-color PushT.

    Mirrors PushTWrapper's planning/rollout helpers (prepare / step_multiple /
    rollout) so it is a drop-in for plan.py's SubprocVectorEnv, and overrides
    goal sampling so a "random goal" places the block at the NAMED target.
    eval_state / update_env are inherited from PushTMultiColorEnv.
    """

    def __init__(self, with_velocity=True, with_target=True, n_targets=4, **kwargs):
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target,
            n_targets=n_targets,
            **kwargs,
        )
        self.action_dim = self.action_space.shape[0]

    def sample_random_init_goal_states(self, seed):
        """Sample + install a decorrelated layout; goal = block at named target.

        Side effect: sets the env's target layout so a subsequent prepare()
        renders the same N decals in both the init and goal frames.
        """
        layout = self.sample_and_set_layout(seed)
        init_state = layout["init_state"].copy()
        goal_state = init_state.copy()
        goal_state[2:5] = layout["goal_pose"]  # relocate block to the named target
        return init_state, goal_state

    def prepare(self, seed, init_state):
        """Reset with a controlled init_state (targets must already be set)."""
        self.seed(seed)
        self.reset_to_state = init_state
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        obses, rewards, dones, infos = [], [], [], []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
