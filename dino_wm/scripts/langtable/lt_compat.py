"""Minimal tf_agents / env-spec compatibility shim for Google Language Table.

The Language Table env + scripted oracles are pure numpy/pybullet, but the oracle
base class imports `tf_agents` (only for the PyPolicy base + PolicyStep + an unused
time_step module) and expects a tf_agents-style env (TimeStep from reset/step,
time_step_spec/action_spec). Installing the real (2022, rotted) tf_agents/TF stack
is unnecessary. This module:

  1. install_tf_agents_shim() -> injects a tiny fake `tf_agents.*` into sys.modules
     BEFORE the oracle is imported, so the import + PyPolicy machinery work.
  2. GymToTFAgentsEnv -> wraps a raw LanguageTable gym env so reset()/step() return
     a TimeStep with is_first()/is_last(), while delegating compute_state(),
     get_control_frequency(), render(), succeeded, action_space to the raw env.

Verified against language_table.environments.oracles.{oriented_push_oracle,
push_oracle_rrt_slowdown} (DINO-WM Language Table port, 2026-06-15).
"""
import collections
import sys
import types

import numpy as np


# --- TimeStep / StepType (tf_agents.trajectories.time_step analogue) ---
class StepType:
    FIRST = 0
    MID = 1
    LAST = 2


class TimeStep:
    def __init__(self, step_type, reward, discount, observation):
        self.step_type = step_type
        self.reward = reward
        self.discount = discount
        self.observation = observation

    def is_first(self):
        return self.step_type == StepType.FIRST

    def is_mid(self):
        return self.step_type == StepType.MID

    def is_last(self):
        return self.step_type == StepType.LAST


PolicyStep = collections.namedtuple("PolicyStep", ["action", "state", "info"])
PolicyStep.__new__.__defaults__ = ((), ())


class PyPolicy:
    """Minimal stand-in for tf_agents.policies.py_policy.PyPolicy."""

    def __init__(self, time_step_spec=None, action_spec=None, *args, **kwargs):
        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

    def action(self, time_step, policy_state=(), seed=None):
        return self._action(time_step, policy_state, seed)


def install_tf_agents_shim():
    """Register a minimal fake tf_agents in sys.modules. Idempotent.

    Must be called BEFORE importing the language_table oracle modules.
    """
    if "tf_agents" in sys.modules and getattr(
        sys.modules["tf_agents"], "_lt_shim", False):
        return

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    ta = _mod("tf_agents")
    ta._lt_shim = True
    pol = _mod("tf_agents.policies", py_policy=None)
    pp = _mod("tf_agents.policies.py_policy", PyPolicy=PyPolicy)
    pol.py_policy = pp
    ta.policies = pol
    traj = _mod("tf_agents.trajectories")
    ps = _mod("tf_agents.trajectories.policy_step", PolicyStep=PolicyStep)
    tsmod = _mod(
        "tf_agents.trajectories.time_step",
        StepType=StepType,
        TimeStep=TimeStep,
        restart=lambda obs: TimeStep(StepType.FIRST, 0.0, 1.0, obs),
        transition=lambda obs, reward, discount=1.0: TimeStep(
            StepType.MID, reward, discount, obs),
        termination=lambda obs, reward: TimeStep(StepType.LAST, reward, 0.0, obs),
    )
    traj.policy_step = ps
    traj.time_step = tsmod
    ta.trajectories = traj


class GymToTFAgentsEnv:
    """Wrap a raw LanguageTable gym env to the tf_agents-style API the oracle needs."""

    def __init__(self, env):
        self._env = env
        self.last_obs = None

    def reset(self):
        obs = self._env.reset()
        self.last_obs = obs
        return TimeStep(StepType.FIRST, 0.0, 1.0, obs)

    def step(self, action):
        obs, reward, done, info = self._env.step(np.asarray(action, np.float32))
        self.last_obs = obs
        self._last_info = info
        step_type = StepType.LAST if done else StepType.MID
        discount = 0.0 if done else 1.0
        return TimeStep(step_type, reward, discount, obs)

    # --- delegated to the raw env ---
    def compute_state(self):
        return self._env.compute_state()

    def get_control_frequency(self):
        f = getattr(self._env, "get_control_frequency", None)
        return f() if callable(f) else 10.0

    def render(self, *a, **k):
        return self._env.render(*a, **k)

    def time_step_spec(self):
        return None

    def action_spec(self):
        return None

    @property
    def succeeded(self):
        return getattr(self._env, "succeeded", False)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def raw(self):
        return self._env

    def __getattr__(self, name):
        # Delegate any other attribute (e.g. pybullet_client) to the wrapped env.
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)
