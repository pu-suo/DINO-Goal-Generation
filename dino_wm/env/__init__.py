from gym.envs.registration import register

# pointmaze pulls in mujoco_py + d4rl at import time. Those are absent in the
# local Mac dev env (they live only on the GPU box). Make the import optional so
# the PushT path is usable everywhere; no behavior change where mujoco exists.
try:
    from .pointmaze import U_MAZE
    _HAS_POINTMAZE = True
except Exception as _e:  # noqa: BLE001
    import warnings
    warnings.warn(
        f"pointmaze unavailable ({type(_e).__name__}: {_e}); skipping its registration. "
        "This is expected on hosts without mujoco/d4rl (e.g. local Mac dev)."
    )
    _HAS_POINTMAZE = False

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
register(
    id="pusht_multicolor",
    entry_point="env.pusht.pusht_multicolor_wrapper:PushTMultiColorWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
if _HAS_POINTMAZE:
    register(
        id='point_maze',
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec':U_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
        }
    )
register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)