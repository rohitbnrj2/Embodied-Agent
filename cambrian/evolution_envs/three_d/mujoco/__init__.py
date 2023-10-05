from gym.envs.registration import register
from cambrian.evolution_envs.three_d.mujoco import maze_env


register(
    id='antmaze-umaze-v0',
    entry_point='cambrian.evolution_envs.three_d.mujoco.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'deprecated': True,
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval': True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)