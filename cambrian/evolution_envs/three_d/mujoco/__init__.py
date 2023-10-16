from gymnasium.envs.registration import register
from cambrian.evolution_envs.three_d.mujoco import maze


def register_envs():
    register(
        id='antmaze-umaze-v0',
        entry_point='cambrian.evolution_envs.three_d.mujoco.ant_v4_maze:AntMazeEnv',
        max_episode_steps= 700,
        kwargs={
            'maze_map': maze.U_MAZE,
            'reward_type':'sparse',
        }
    )
    
    register(
        id='antmaze-lmazediversegr-v0',
        entry_point='cambrian.evolution_envs.three_d.mujoco.ant_v4_maze:AntMazeEnv',
        max_episode_steps= 700,
        kwargs={
            'maze_map': maze.LARGE_MAZE_DIVERSE_GR,
            'reward_type':'sparse',
        }
    )

    print("Registration complete...")