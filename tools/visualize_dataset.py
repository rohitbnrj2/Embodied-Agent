import argparse
import gym

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='antmaze-umaze-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    
    dataset = env.get_dataset()
    if 'infos/qpos' not in dataset:
        raise ValueError('Only MuJoCo-based environments can be visualized')
    qpos = dataset['infos/qpos']
    qvel = dataset['infos/qvel']
    rewards = dataset['rewards']
    actions = dataset['actions']

    env.reset()
    env.set_state(qpos[0], qvel[0])
    for t in range(qpos.shape[0]):
        env.set_state(qpos[t], qvel[t])
        env.render()