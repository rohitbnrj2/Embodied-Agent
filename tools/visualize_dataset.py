import argparse
from cambrian.evolution_envs.three_d.mujoco import register_envs
import gymnasium as gym
from PIL import Image

register_envs()
# export LD_PRELOAD=/usr/lib/libGL.so.1
# export MUJOCO_GL=egl
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='antmaze-umaze-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name, render_mode="rgb_array")

    NUM_ACTIONS = 5
    env.reset()
    for t in range(NUM_ACTIONS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rgb_array = env.render()
        Image.fromarray(rgb_array).save(f"./3d_debug/rgb_array_large_{t}.png")