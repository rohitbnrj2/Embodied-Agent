import argparse
from cambrian.evolution_envs.three_d.mujoco import register_envs
import gymnasium as gym
from PIL import Image
import numpy as np

register_envs()
# export LD_PRELOAD=/usr/lib/libGL.so.1 && export MUJOCO_GL=egl
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# this works for rendering instead: 
# export PYOPENGL_PLATFORM=osmesa && export MUJOCO_GL=osmesa && export MUJOCO_EGL_DEVICE_ID=0
if __name__ == "__main__":
    import os 
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["DISPLAY"] = ":0"
    os.environ["MUJOCO_GL"]="osmesa"
    os.environ['LD_PRELOAD']="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env_name', type=str, default='antmaze-lmazediversegr-v0') #antmaze-umaze-v0')
    parser.add_argument('--env_name', type=str, default='antmaze-umaze-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name, render_mode="depth_array")

    NUM_ACTIONS = 5
    env.reset()
    for t in range(NUM_ACTIONS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.ant_env.render_mode = "depth_array"
        arr = env.render()
        np.save(f"./3d_debug/depth_array_{t}.npy", arr)
        env.ant_env.render_mode = "rgb_array"
        arr = env.render()
        Image.fromarray(arr).save(f"./3d_debug/rgb_array_large_{t}.png")