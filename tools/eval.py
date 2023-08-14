from pathlib import Path
import matplotlib.pyplot as plt
import os
import pdb
import random
import numpy as np
import sys
from prodict import Prodict
import yaml
from time import time
import shutil
#import torch
import gym
import time

from stable_baselines3 import PPO
from collections import deque 
from copy import deepcopy
# from cambrian.reinforce.env_v1 import BeeEnvV1
from cambrian.reinforce.env_v2 import BeeEnv as BeeEnvV2, set_global_seeds

def eval(ppo_path, config_file, overwrite_path, seed):

    print("Loading PPO from path: {}".format(ppo_path))
    model = PPO.load(ppo_path, print_system_info=True)    
    
    print("Running Simulation...")
    num_simulations = 1
    best_env = None
    best_reward = -1.0 * np.inf
    for i in range(num_simulations):
        sim_observations = []
        sim_actions = []
        sim_rewards = []
        
        done = False 
        env = BeeEnvV2(config_file, rendering_env=True, force_set_env_rendering=True)
        set_global_seeds(seed)
        env.seed(seed)
        obs = env.reset()
        
        total_reward = 0
        while not done: 
            action = model.predict(obs)[0]
            obs, reward, done, info = env.step(action)
            
            sim_observations.append(obs)
            sim_actions.append(action)
            sim_rewards.append(reward)
            
            total_reward += reward
        
        if total_reward > best_reward:
            # parse observations @Bhavya's code 
            
            best_reward = total_reward
            best_env = deepcopy(env)
            print("best reward: {}".format(best_reward))
            save_path = best_env.sim.logdir.joinpath("{}".format(overwrite_path))
            save_path = str(save_path)
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            #convert recorded information to graphs and save
            actions = {'velocity': [], 'theta':[]}
            rewards = []
            for obs,action,reward in zip(sim_observations, sim_actions, sim_rewards):
                intensities = obs['intensity']
                actions['velocity'] += [action[0] for j in range(intensities.shape[0])]
                actions['theta'] += [action[1] for j in range(intensities.shape[0])]
                
                rewards += [reward for j in range(intensities.shape[0])]
                
            total_steps = len(rewards)
            # for i in range(best_env.n_cams):
            #     plt.plot([j for j in range(total_steps)], intensity_recordings[i], label = f"Cam_{i}")
            #     plt.xlabel("Episode Timestep")
            #     plt.ylabel("Intensity")
            #     plt.legend()
            #     plt.title("Cam Intensity")
                
            
            plt.savefig(save_path + "/cam_intensity")
            plt.close()
            
            
            plt.plot([j for j in range(total_steps)], actions['velocity'])
            plt.xlabel("Episode Timestep")
            plt.ylabel("Velocity")
            plt.title("Velocity Graph")
            
            
            plt.savefig(save_path + "/velocity")
            plt.close()
           
            
            plt.plot([j for j in range(total_steps)], actions['theta'])
            plt.xlabel("Episode Timestep")
            plt.ylabel("Theta")
            plt.title("Theta Graph")
            
            
            plt.savefig(save_path + "/theta")
            plt.close()
            
            
            plt.plot([j for j in range(total_steps)], rewards)
            plt.xlabel("Episode Timestep")
            plt.ylabel("Episode Reward")
            plt.title("Reward Graph")
            
            
            plt.savefig(save_path + "/reward")
            plt.close()
            
    # visualize rollout 
    best_env.sim.render(current_canvas=False, overwrite_path=overwrite_path)

if __name__ == "__main__":
    ppo_path = sys.argv[1]
    config_path  = sys.argv[2]
    overwrite_path = sys.argv[3]
    print(len(sys.argv), sys.argv)
    if len(sys.argv[1:]) > 3:
        seed = sys.argv[4]
    else: 
        seed = 424242

    logdir = Path(overwrite_path)
    if not logdir.exists():
        print("creating directory: {}".format(str(logdir)))
        Path.mkdir(logdir)

    eval(ppo_path, config_path, overwrite_path, seed=seed)
