# +
import functools
import csv
import numpy as np
import random
import yaml
from copy import deepcopy
from prodict import Prodict
from env_v2 import make_env, BeeEnv
import os

import math
from pathlib import Path
import pdb
import random
import sys
from cambrian.reinforce.models import MultiInputFeatureExtractor
from cambrian.utils.renderer_utils import map_one_range_to_other
from cambrian.utils.rl_utils import SaveOnBestTrainingRewardCallback
import numpy as np
import sys
from prodict import Prodict
import yaml
from time import time
import shutil
from tools.bee_sim import BeeSimulator
import torch
import gym
import time

from gym import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from collections import deque 

# +
project_path = "/home/gridsan/bagrawalla/EyesOfCambrian-main/"
maze_folder_path = project_path + "new_mazes_paint/"

#train generation[i] in maze[i]
generation_wise_mazes = ["right_turn_constant_width.png",
                         "right_turn_constant_width.png",
                         "right_turn_constant_width.png",
                         "right_turn_changing_width.png",
                        "right_turn_changing_width.png",
                        "right_turn_changing_width.png",]
init_animal_config_path = project_path + "configs_evo/debug.yaml"
# -


class TestEvoRun:
    """
    TestEvoRun is a very simple evo run, where n agents mutate into kn children through asexual 
    after every evolution epoch and the best n children are selected for further mutation. 
    DERL is more complex than this.
    """
    
    def __init__(self, 
                 init_animal_config_path, 
                 generation_wise_mazes,
                 init_population_size = 3,
                 best_agents_per_generation = 3,
                 trainsteps_per_agent = 50,
                 num_mutations_per_agent = 2, 
                 max_population_size = 6,
                 max_generations = 6):
        
                
        # initialize init_population many zeroth generation animals.
        #self.population stores agent_config_file and its id
        
        self.population = [(init_animal_config_path,i) for i in range(init_population_size)]
        
        self.max_population_size = max_population_size
        self.max_generations = max_generations
        self.trainsteps_per_agent = trainsteps_per_agent
        self.num_mutations_per_agent = num_mutations_per_agent
        self.generation_wise_mazes = generation_wise_mazes
        self.best_agents_per_generation = best_agents_per_generation
        
        #some constraints for the current codebase
        assert len(self.generation_wise_mazes) == max_generations
        assert init_population_size == best_agents_per_generation
        assert best_agents_per_generation*num_mutations_per_agent == max_population_size
    
    def save_prodict_as_yaml(self, prodict, yaml_file_path):
        #saves given prodict into the specified yaml file path
        
        logdir = yaml_file_path[:-10] #"config.yml" is 10 characters
        
        if not os.path.exists(logdir):
            print("creating directory: {}".format(str(logdir)))
            os.makedirs(logdir)
        
        with open(yaml_file_path, "w") as ymlfile:
            yaml.dump(prodict, ymlfile)
    
    def mutate_single_agent(self, agent, agent_id, maze_path, evo_epoch, num_mutations = 2):
        
        # mutate a single agent through asexual reproduction, this also kills the parent. Puts the new agent in 
        # specified maze_path
        
        children = []
        
        for i in range(num_mutations):
            
            child_cfg = self.create_random_symmetric_mutation(agent, maze_path)
            child_cfg_save_path = "{}/{}/evo_epoch = {}/agent_id = {}/config.yml".format(child_cfg.env_config.logdir, 
                                                                                         child_cfg.env_config.exp_name, 
                                                                                         evo_epoch,
                                                                                         f"{agent_id}_{i}")
            self.save_prodict_as_yaml(child_cfg, child_cfg_save_path)
            
            children.append((child_cfg_save_path, f"{agent_id}_{i}"))
            
            
            
        
        return children
    
    def create_random_symmetric_mutation(self, agent, maze_path):
        
        #adds or subtracts pixels symmetrically and randomly from agent
        
        allowed_angles = [10.*i for i in range(18)] #allowed angles for pixel placement on either side
        
        max_pixels = 2*len(allowed_angles)
     
        with open(agent, "r") as ymlfile:
                agent_dict_cfg = yaml.load(ymlfile, Loader=yaml.Loader)
                agent_cfg = Prodict.from_dict(agent_dict_cfg)
        
        
        num_parent_pixels = agent_cfg.animal_config.init_configuration.num_pixels
        
        parent_fov = agent_cfg.animal_config.init_configuration.fov
        parent_sensor_size = agent_cfg.animal_config.init_configuration.sensor_size
        parent_direction = agent_cfg.animal_config.init_configuration.direction
        parent_imaging_model = agent_cfg.animal_config.init_configuration.imaging_model
        
        parent_angles = agent_cfg.animal_config.init_configuration.angle 
        
        
        child_cfg = deepcopy(agent_cfg)
        
        if num_parent_pixels == max_pixels:
            mutation_type = "subtract_pixel"
        
        elif num_parent_pixels <= 2: #have at least 2 pixels!
            mutation_type = "add_pixel"
        
        else:
            mutation_type = random.choice(["add_pixel", "subtract_pixel"])
        
        
        child_angles = self.get_child_angles(parent_angles, allowed_angles, mutation_type)
        
        child_cfg.scene_config.load_path = maze_path
        
        child_cfg.animal_config.init_configuration.angle = deepcopy(child_angles)
        
        child_cfg.animal_config.init_configuration.fov = parent_fov + parent_fov[-2:]
        child_cfg.animal_config.init_configuration.sensor_size = parent_sensor_size + parent_sensor_size[-2:]
        child_cfg.animal_config.init_configuration.direction = parent_direction + parent_direction[-2:]
        child_cfg.animal_config.init_configuration.imaging_model = parent_imaging_model + parent_imaging_model[-2:]
        
        child_cfg.animal_config.init_configuration.num_pixels = len(child_angles)
        
        
        return child_cfg
        
    
    def get_child_angles(self, parent_angles, allowed_angles, mutation_type):
        
        pixels_each_side = int(len(parent_angles)/2)
                
        if mutation_type == "add_pixel":
            
            permuted_angles = np.random.permutation(allowed_angles)
            
            for angle in permuted_angles:
                if angle not in parent_angles:
                    
                    child_angles = parent_angles + [angle, angle] #once for both left and right
                    
            
            
            
        
        elif mutation_type == "subtract_pixel":
            
            remove_index = random.choice([i for i in range(pixels_each_side)])
            
            remove_element = parent_angles[2*remove_index]
            
            child_angles = deepcopy(parent_angles)
            
            child_angles.remove(remove_element) #once for left
            child_angles.remove(remove_element) #again for right
            
            
        
        else:
            raise NotImplementedError
        
        return child_angles
        
        
    def mutate_current_generation(self, evo_epoch):
        
        #mutates each agent in the current generation through asexual reproductions
        
        maze_path = maze_folder_path + self.generation_wise_mazes[evo_epoch]
        
        next_generation = []
        
        for agent, agent_id in self.population:
            
            children = self.mutate_single_agent(agent, agent_id, maze_path, evo_epoch, self.num_mutations_per_agent)
            
            next_generation += children
        
        self.population = next_generation
        
    
    def train_single_agent(self, agent, agent_id, evo_epoch, trainsteps_per_agent = 1e4):
        
        #trains and saves single agent in the evo_epoch folder
        
        with open(agent, "r") as ymlfile:
                agent_dict_cfg = yaml.load(ymlfile, Loader=yaml.Loader)
                cfg = Prodict.from_dict(agent_dict_cfg)
                
        
        print("____________________________________________________________________")
        print("")
        print(f"Training agent with agent_id = {agent_id}")
        print("")
        print("_____________________________________________________________________")
        
        logdir = Path("{}/{}/evo_epoch = {}/agent_id = {}/".format(cfg.env_config.logdir, 
                                                                   cfg.env_config.exp_name, 
                                                                   evo_epoch,
                                                                   agent_id))
        if not logdir.exists():
            print("creating directory: {}".format(str(logdir)))
            Path.mkdir(logdir)
        
        
        ppo_path = Path("{}/{}/evo_epoch = {}/agent_id = {}/ppo/".format(cfg.env_config.logdir, 
                                                                         cfg.env_config.exp_name, 
                                                                         evo_epoch,
                                                                         agent_id))
        
        if not ppo_path.exists():
            print("creating directory: {}".format(str(ppo_path)))
            Path.mkdir(ppo_path)
            

        env = SubprocVecEnv([make_env(rank=i, seed=cfg.env_config.seed, config_file= agent, idx=i) 
                            for i in range(cfg.env_config.num_cpu)])
        env = VecMonitor(env, str(ppo_path))

        callback = SaveOnBestTrainingRewardCallback(check_freq=cfg.env_config.check_freq, log_dir=str(ppo_path), verbose=2)
        
        print("Init policy from Scratch")
      
        model = PPO("MultiInputPolicy",
                env,
                n_steps=cfg.env_config.n_steps,
                batch_size=cfg.env_config.batch_size,
                verbose=2)
        
        start = time.time()

        print("Training now...")
        model.learn(total_timesteps= self.trainsteps_per_agent, callback=callback)
        save_path = "{}/{}".format(logdir, "final_rl_model.pt")
        model.save(save_path)
        
        
        
    
    def train_new_generation(self, evo_epoch = 0):
        print("_____________________________________________________")
        print(f"Starting evo_epoch = {evo_epoch}")
        print("_____________________________________________________")
        
        for agent, agent_id in self.population:
            self.train_single_agent(agent, agent_id, evo_epoch, self.trainsteps_per_agent)
            
            
    def parse_reward_file(self, reward_file, num_last_episodes = 1, ignore = 2):
        
        x = [0]
        y = []
        with open(reward_file, 'r') as csvfile:
            
            episodes = csv.reader(csvfile, delimiter = ',')
            count = 0
            for ep in episodes:
                if count < ignore:
                    count += 1
                    continue
                    
                a,b = ep[0], ep[1]
                x.append(x[-1] + float(b))
                y.append(float(a))
                
        return np.mean(y[-1*num_last_episodes : ])
                
                
    
    def compare_agents(self, agent1, agent2):
        
        cfg_path_1, _ = agent1
        cfg_path_2, _ = agent2
        
        rew_file_1 = cfg_path_1[: -10] + "ppo/monitor.csv" #"config.yml" is 10 characters
        rew_file_2 = cfg_path_2[: -10] + "ppo/monitor.csv"
        
        rew1 = self.parse_reward_file(rew_file_1)
        rew2 = self.parse_reward_file(rew_file_2)
        
        if rew1 > rew2:
            return 1
        
        elif rew1 < rew2:
            return -1
        
        else:
            return 0
            
    
    def select_fittest_agents(self,):
        
        #select top best_agents_per_generation agents
        
        
        self.population = sorted(self.population, key=functools.cmp_to_key(self.compare_agents), reverse = True)
        
        self.population = self.population[: self.best_agents_per_generation]
    
        
        
    
            
    def run(self,):
        
        #basic evo algorithm!
        print("..........................Starting New Evolutionary Run....................................")
        for evo_epoch in range(self.max_generations):
            
            self.mutate_current_generation(evo_epoch)
            
            self.train_new_generation(evo_epoch)
            
            self.select_fittest_agents()     

if __name__ == "__main__":
    evolution = TestEvoRun(init_animal_config_path,
                           generation_wise_mazes)
    evolution.run()







