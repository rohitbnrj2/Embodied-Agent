import math
from pathlib import Path
import pdb
import random
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

def set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def make_env(rank, seed, config_file, idx):
    """
    Utility function for multiprocessed env.

    :param env_str: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :param config_file: (file) config_file
    """

    def _init():
        env = BeeEnv(config_file=config_file, rendering_env= True if idx < 1 else False)
        env.seed(seed + rank)
        return env
    
    set_global_seeds(seed)
    return _init


class BeeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, config_file, rendering_env=True, force_set_env_rendering=False):
        """
        Experiment validates if compound eyes perform better than simple. 
        - fixed number of photoreceptors
        - agent can add an eye to itself of type 
        -  

        """
        super(BeeEnv, self).__init__()
        with open(config_file, "r") as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.Loader)
            self.cfg = Prodict.from_dict(self.cfg)

        self.sim = BeeSimulator(config_file)
        self.sim.init_maze('train')
        self.sim.init_animal(init_pos=None)

        self.episode_step = 0 # episode step
        self._timestep = 1
        self.rendering_env = rendering_env
        self.force_set_env_rendering = force_set_env_rendering
        if self.rendering_env: 
            print("Main Env that will render scenes...")
        
        # Action Space 
        action_space_size = 2 # velocity + direction
        if self.cfg.env_config.contininous_action_space:
            print("Using Continious Action Space")
            low = -1. * np.ones(action_space_size)
            high = np.ones(action_space_size)
            self.action_space = spaces.Box(low=low,
                                        high=high,
                                        dtype=np.float32)
        else:
            raise NotImplementedError()
            print("Using MultiBinary Action Space")

        print("Action Space size:", self.action_space.sample().shape)
        
        # self.velocity_range = [5,25]
        self.velocity_range = [2,10]
        # self.theta_range = [math.radians(15), math.radians(165.)] 
        self.theta_range = [math.radians(0), math.radians(180.)] 
        # self.theta_range = [0, np.pi]
        
        # Create Observation Space 
        self.obs_dim = self.cfg.animal_config.init_configuration.num_pixels #max_photoreceptors

        self.obs_size = self.cfg.env_config.observation_size * self.cfg.env_config.steps_per_measurment
        self.ra_size = int(self.obs_size/self.cfg.env_config.steps_per_measurment)

        print("obs size: {}, ra size: {}".format(self.obs_size, self.ra_size))
        if self.cfg.env_config.normalize_obs_max_obs_value:
            _obs_range = [0., 1.]
        else:
            _obs_range = [0., float(self.cfg.animal_config.max_photoreceptors)/float(self.obs_dim)]

        # all network inputs should be normalized between 0 and 1!
        obs_dict = {
            'intensity': spaces.Box(_obs_range[0], _obs_range[1], (self.obs_size, self.obs_dim), dtype=np.float32)
                    }
        
        if self.cfg.env_config.add_action_to_obs_space:
            # give lidar data, fourier data? 
            obs_dict['action_history'] = spaces.Box(0., 1., (self.ra_size, 2), dtype=np.float32)

        if self.cfg.env_config.add_pos_to_obs_space:
            obs_dict['position_history'] = spaces.Box(0., 1., (self.ra_size, 2), dtype=np.float32)

        if self.cfg.env_config.add_goal_pos_to_obs_space:
            obs_dict['goal_position'] = spaces.Box(0., 1., (2,), dtype=np.float32)

        # add animal config. 
        obs_dict['animal_config'] = spaces.Box(0., np.pi, (self.obs_dim, 3), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_dict)
        print("Observation Shape: {}".format(self.observation_space.shape))
        self.reset()

    def reset_sim(self,):

        # save the run with images 
        if self.sim.mode == 'test':
            print("Visualizing rollout... at {}".format(self._timestep))
            st = time.time()
            self.sim.render(current_canvas=False)
            tt = time.time()-st
            print("Visualization Time {}".format(tt))


        if self.force_set_env_rendering: 
            self.sim.init_maze(mode='test') # forcefully set to test mode (used for evaluation)
        else:
            if self.rendering_env and self._timestep % self.cfg.env_config.check_freq == 0:
                # print("Setting mode to test to visualize ")
                self.sim.init_maze(mode='test') # enable test mode to visualize rollout
            else:
                self.sim.init_maze(mode='train')

        # self.sim.init_maze(mode='train')

        ## reset simulator
        self.sim.reset_simulator()

    def reset(self,):
        # reset simulator & update 
        self.reset_sim()
        self._timestep +=1  
        
        self.episode_step = 0
        self.obs_intensity = deque(np.zeros((self.obs_size, self.obs_dim), 
                                            dtype = np.float32), 
                                            maxlen=self.obs_size) # should contain NON-normalized values
        self.animal_config = self.sim.animal.get_animal_encoding()
        self.obs_position_history = deque(np.zeros((self.ra_size,2), dtype = np.float32), maxlen=self.ra_size)
        self.obs_action_history = deque(np.zeros((self.ra_size,2), dtype = np.float32), maxlen=self.ra_size)
        self.obs_goal_position = np.array([self.sim.maze.goal_end_pos[0]/self.sim.maze.window_size[0], 
                                           self.sim.maze.goal_end_pos[1]/self.sim.maze.window_size[1]
                                          ])
        # self.obs_reward = np.zeros(1, dtype = np.float32)
        obs_space = {}
        obs_space['intensity'] = np.array(self.obs_intensity)
        obs_space['animal_config'] = np.array(self.animal_config)

        if self.cfg.env_config.add_pos_to_obs_space:
            obs_space['position_history'] = np.array(self.obs_position_history)

        if self.cfg.env_config.add_action_to_obs_space:
            obs_space['action_history'] = np.array(self.obs_action_history)
        
        if self.cfg.env_config.add_goal_pos_to_obs_space:
            obs_space['goal_position'] = np.array(self.obs_goal_position).reshape(-1)

        return obs_space 
    
    def _rescale_theta(self, theta):
        """
        Actions are normalized between -1 and 1, so we need to rescale
        """
        min_value = -1.0
        max_value = 1.0
        # print(actions)
        a = self.theta_range[0]
        b = self.theta_range[1]
        theta = (b-a) * (theta - min_value) / (max_value - min_value) + a
        return np.array(theta)

    def _rescale_velocity(self, velocity):
        """
        Actions are normalized between -1 and 1, so we need to rescale
        """
        min_value = -1.0
        max_value = 1.0
        # print(actions)
        a = self.velocity_range[0]
        b = self.velocity_range[1]
        velocity = (b-a) * (velocity - min_value) / (max_value - min_value) + a
        velocity = np.clip(velocity, a, b)
        return np.array(velocity)

    def _discrete_theta_map(self, theta):
        if theta == 0: 
            # action 0 is go down
            dx = 0 
            dy = 1 
            # return np.radians(90.)
        elif theta == 1: 
            # action 2 is down-left
            dx = -0.5
            dy = 0.5 
        elif theta == 2: 
            # action 2 is down-right
            dx = 0.5
            dy = 0.5 
        else: 
            raise NotImplementedError('{} not found'.format(theta))
        return dx, dy

    def _get_obs(self, action):

        collision = False 
        out_of_bounds = False
        
        # scale action 
        if self.cfg.env_config.contininous_action_space:
            p = int(self._rescale_velocity(action[0])) # p is velocity being controlled
            theta = self._rescale_theta(action[1])
            dx = p*np.cos(theta)
            dy = p*np.sin(theta)
            _aval = np.array([action[0], action[1]]).astype(np.float32)
        else:
            ap_size = action[0]
            dx, dy = self._discrete_theta_map(action[1])
            a1 = map_one_range_to_other(ap_size, 0., 1., 0, self.aperture_range[1]-1)
            a2 = map_one_range_to_other(action[1], 0., 1., 0, 2) # end is inclusive
            _aval = np.array([a1, a2]).astype(np.float32)
    
        if self.cfg.env_config.add_action_to_obs_space:
            self.obs_action_history.appendleft(_aval) # shift the deque & append new action 

        if self.cfg.env_config.add_pos_to_obs_space:
            _pos = np.array([float(self.sim.animal.x/self.sim.maze.window_size[0]),
                                float(self.sim.animal.y/self.sim.maze.window_size[1])]
                                ).astype(np.float32)
            self.obs_position_history.appendleft(_pos) # append new positoin 

        # save prev location since we take 10 steps in the same direction. 
        self.prev_sim_x = self.sim.animal.x
        self.prev_sim_y = self.sim.animal.y
        curr_obs = np.zeros((self.cfg.env_config.steps_per_measurment, self.obs_dim))

        # Get Observations for N steps 
        for i in range(self.cfg.env_config.steps_per_measurment):
            processed_eye_intensity, raw_eye_out , c, ob = self.sim.step(dx, dy)
            
            cur_intensities = []
            
            for ct, p_intensity in enumerate(processed_eye_intensity):
                cur_intensities.append(p_intensity)
                
            _pval = np.array(cur_intensities).astype(np.float32)
            curr_obs[i] = _pval
            # normalize position to be between 0 & 1
            if c: 
                collision = True
                break 
            if ob: 
                out_of_bounds = True
                break 

        curr_obs = np.array(curr_obs).astype(np.float32)
        # print("curr_obs --> ", curr_obs.shape, curr_obs)
        # normalize set of observations per n steps 
        if self.cfg.env_config.normalize_obs_max_obs_value:
            # print(curr_obs.max())
            curr_obs = self._noramlize_obs(curr_obs)
            # print(curr_obs.max())

        # last element of the deque() is the most recent observation
        for i in range(curr_obs.shape[0]):
            _pval = curr_obs[i]
            self.obs_intensity.appendleft(_pval) #  shift the deque & append new intensity 

        return collision, out_of_bounds

    def _reach_goal(self,):
        # _rect_x = self.sim.window_size[0]
        _rect_y = self.sim.maze.goal_end_pos[1]
        # should get close to the bottom!
        # _GOAL_THRESHOLD_ = _rect_y * 0.6 # 60 percent of the way there.
        _GOAL_THRESHOLD_ = _rect_y * 0.1 # 90 percent of the way there.
        if np.abs(self.sim.animal.y - _rect_y) < _GOAL_THRESHOLD_: 
            # if it is inside a 50 pixel radius
            return True
        return False

    def _noramlize_obs(self, obs_intensity, noramlize_joint=True):
        if noramlize_joint:
            # print(obs_intensity, obs_intensity.min(), obs_intensity.max())
            # pdb.set_trace()
            obs_intensity = map_one_range_to_other(obs_intensity, 
                                                   0., 1., 
                                                   obs_intensity.min(), 
                                                   obs_intensity.max())
        else:
            # noramlize left and right seperately 
            obs_intensity[:,0] = map_one_range_to_other(obs_intensity[:,0], 0, 1, 
                                                        obs_intensity[:,0].min(), 
                                                        obs_intensity[:,0].max())
            obs_intensity[:,1] = map_one_range_to_other(obs_intensity[:,1], 0, 1, 
                                                        obs_intensity[:,1].min(), 
                                                        obs_intensity[:,1].max())
        return obs_intensity

    def step(self, action):
        if self.cfg.env_config.use_random_actions:
            # print("Randomly Sampling Actions!") 
            action = self.action_space.sample()

        if action.shape[0] == 1:
            action = action[0]
            
        collision, out_of_bounds = self._get_obs(action)
        rg = self._reach_goal()
        obs_space = {}

        obs_space['intensity'] = np.array(self.obs_intensity)
        obs_space['animal_config'] = np.array(self.sim.animal.get_animal_encoding())
        
        # compute reward 
        # reward, done = self._sparse_reward(rg, collision, out_of_bounds)
        reward, done = self._fixed_n_steps(rg, collision, out_of_bounds)

        if self.cfg.env_config.add_pos_to_obs_space:
            obs_space['position_history'] = np.array(self.obs_position_history)        

        if self.cfg.env_config.add_action_to_obs_space:
            obs_space['action_history'] = np.array(self.obs_action_history)

        if self.cfg.env_config.add_goal_pos_to_obs_space:
            obs_space['goal_position'] = np.array(self.obs_goal_position)

        self.episode_step +=1 

        return obs_space , reward, done, {}

    def _fixed_n_steps(self, rg, collision, out_of_bounds):
        done = False
        reward = 0.
        if rg:
            # check rg first, if satisfied then dont need to check collision. 
            done = True 
            reward = 5.0 
        elif collision or out_of_bounds: 
            done = True 
            reward = -1.0
        else: 
            MAX_STEPS = self.cfg.env_config.max_steps_per_episode #+ int(self.sim.window_size[1]/self.cfg.env_config.steps_per_measurment)
            if not (MAX_STEPS == 0): 
                if self.episode_step > MAX_STEPS: 
                    done = True 
                    reward = -1.0
                    #reward = -1.0 * np.abs(self.sim.window_size[1] - self.sim.y)/self.sim.window_size[1]
            if not done:
                # reward = 0.1 # incentivizes staying alive
                #reward = 1.0 - np.abs(self.sim.window_size[1] - self.sim.y)/self.sim.window_size[1]
                
                if np.abs(self.sim.animal.y - self.prev_sim_y) > 5 or np.abs(self.sim.animal.x - self.prev_sim_x) > 5:
                     reward = 0.10 # incentivizes downward movement
                else:
                     reward = -0.01 # incentivizes downward movement; so it doesn't go left <-> right
                
                #reward += (self.episode_step - 20) #encourage to go in straight line so tak minimum steps!
                #reward = -1.0
        return reward, done

    def _sparse_reward(self, rg, collision, out_of_bounds):
        done = False
        reward = 0.
        if rg:
            # check rg first, if satisfied then dont need to check collision. 
            done = True 
            reward = 5.0 
        else: 
            if collision or out_of_bounds: 
                done = True 
                reward = -1.0
            else:
                # reward = 0.1 # incentivizes staying alive
                """
                if np.abs(self.sim.y - self.prev_sim_y) > 4:
                    reward = 0.15 # incentivizes downward movement
                else:
                    reward = -0.05 # incentivizes downward movement; so it doesn't go left <-> right
                """

                reward += (self.episode_step - 20) #encourage to go in straight line so tak minimum steps!
        return reward, done

if __name__ == "__main__": 
    print("_____________________Running Gym Environment_____________________")
    config_file = sys.argv[1]
    print("-> Config File path: {}".format(config_file))

    with open(config_file, "r") as ymlfile:
        dict_cfg = yaml.load(ymlfile, Loader=yaml.Loader)
        cfg = Prodict.from_dict(dict_cfg)

    logdir = Path("{}/{}".format(cfg.env_config.logdir, cfg.env_config.exp_name))
    if not logdir.exists():
        print("creating directory: {}".format(str(logdir)))
        Path.mkdir(logdir)

    ppo_path = Path("{}/{}/ppo/".format(cfg.env_config.logdir, cfg.env_config.exp_name))
    if not ppo_path.exists():
        print("creating directory: {}".format(str(ppo_path)))
        Path.mkdir(ppo_path)

    # save config file 
    target = str(Path.joinpath(logdir,'config.yml'))
    with open(target, 'w') as outfile:
        yaml.dump(dict_cfg, outfile, default_flow_style=False)

    # setup PPO
    # env = BeeEnv(config_file=config_file)
    # env.seed(42)

    env = SubprocVecEnv([make_env(rank=i, seed=cfg.env_config.seed, config_file=config_file, idx=i) 
                         for i in range(cfg.env_config.num_cpu)])
    env = VecMonitor(env, str(ppo_path))

    callback = SaveOnBestTrainingRewardCallback(check_freq=cfg.env_config.check_freq, log_dir=str(ppo_path), verbose=2)

    if cfg.env_config.load_from_ppo_checkpoint:
        # ------------ LOAD PPO MODEL FROM CHECKPOINT ------------
        print("Loading PPO from path: {}".format(cfg.env_config.ppo_checkpoint_path))
        model = PPO.load(cfg.env_config.ppo_checkpoint_path, env=env, print_system_info=True)
        # ------------ LOAD PPO MODEL FROM CHECKPOINT ------------
    else:
        print("Init policy from Scratch")
        policy_kwargs = dict(features_extractor_class=MultiInputFeatureExtractor,
                             features_extractor_kwargs=dict(features_dim = 256),)
        
        model = PPO("MultiInputPolicy",
                env,
                n_steps=cfg.env_config.n_steps,
                batch_size=cfg.env_config.batch_size,
                policy_kwargs=policy_kwargs,
                verbose=2)

    timesteps = 1e12
    start = time.time()

    print("Training now...")
    model.learn(total_timesteps=int(timesteps), callback=callback)
    save_path = "{}/{}".format(logdir, "final_rl_model.pt")
    model.save(save_path)

    print("_________Run Over________________")





