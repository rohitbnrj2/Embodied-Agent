from copy import deepcopy
import math
from cambrian.evolution_envs.animal import OculozoicAnimal
from cambrian.utils.renderer_utils import visualize_rays
from pygame.locals import *
import numpy as np
import pygame
import sys
from pathlib import Path

from cambrian.renderer.maze import Maze
from cambrian.renderer.renderer_v1 import ApertureMask, Camera, TwoDRenderer
from cambrian.utils.utils import make_video
from prodict import Prodict
import yaml

import matplotlib.pyplot as plt
import time 
import pdb
import tqdm
import glob


class BeeSimulator:
    def __init__(self, config_file, cfg=None):
        if config_file is None: 
            # initilize directly from cfg dict instead of file
                if isinstance(cfg, dict):
                    self.cfg = Prodict.from_dict(cfg)
                else:
                    self.cfg = cfg
        else:
            with open(config_file, "r") as ymlfile:
                self.cfg = yaml.load(ymlfile, Loader=yaml.Loader)
                self.cfg = Prodict.from_dict(self.cfg)
        
        # self.window_size = self.cfg.sim_config.window_size # Width x Height in pixels
        self.logdir = Path("{}/{}".format(self.cfg.env_config.logdir, 
                                          self.cfg.env_config.exp_name))
        if not self.logdir.exists():
            print("creating directory: {}".format(str(self.logdir)))
            Path.mkdir(self.logdir)

        self.trial = 0 
        self.sim_states = self._reset_sim_states()

    def init_maze(self, mode='train'):
        # call reset & create an environment & animal that can sample the world 
        self.mode = mode
        self.maze = Maze(cfg=self.cfg.scene_config)

    def init_animal(self, init_pos):
        # call reset & create an environment & animal that can sample the world 
        self.animal = OculozoicAnimal(self.cfg.animal_config)
        if init_pos is None: 
            init_pos = self.maze.goal_start_pos
        self.animal.init_animal(init_pos[0], init_pos[1])

    def reset_simulator(self, bee_init_pos=None, reset_animal=False, reset_maze=False, 
                        randomize_colors=True, scene_config=None, animal_config=None): 
        # resets the environment
        if reset_maze:
            self.maze = Maze(cfg=scene_config)
        if randomize_colors:
            self.maze.randomize_wall_colors()

        # resets the animal config 
        if reset_animal:
            if animal_config is None: 
                raise ValueError("animal config cannot be none")
            self.animal = OculozoicAnimal(animal_config)

        if bee_init_pos is None: 
            bee_init_pos = self.maze.goal_start_pos

        self.animal.reset_position(bee_init_pos[0], bee_init_pos[1])
        self.sim_states = self._reset_sim_states()

    def step(self, dx, dy):
        # check collision
        processed_eye_intensity, eye_out = self.animal.observe_scene(dx, dy, self.maze)
        # pdb.set_trace()
        is_collision = self.maze.collision(self.animal.x, self.animal.y, self.cfg.sim_config.collision_threshold)
        out_of_bounds = self.maze.check_bounds(self.animal.x, self.animal.y)
        
        self._update_state_dict(self.animal.x, self.animal.y, 
                                processed_eye_intensity, eye_out, 
                                is_collision, out_of_bounds)
        return processed_eye_intensity, eye_out, is_collision, out_of_bounds

    def render(self, current_canvas=True, render_video=True, overwrite_path = None):
        if current_canvas: 
            self._visualize_rollout(start_idx=len(self.sim_states.positions_rollout)-1, 
                                    render_video=False, overwrite_path=overwrite_path)
        else: 
            self._visualize_rollout(start_idx=0, render_video=render_video, overwrite_path=overwrite_path)

    def _reset_sim_states(self, ):
        _dict =  {
            # 'scene_config': None, 
            # 'list_of_camera_configs': None,
            # 'geometry': [], # walls 
            'positions_rollout': [],
            'collision_rollout': [],
            'animal_obsevations_rollout': [],
            'animal_raw_obsevations_rollout': [],
            'animal_state_rollout': [],
        }
        return Prodict.from_dict(_dict)

    def _update_state_dict(self, x, y, processed_out, raw_out, is_collision, out_of_bounds):
        # update trial number
        self.trial += 1
        self.sim_states.positions_rollout.append([x,y]) # should be a list of [[x,y], ...]
        self.sim_states.collision_rollout.append(True if is_collision or out_of_bounds else False) # should be a list of [FFFT]
        self.curr_animal_obsevations = (processed_out, raw_out)
        if self.mode == 'test':
            # in train state there is no renderer states...
            self.sim_states.animal_obsevations_rollout.append(deepcopy(processed_out)) # should be a list of [[x,y], ...]
            self.sim_states.animal_raw_obsevations_rollout.append(deepcopy(raw_out)) # should be a list of [[x,y], ...]
            # self.sim_states.animal_state_rollout.append(deepcopy(render_dict)) # should be a list of [[x,y], ...]

    def _visualize_rollout(self, start_idx= 0, render_video=None, overwrite_path=None):
        if self.mode == 'train':
            print("Cannot Visualize in train mode. Switch to tests")
            return 
        # pygame simulator specifics 
        DEPTH = 32
        FLAGS = 0
        idx = start_idx
        if not self.cfg.sim_config.use_display:
            DISPLAY = (1, 1)
        else:
            DISPLAY = tuple(self.maze.window_size)
            display = pygame.Surface(self.maze.window_size)
            pygame.display.set_caption('RoboBee Simulator')

        screen = pygame.display.set_mode(DISPLAY, FLAGS, DEPTH)

        # save paths 
        if overwrite_path is None: 
            save_img_dir = self.logdir.joinpath("{}".format(self.trial))
        else: 
            save_img_dir = self.logdir.joinpath("{}".format(overwrite_path))
            
        save_img_dir.mkdir(exist_ok=True)
        while idx < len(self.sim_states.positions_rollout):
            work_surface = pygame.Surface(tuple(self.maze.window_size))
            work_surface.fill((self.cfg.sim_config.background_color))
            pos = self.sim_states.positions_rollout[idx]
            end = self.sim_states.collision_rollout[idx]
            if end:
                ct = self.cfg.sim_config.collision_threshold 
                bee_rect = pygame.Rect(self.animal.x - ct, self.animal.y, 2*ct, ct)
                pygame.draw.rect(work_surface, [255,0,0], bee_rect)

            # draw bee position 
            pygame.draw.circle(work_surface,(0,0,255), pos, 15)
            work_surface = self.maze.render(work_surface=work_surface)
            # save images 
            raw_eye_out = self.sim_states.animal_raw_obsevations_rollout[idx]
            processed_eye = self.sim_states.animal_obsevations_rollout[idx]
            work_surface = self.maze.render_intensity(work_surface, processed_eye)
            # import pdb; pdb.set_trace()
            if not self.cfg.sim_config.use_display:
                for raw_photoreceptor_output in raw_eye_out:
                    work_surface = visualize_rays(work_surface, raw_photoreceptor_output)
                save_img = save_img_dir.joinpath("sim_{:04d}.jpg".format(idx))
                pygame.image.save(work_surface, save_img)
                del work_surface
            else:
                # doesnt work :(
                raise NotImplementedError("On Display doesn't work, use video.")
                # work_surface = self.visualize_single_pass(work_surface, renderer_state)
                # clock.tick(60) 
            idx += 1 

        # save video of episode 
        if render_video:
            try:
                make_video("./{}/".format(str(save_img_dir)), 
                    sorted(glob.glob("{}/sim_*.jpg".format(str(save_img_dir)))))
                print("Visualizations saved at: {}".format(str(save_img_dir)))
            except Exception as e: 
                print("can't make video cause of {}".format(e))
        
if __name__ == "__main__":
    # expt_path = "./configs/v1.yaml"
    expt_path = sys.argv[1]
    sim = BeeSimulator(expt_path)
    sim.init_maze(mode='test')
    sim.init_animal(init_pos=None)
    print("num walls", len(sim.maze.walls))
    # simulate a trajectory of 100 steps going forward  
    num_steps = 10 #30 #00# 270
    p = 5

    st = time.time()
    for i in tqdm.tqdm(range(num_steps)):
        theta = np.pi/2 #np.random.uniform() * np.pi
        dx = p*np.cos(theta)
        dy = p*np.sin(theta)

        #####
        mut_type = np.random.choice(sim.animal.mutation_types)
        # mut_type = 'simple_to_lens'
        # if i >= 1: 
        #     mut_type = 'update_pixel'
        # if i >= 5: 
        mut_type = 'add_pixel'
        # mut_type = 'add_photoreceptor'

        if mut_type == 'add_photoreceptor':
            mut_args = None
        elif mut_type == 'simple_to_lens':
            mut_args = Prodict() 
            mut_args.pixel_idx = None

        elif mut_type == 'add_pixel':
            mut_args = Prodict() 
            mut_args.imaging_model = np.random.choice(['simple', 'lens'])
            mut_args.imaging_model = np.random.choice(['lens'])
            mut_args.direction = np.random.choice(['right'])
            # mut_args.direction = np.random.choice(['left', 'right'])
            mut_args.fov = 45.
            mut_args.sensor_size = 5.
            if mut_args.direction == 'left':
                mut_args.angle = None #np.random.uniform(0, 90.)
            elif mut_args.direction == 'right':
                mut_args.angle = None #np.random.uniform(110, 180.)
            else:
                raise ValueError("??")

        elif mut_type == 'update_pixel':
            mut_args = Prodict() 
            mut_args.pixel_idx = None # picks rangomly 
            mut_args.fov_r_update = math.radians(np.random.uniform(-10,10))
            mut_args.angel_r_update = math.radians(np.random.uniform(-10,10)) #math.radians(-10)
            mut_args.sensor_update = None

        # print('mutating animal with op: {}'.format(mut_type))
        # sim.animal.mutate(mut_type, mut_args=mut_args)
        # sim.animal.print_state()
        # sim.animal.save_animal_state(sim.logdir)

        for j in range (1):
        # for j in range (sim.cfg.env_config.steps_per_measurment):
            _, _, c, oob = sim.step(dx, dy) # go down 
            if c or oob:
                print("out of bounds: {}; collision: {}".format(oob, c))
                break
        if c or oob:
            break
        
        # break
    sim.render(current_canvas=False)
