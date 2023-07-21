from copy import deepcopy
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
                self.cfg = Prodict.from_dict(cfg)
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

    def init_maze(self, scene_config, mode='train'):
        # call reset & create an environment & animal that can sample the world 
        self.mode = mode
        self.maze = Maze(cfg=scene_config)

    def init_animal(self, animal_config, init_x, init_y):
        # call reset & create an environment & animal that can sample the world 
        self.animal = OculozoicAnimal(animal_config)
        self.animal.init_animal(init_x, init_y)

    def reset_simulator(self, bee_init_pos, reset_animal=False, reset_maze=False, randomize_colors=True, scene_config=None, animal_config=None): 
        # resets the environment
        if reset_maze:
            self.maze = Maze(cfg=scene_config)
        if randomize_colors:
            self.maze.randomize_wall_colors()

        # resets the animal config 
        if reset_animal:
            self.animal = OculozoicAnimal(animal_config)

        self.animal.reset_position(bee_init_pos[0], bee_init_pos[1])
        self.sim_states = self._reset_sim_states()

    def _reset_sim_states(self, ):
        _dict =  {
            # 'scene_config': None, 
            # 'list_of_camera_configs': None,
            # 'geometry': [], # walls 
            'positions_rollout': [],
            'collision_rollout': [],
            'animal_obsevations_rollout': [],
            'animal_state_rollout': [],
            'cam_intensities' : {f"cam_{i}":[] for i in range(self.n_cams)}
        }
        return Prodict.from_dict(_dict)

    def update_state_dict(self, x, y, eye_obs, is_collision, out_of_bounds):
        # update trial number
        self.trial += 1
        self.sim_states.positions_rollout.append([x,y]) # should be a list of [[x,y], ...]
        self.sim_states.collision_rollout.append(True if is_collision or out_of_bounds else False) # should be a list of [FFFT]
        self.curr_animal_obsevations = eye_obs
        if self.mode == 'test':
            # in train state there is no renderer states...
            self.sim_states.animal_obsevations_rollout.append(deepcopy(eye_obs)) # should be a list of [[x,y], ...]
            # self.sim_states.animal_state_rollout.append(deepcopy(render_dict)) # should be a list of [[x,y], ...]

    def step(self, dx, dy):
        self.x += dx 
        self.y += dy

        # print("Time per render all cameras {}".format(tt))
        # print("length of cameras: ", self.renderer.cameras)
        # print("length of walls: ", len(self.maze.walls))

        # check collision
        is_collision = self.maze.collision(self.x, self.y, self.cfg.sim_config.collision_threshold)
        out_of_bounds = self.maze.check_bounds(self.x, self.y)
        
        eye_out = self.animal.observe_scene(dx, dy, self.maze)
        self.update_state_dict(self.x, self.y, eye_out, is_collision, out_of_bounds)
        return eye_out, is_collision, out_of_bounds

    def render(self, current_canvas=True, render_video=True, overwrite_path = None):
        if current_canvas: 
            self._visualize_rollout(start_idx=len(self.sim_states.positions_rollout)-1, 
                                    render_video=False, overwrite_path=overwrite_path)
        else: 
            self._visualize_rollout(start_idx=0, render_video=render_video, overwrite_path=overwrite_path)

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
            self.display = pygame.Surface(self.maze.window_size)
            pygame.display.set_caption('RoboBee Simulator')

        self.screen = pygame.display.set_mode(DISPLAY, FLAGS, DEPTH)
        self.font = pygame.font.SysFont('Arial', 20)

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
                bee_rect = pygame.Rect(self.x - ct, self.y, 2*ct, ct)
                pygame.draw.rect(work_surface, [255,0,0], bee_rect)

            # draw bee position 
            pygame.draw.circle(work_surface,(0,0,255), pos, 15)
            # save images 
            photoreceptor_output = self.sim_states.animal_obsevations_rollout[idx]
            if not self.cfg.sim_config.use_display:
                work_surface = visualize_rays(work_surface, photoreceptor_output)
                save_img = save_img_dir.joinpath("sim_{:04d}.jpg".format(idx))
                pygame.image.save(work_surface, save_img)
                del work_surface
            else:
                # doesnt work :(
                raise NotImplementedError("On Display doesn't work, use video.")
                # work_surface = self.visualize_single_pass(work_surface, renderer_state)
                # clock.tick(60) 
            idx += 1 

        # last plot the intensities as graph
        """
        xs = range(0, len(self.sim_states.camera_left_intensity))
        y1 = self.sim_states.camera_left_intensity
        y2 = self.sim_states.camera_right_intensity 
        # pdb.set_trace()
        plt.plot(xs, y1, label="camera left intensity")
        plt.plot(xs, y2, label="camera right intensity")
        plt.legend()
        plt.savefig("{}/camera_intensity_plot.png".format(str(save_img_dir)))
        # save signal as dict
        _save = {'left': y1, 'right': y2, 'xs':xs}
        save("{}/camera_intensity".format(str(save_img_dir)), _save)
        """
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
    sim.init(mode='test')
    start_pos_x = sim.maze.goal_start_pos[0]
    start_pos_y = sim.maze.goal_start_pos[1] 
    print("start_pos:", start_pos_x, start_pos_y)
    sim.reset(start_pos_x, start_pos_y)
    print("num walls", len(sim.maze.walls))
    # simulate a trajectory of 100 steps going forward  
    num_steps = 10 #00# 270
    p = 5

    st = time.time()
    for i in tqdm.tqdm(range(num_steps)):
        ap = ApertureMask(11)
        s = np.random.uniform(0, int(11/2 + 1))
        # ap.create_narrow_aperture(int(s))
        #ap.randomize_aperture(round=True)
        # sim.update_aperture(ap.get_mask(), 'left')
        # sim.update_aperture(ap.get_mask(), 'right')
        # print("Aperture at step {} with size {}: {}".format(i, s, ap.get_mask()))
        theta = np.random.uniform() * np.pi
        dx = p*np.cos(theta)
        dy = p*np.sin(theta)
        for j in range (sim.cfg.env_config.steps_per_measurment):
            _, c, oob = sim.step(dx, dy) # go down 
            print("Aperture at size {}: {}".format(i * num_steps + j, sim.cameras[0].aperture_mask))
            # print("theta {}, dx: {}, dy: {}".format(theta, dx, dy))
            if c or oob:
                print("out of bounds: {}; collision: {}".format(oob, c))
                break
        if c or oob:
            break

    sim.visualize_rollout()






