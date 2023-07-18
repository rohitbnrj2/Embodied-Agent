from copy import deepcopy
from pygame.locals import *
import numpy as np
import pygame
import sys
from pathlib import Path

from cambrian.renderer.maze import Maze
from cambrian.renderer.renderer import ApertureMask, Camera, TwoDRenderer
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
        self.cameras = []

        # add noise to each observation 
        seed = 89012
        self.rs = np.random.RandomState(seed)
        self.num_photons = 2

    def init(self, init_x=None, init_y=None, mode='train'):
        self.mode = mode
        # call reset 
        self.create_simulator(self.cfg.geometry, self.cfg.imaging, self.cfg.renderer_config)
        self.reset(init_x, init_y)

    def update_state_dict(self, x,y, render_dict, is_collision, out_of_bounds):
        # update trial number
        self.trial += 1
        self.sim_states.positions_rollout.append([x,y]) # should be a list of [[x,y], ...]
        self.sim_states.collision_rollout.append(True if is_collision or out_of_bounds else False) # should be a list of [FFFT]
        self.curr_render_dict = render_dict
        if self.mode == 'test':
            # in train state there is no renderer states...
            self.sim_states.renderer_state_rollout.append(deepcopy(render_dict)) # should be a list of [[x,y], ...]
        # pdb.set_trace()
        for cam in render_dict:
            cam_intensity = render_dict[cam]['curr_render_dict'][0]['intensity']
            self.sim_states.cam_intensities[cam].append(cam_intensity)

    def reset_sim_states(self):
        _dict =  {
            # 'scene_config': None, 
            # 'list_of_camera_configs': None,
            # 'geometry': [], # walls 
            'positions_rollout': [],
            'collision_rollout': [],
            'renderer_state_rollout': [],
            'cam_intensities' : {f"cam_{i}":[] for i in range(self.n_cams)}
        }
        return Prodict.from_dict(_dict)

    def reset(self, init_x=None, init_y=None):
        if init_x == None: 
            self.init_x = self.maze.start_pos[0] + (self.maze.tunnel_width/2)
            self.x = self.init_x + self.cfg.sim_config.x_offset
        else:
            self.x = init_x

        if init_y == None: 
            self.init_y = self.cfg.sim_config.init_y #self.tunnel.start_pos[1]
            self.y = self.init_y + self.cfg.sim_config.y_offset
        else:
            self.y = init_y
        self.sim_states = self.reset_sim_states()

    def step(self, dx, dy):
        self.x += dx 
        self.y += dy
        self.renderer.update(self.x, self.y)
        
        # st = time.time()
        self.renderer.render_all_cameras()
        # tt = time.time()-st
        # print("Time per render all cameras {}".format(tt))
        # print("length of cameras: ", self.renderer.cameras)
        # print("length of walls: ", len(self.maze.walls))

        # check collision
        is_collision = self.maze.collision(self.x, self.y, self.cfg.sim_config.collision_threshold)
        out_of_bounds = self.maze.check_bounds(self.x, self.y)
        self.update_state_dict(self.x, self.y, self.renderer.renderer_state, is_collision, out_of_bounds)
        
        cur_cam_intensities = {f"cam_{i}": 0 for i in range(self.n_cams)}
        for cam in cur_cam_intensities:
            cur_cam_intensities[cam] = self.sim_states.cam_intensities[cam][-1]
        return cur_cam_intensities, is_collision, out_of_bounds

    def visualize_rollout(self, overwrite_path=None):
        if self.mode == 'train':
            print("Cannot Visualize in train mode. Switch to tests")
            return 
        # pygame simulator specifics 
        DEPTH = 32
        FLAGS = 0
        idx = 0 
        if not self.cfg.sim_config.use_display:
            DISPLAY = (1, 1)
        else:
            DISPLAY = tuple(self.maze.window_size)
            self.display = pygame.Surface(self.maze.window_size)
            pygame.display.set_caption('RoboBee Simulator')

        self.screen = pygame.display.set_mode(DISPLAY, FLAGS, DEPTH)
        clock = pygame.time.Clock()

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
            mx, my = pos[0], pos[1]
            pygame.draw.circle(work_surface,(0,0,255), pos, 15)
            # print("Bee Position :", mx, my)
            # pdb.set_trace()
            # print("Positions: ", mx, my, self.sim_states.camera_right_intensity[idx])

            renderer_state = self.sim_states.renderer_state_rollout[idx]
            # save images 
            if not self.cfg.sim_config.use_display:
                work_surface = self.visualize_single_pass(work_surface, renderer_state)
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
        try:
            make_video("./{}/".format(str(save_img_dir)), 
                   sorted(glob.glob("{}/sim_*.jpg".format(str(save_img_dir)))))
            print("Visualizations saved at: {}".format(str(save_img_dir)))
        except Exception as e: 
            print("can't make video cause of {}".format(e))
            

    def visualize_single_pass(self, work_surface, renderer_state):
        # output the camera image at current step.

        work_surface = self.maze.render(work_surface)
        
        work_surface = self.renderer.visualize_all_cameras(work_surface, 
                                                           renderer_state=renderer_state, 
                                                           visualize_aperture=self.cfg.sim_config.visualize_aperture)
        
        work_surface = self.renderer.visualize_intensity(work_surface, self.font, 
                                                         work_surface, self.maze.window_size, 
                                                         renderer_state=renderer_state)
        return work_surface

    def create_simulator(self, scene_config, imaging_config, renderer_config, randomize_ap=False):
        self.maze = Maze(cfg=scene_config)
        
        if True:
            cam_props = imaging_config.camera_properties
            cam_props = Prodict.from_dict(cam_props)
            n_cams = cam_props.n_cams
            self.n_cams = n_cams
            camera_dict = {}
            angles = cam_props.angles
            renderer_config.num_rays_per_pixel = int(self.cfg.renderer_config.total_rays/n_cams) + 1
            for i in range(n_cams):
                ap = ApertureMask(cam_props.aperture_size)
                if randomize_ap: 
                    ap.randomize_aperture(round=False)
            
                if self.mode == 'test':
                    ap.create_narrow_aperture(cam_props.aperture_narrow_size)
                theta = angles[i]*(np.pi/180.)
                if angles[i] == 90.:
                    raise ValueError("Behaviour seems weird near 90 degrees!")
                    
                cam = Camera(name= f"cam_{i}", x=cam_props.x, y=cam_props.y, angle= (theta*180.)/np.pi, 
                             fov=cam_props.fov/n_cams, f_dir= (np.cos(theta), np.sin(theta)), 
                             num_pixels=cam_props.num_pixels, 
                             sensor_size= cam_props.sensor_size, aperture_mask=ap.get_mask(), 
                             num_rays_per_pixel= renderer_config.num_rays_per_pixel)
                self.cameras.append(cam)
        ##### 

        # Create a renderer with all the cameras
        self.renderer = TwoDRenderer(renderer_config, cameras=self.cameras, maze=self.maze)

    def update_aperture(self, renderer_config, aperture, name: str):
        for idx in range(len(self.cameras)): 
            # pdb.set_trace()
            if name == self.cameras[idx].name: 
                self.cameras[idx].aperture_mask = aperture

        self.renderer = TwoDRenderer(renderer_config, cameras=self.cameras, maze=self.maze)

    def reset_geometry(self, scene_config, renderer_config): 
        self.maze = Maze(cfg=scene_config)
        self.renderer = TwoDRenderer(renderer_config, cameras=self.cameras, maze=self.maze)
        
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






