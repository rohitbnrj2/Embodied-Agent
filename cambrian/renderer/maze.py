import pygame
import math
import numpy as np
import random
import csv
import json
from prodict import Prodict
from PIL import Image


class Maze:
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.load_path = self.cfg.load_path
        self.x_scale_factor = self.cfg.x_scale_factor
        self.y_scale_factor = self.cfg.y_scale_factor
        self.binarize_colors = self.cfg.binarize_colors
        self.grey = self.cfg.grey

        maze_img = Image.open(self.cfg.load_path)
        w, h = maze_img.size
        maze_img = maze_img.resize((int(w/8) + 1, int(h/8)+ 1))
        maze_img_array = np.array(maze_img)
        maze_img_array = maze_img_array[:,:,1]
        maze_img_array = np.swapaxes(maze_img_array, 0, 1) #pygame has opposite convention for x,y
        h, w = maze_img_array.shape[0], maze_img_array.shape[1]
        self.walls = []
        self.maze = []
        
        for x in range(h):
            for y in range(w):
                if int(maze_img_array[x,y])/255. < 0.995:
                    self.walls.append(pygame.Rect(self.x_scale_factor*x, self.y_scale_factor*y, self.x_scale_factor, self.y_scale_factor))
                    _dict = {
                        'color' : np.array(self._sample_colors()).astype(np.uint8),
                        'wall' : self.walls[-1]
                    }
                    self.maze.append(Prodict.from_dict(_dict))
                    
        print(f"Creating Maze with {len(self.walls)} Walls!")
        
        h = self.y_scale_factor*h
        w = self.x_scale_factor*w
        self.tunnel_width = h
        self.start_pos = [0,0]
        
        self.goal_start_pos = [h/2 - 2, 0]
        self.goal_end_pos = [h/2 + 2, w]
        self.window_size = [h,w]

    def _sample_colors(self,):
        if self.binarize_colors: 
            return random.choice([[255,255,255], [0.0,0.0,0.0]])
        else: 
            if self.grey:
                g = np.random.uniform() * 255
                return [g,g,g]
            else:
                c = np.random.uniform(size=[3]) * 255
                return c                

    def randomize_wall_colors(self, ):
        
        for i in range(len(self.maze)):
            self.maze[i].color = self._sample_colors()

    def check_ray_collision(self, ray):
        """
        return ray color as type float.
        """
        closest = 100000
        closestPoint = None
        ray_color = None

        for i in range(len(self.maze)): 
            ret = ray.checkPyGameRectCollision(self.maze[i].wall, self.window_size)
            if ret is not None: 
                intersectPoint, distance = ret
                if (distance < closest):
                    closestPoint = intersectPoint
                    closest = distance
                    ray_color = self.maze[i].color.astype(np.float32)/255.

        if ray_color is not None: 
            return closest, closestPoint, ray_color
        
        return None

    def collision(self, x, y, ct):
        """
        ct: collision_threshold 
        """
        bee_rect = pygame.Rect(x - ct,
                               y,
                               2*ct,
                               ct)
        coll = bee_rect.collidelist(self.walls)
        
        return coll != -1

    def check_bounds(self, x, y, ct = 50):
        """
        returns True if out of bounds of pygame
        """
        if np.abs(x - self.window_size[0]) < ct or np.abs(y - self.window_size[1]) < ct: 
            return True
        return False
    
    def render(self, work_surface):
        pygame.draw.circle(work_surface,(0,255,0), self.goal_end_pos, 20)
        for rect in self.maze: 
            pygame.draw.rect(work_surface, rect.color, rect.wall)
        return work_surface
