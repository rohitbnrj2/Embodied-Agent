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
        self.window_size = self.cfg.window_size
        self.font = pygame.font.SysFont('Arial', 20)


        maze_img = Image.open(self.cfg.load_path)
        maze_img = np.array(maze_img)
        maze_img = maze_img[:,:, :3]
        maze_img = np.swapaxes(maze_img, 0, 1)
        
        yellow_pixels = []
        green_pixels = []
        h, w = maze_img.shape[0], maze_img.shape[1]
        self.walls = []
        self.maze = []
        
        for i in range(maze_img.shape[0]):
            for j in range(maze_img.shape[1]):
                
                if list(maze_img[i,j,:]) == [0,0,0]:
                    wall_ = pygame.Rect(self.x_scale_factor*i, self.y_scale_factor*j,
                                        self.x_scale_factor, self.y_scale_factor)
                    self.walls.append(wall_)
                    _dict = {
                        'color' : np.array(self._sample_colors()).astype(np.uint8),
                        'wall' : wall_
                    }
                    self.maze.append(Prodict.from_dict(_dict))
                    
                elif list(maze_img[i,j,:]) == [250,255,8]:
                    yellow_pixels.append((self.x_scale_factor*i, self.y_scale_factor*j))
                
                elif list(maze_img[i,j,:]) == [0,255,11]:
                    green_pixels.append((self.x_scale_factor*i, self.y_scale_factor*j))
                    
        self.goal_start_pos = random.choice(yellow_pixels)
        self.goal_end_pos = random.choice(green_pixels)
        self.window_size = self.window_size
        self.tunnel_width = self.window_size[0]
        print(f"Creating Maze with {len(self.walls)} Walls fo size ({h, w})!")

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
    
    def render_intensity(self, work_surface, processed_eye):
        # draw the intensities 
        n_cams = len(processed_eye)
        for ct, p_intensity in enumerate(processed_eye):
            # print("p_intensity", p_intensity)
            p_intensity = (p_intensity * 255).astype(np.uint8)
            color = pygame.Color([p_intensity,p_intensity,p_intensity])
            
            rect = pygame.Rect(self.window_size[0] - 300 + 300*(ct/n_cams), self.window_size[1]-100, 300*(ct/n_cams), 100)
            pygame.draw.rect(work_surface, color, rect)
            text = self.font.render('Pixel: {}'.format(ct), True, (255,0,0))
            work_surface.blit(text, (self.window_size[0] - 300 + 300*(ct/n_cams), self.window_size[1]-100))

        return work_surface