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

        if isinstance(self.load_path, list):
            # selects a maze at random if it is a list of mazes
            load_path = self._select_maze(self.load_path)
        else:
            load_path = self.load_path

        maze_img = Image.open(load_path)
        maze_img = np.array(maze_img)
        maze_img = maze_img[:,:, :3]
        maze_img = np.swapaxes(maze_img, 0, 1)
        
        yellow_pixels = []
        green_pixels = []
        h, w = maze_img.shape[0], maze_img.shape[1]
        self.walls = []
        self.maze = []
        color_change_period = 8
        
        for i in range(maze_img.shape[0]):
            for j in range(maze_img.shape[1]):
                
                if list(maze_img[i,j,:]) == [0,0,0]:
                    wall_ = pygame.Rect(self.x_scale_factor*i, self.y_scale_factor*j,
                                        self.x_scale_factor, self.y_scale_factor)
                    self.walls.append(wall_)
                    # _color = np.array(self._sample_colors()).astype(np.uint8)
                    _color = np.array(self._sample_colors_freq(i, j, color_change_period)).astype(np.uint8)
                    _dict = {
                        'color' : _color,
                        'wall' : wall_
                    }
                    self.maze.append(Prodict.from_dict(_dict))
                    
                elif list(maze_img[i,j,:]) == [250,255,8]:
                    yellow_pixels.append((self.x_scale_factor*i, self.y_scale_factor*j))
                
                elif list(maze_img[i,j,:]) == [0,255,11] or list(maze_img[i,j,:]) == [0,255,123]:
                # elif maze_img[i,j,0] == 0 or maze_img[i,j,1] == 255:
                    green_pixels.append((self.x_scale_factor*i, self.y_scale_factor*j))
                    
        self.occupancy_grid = self.convert_to_occupancy_grid(self.maze, w, h)

        self.goal_start_pos = random.choice(yellow_pixels)
        self.goal_end_pos = random.choice(green_pixels)
        # self.goal_end_pos = (self.goal_end_pos[0] * self.x_scale_factor, self.goal_end_pos[1] * self.y_scale_factor)
        # print("goal_end_position: {}".format(self.goal_end_pos))
        # print("green_pixels", green_pixels)
        self.window_size = (h * self.x_scale_factor, w * self.y_scale_factor)
        # self.window_size = (h, w)
        self.tunnel_width = self.window_size[0]
        print(f"Creating Maze with {len(self.walls)} Walls of size ({h, w}) and goal end position: {self.goal_end_pos}!")

    def convert_to_occupancy_grid(self, walls, grid_width, grid_height):
        # Create an empty grid filled with zeros
        occupancy_grid = np.full((grid_width, grid_height, 3), None)

        # Iterate through the list of walls and mark corresponding grid cells as occupied
        for wall_info in walls:
            color = wall_info.color
            wall = wall_info.wall
            x1, y1, x2, y2 = wall.left, wall.top, wall.right, wall.bottom

            # Convert wall coordinates to grid coordinates
            x1 //= 2  # Assuming each cell is represented as 2x2 units
            x2 //= 2
            y1 //= 2
            y2 //= 2

            # Mark the grid cells occupied by the wall
            occupancy_grid[x1:x2, y1:y2] = color

        return occupancy_grid

    def _select_maze(self, mazes):
        return np.random.choice(mazes)

    def _sample_colors_freq(self,  i, j, color_change_period=8,):
        if int((i+j)/color_change_period) % 2 == 0:
            color = [0.0,0.0,0.0]
        else:
            color = [255,255,255]
        return color

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

        w,h = self.occupancy_grid.shape[0], self.occupancy_grid.shape[1]
        x,y = ray.x, ray.y
        dx, dy = ray.dir / np.linalg.norm(ray.dir)
        for _ in range(closest):
            x += dx
            y += dy

            map_x = int(x / self.x_scale_factor)
            map_y = int(y / self.y_scale_factor)
            if map_x > w - 1 or map_x < 0 or map_y > h - 1 or map_y < 0:
                return None

            square = self.occupancy_grid[map_x][map_y]
            if np.all(square != None):
                ray_color = np.array(square, dtype=np.float32) / 255.
                closestPoint = np.array([x,y])
                closest = np.linalg.norm(np.array([ray.x, ray.y]) - closestPoint)
                break

        return closest, closestPoint, ray_color

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
        # if np.abs(x - self.window_size[0]) < ct or np.abs(y - self.window_size[1]) < ct: 
        if x - ct < 0 or y - ct < 0:
            return True
        if x + ct > self.window_size[0] or y + ct > self.window_size[1]:
            return True
        return False
    
    def render(self, work_surface):
        pygame.draw.circle(work_surface,(255,255,255), self.goal_end_pos, 20)
        for rect in self.maze: 
            pygame.draw.rect(work_surface, rect.color, rect.wall)
        return work_surface
    
    def render_intensity(self, work_surface, processed_eye):
        # draw the intensities 
        n_cams = len(processed_eye)
        self.font = pygame.font.SysFont('Arial', 20)
        for ct, p_intensity in enumerate(processed_eye):
            p_intensity = np.clip(p_intensity, 0, 1) * 255.
            c_ = np.array([p_intensity, p_intensity, p_intensity]).astype(np.uint8)
            color = pygame.Color(c_)
            
            rect = pygame.Rect(self.window_size[0] - 300 + 300*(ct/n_cams), self.window_size[1]-100, 300*(ct/n_cams), 100)
            pygame.draw.rect(work_surface, color, rect)
            text = self.font.render('Pixel: {}'.format(ct), True, (255,0,0))
            work_surface.blit(text, (self.window_size[0] - 300 + 300*(ct/n_cams), self.window_size[1]-100))

        return work_surface
