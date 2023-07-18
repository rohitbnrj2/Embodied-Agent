import random
import pygame
import numpy as np
from pygame.locals import *
from operator import add, sub
import math
from math import cos, sin, radians

from cambrian.renderer.ray import Ray
from cambrian.renderer.wall import Wall
from cambrian.utils.renderer_utils import map_one_range_to_other, scale_intensity, touint8
from cambrian.utils.pygame_colors import colors as PY_GAME_COLORS, rgb2grey


class TwoDRenderer:
    """
    TwoDRenderer
    """
    def __init__(self, cfg, cameras, maze):
        """
        collision_threshold: if pos < thres, sets collision=True (in pixel space)
        """
        self.maze = maze
        self.cameras = {}
        self.cfg = cfg
        self.num_rays_per_pixel = cfg.num_rays_per_pixel
        # stores the current rendered rays
        self.renderer_state = {}
        for camera in cameras:
            # print("Adding Camera {} to renderer.".format(camera.name))
            self.cameras[camera.name] = camera
            self.renderer_state[camera.name] = {
                'curr_render_dict': None
                }

    def update(self, mx, my):
        self.x = mx 
        self.y = my

    def render_all_cameras(self,):
        for i in self.cameras.keys():
            self.render_rays(i)

    def render_rays(self, camera_idx):
        camera = self.cameras[camera_idx]
        # update the camera location
        camera.update(self.x, self.y)
        # shoot rays from camera to the world 
        px_to_rays = camera.shoot_rays()
        render_dict = {} # does this render state cause a collision? 
        # 1. for each pixel: 
        cnt = 0
        for i in range(camera.num_pixels): 
            # 2. figure out which rays to render
            render_dict[i] = {'rays':[], 'closest_distance': []}
            # import pdb; pdb.set_trace()
            render_rays = px_to_rays[i]['rays']
            intensity = 0. 
            rgb_intensity = np.zeros(3)
            intersection_points = []
            true_ray_colors = []
            aperture_ray = []
            # 3. for each ray, get intersection color
            for k in range(len(render_rays)):
                r = render_rays[k]
                viz = px_to_rays[i]['render'][k]
                # rays that will be rendered for that pixel
                ret = self.maze.check_ray_collision(r)
                # add color of wall to total intensity at pixel
                if ret is not None: 
                    closest, closestPoint, ray_color = ret
                    render_dict[i]['rays'].append(r) 
                    render_dict[i]['closest_distance'].append(closest) 
                    intersection_points.append(closestPoint)
                    if viz:
                        true_ray_colors.append(ray_color)
                        intensity += rgb2grey(ray_color)
                        # if camera_idx == 'left': print(ray_color.rgb2grey())
                        rgb_intensity += ray_color # output is in float: 0-1
                        aperture_ray.append(False)
                        cnt+=1
                    else:
                        true_ray_colors.append(PY_GAME_COLORS['indianred'].rgb()) # all aperture rays should be 'red'
                        aperture_ray.append(True)

                else: 
                    intensity += self.cfg.ambient_color 
                    # rgb_intensity += np.zeros(3)
                    intersection_points.append(None)
                    true_ray_colors.append([0,0,0])

            # 4. final intensity value per pixel per-pixel
            # if camera_idx == 'left': print("number of rays viz:", cnt, intensity)
            render_dict[i]['intensity'] = intensity # intensity of rays at that pixel
            render_dict[i]['scaled_intensity'] = scale_intensity(intensity, self.cfg.num_rays_per_pixel) # intensity of rays at that pixel
            render_dict[i]['rgb'] = scale_intensity(rgb_intensity, self.cfg.num_rays_per_pixel)
            render_dict[i]['intersection_points'] = intersection_points # intersection points 
            render_dict[i]['true_ray_colors'] = true_ray_colors
            render_dict[i]['aperture_ray'] = aperture_ray
            # print("intersection_points", intersection_points)

        self.renderer_state[camera_idx]['curr_render_dict'] = render_dict

    def visualize_intensity(self, screen, font, display, max_window_size, renderer_state=None):
        # Draws rectangle 
        # assuming single pixel cams for now..
        n_cams = len(self.cameras.keys())
        ct = 0
        for i in self.cameras.keys(): 
            if renderer_state is None: 
                _render_dict = self.renderer_state[i]['curr_render_dict']
            else:
                _render_dict = renderer_state[i]['curr_render_dict']
            for k in range(self.cameras[i].num_pixels):
                color = _render_dict[k]['scaled_intensity']
                color = touint8(color)
                color = pygame.Color([color,color,color])
                text = font.render('Cam: {}'.format(i), True, (255,0,0))
                rect = pygame.Rect(max_window_size[0] - 300 + 300*(ct/n_cams), max_window_size[1]-100, 300*(ct/n_cams), 100)
                pygame.draw.rect(display, color, rect)
                display.blit(text, (max_window_size[0] - 300 + 300*(ct/n_cams), max_window_size[1]-100))
            ct += 1

            # pygame.draw.rect(display, color, rect)
        return screen
 
    # def visualize_camera(self, camera_idx, display):
    #     camera = self.cameras[camera_idx]
    #     camera.sensor_wall.draw(display)
    #     camera.aperture_wall.draw(display)

    def visualize_all_cameras(self, display, renderer_state=None, visualize_aperture=True):
        for i in self.cameras.keys():
            display = self.visualize_camera_rays(i, display, renderer_state, visualize_aperture)
        return display
    
    def visualize_camera_rays(self, camera_idx, display, renderer_state=None, visualize_aperture=True):
        
        if renderer_state is None: 
            render_dict = self.renderer_state[camera_idx]['curr_render_dict']
        else:
            render_dict = renderer_state[camera_idx]['curr_render_dict']

        for k in range(self.cameras[camera_idx].num_pixels):
            for i in range(len(render_dict[k]['rays'])):
                if render_dict[k]['intersection_points'][i] is not None: 
                    ray = render_dict[k]['rays'][i]
                    rgb = (render_dict[k]['true_ray_colors'][i] * 255).astype(np.uint8) # takes in np.uint8
                    # draw line to closest point
                    # if aperture ray check if we want to visualize it
                    if render_dict[k]['aperture_ray'][i]:
                        if visualize_aperture:
                            pygame.draw.line(display, rgb, (ray.x, ray.y), render_dict[k]['intersection_points'][i])
                    else:
                        pygame.draw.line(display, rgb, (ray.x, ray.y), render_dict[k]['intersection_points'][i])
        return display
            # if visualize_aperture: 
            #     for i in range(len(render_dict[k]['aperture_rays'])):
            #         ray = render_dict[k]['aperture_rays'][i]
            #         pygame.draw.line(display, 'red', (ray.x, ray.y), render_dict[k]['aperture_intersection_points'][i])


class ApertureMask: 
    def __init__(self, size = 100):
        self.aperture_mask = np.zeros(size) # zeros means it lets rays through

    def create_narrow_aperture(self, size):
        size = np.clip(size, 0, self.aperture_mask.shape[0]) # clip size 
        if size > 0:
            self.aperture_mask[0:size] = 1 
            self.aperture_mask[-size:-1] = 1 
            self.aperture_mask[-1] = 1 

    def randomize_aperture(self, round=False):
        self.aperture_mask = np.random.uniform(size=self.aperture_mask.shape)
        if round: 
            self.aperture_mask = self.aperture_mask.round()
            self.aperture_mask.astype(np.int32) # should be 0 or 1
    
    def get_mask(self,):
        return self.aperture_mask


class Camera: 
    def __init__(self, name, x, y, angle, fov, f_dir= -1.0, num_pixels=1, sensor_size=200, num_rays_per_pixel=100, aperture_mask=None):
        self.name = name
        self.x = x
        self.y = y
        self.fov_r = radians(fov) # deg to radians 
        self.angle_r = radians(angle) # deg to radians 
        self.num_pixels = num_pixels
        self.sensor_size = sensor_size
        self.f_dir = f_dir
        self.num_rays_per_pixel = num_rays_per_pixel 
        if aperture_mask is not None: 
            self.aperture_mask = aperture_mask
        else:
            self.aperture_mask = np.zeros(100) # zeros means it lets rays through
        
        # print("Aperture Pattern for Camera {}: {}".format(name, self.aperture_mask))

    def update(self, mx, my):
        # the prinicipal point is mouse
        self.x = mx
        self.y = my

    def update_aperture_mask(self, aperture_mask):
        self.aperture_mask = aperture_mask

    def shoot_rays(self,):
        self.pos = np.array([[self.x], [self.y]])
        self.f = 1. * (self.sensor_size/2) / (np.tan(self.fov_r/2)) 
        # print("Focal Length for {}: {}".format(self.name, self.f))
        # self.f = -1.0 * self.sensor_size / (2*np.sin(self.fov_r)) # focal_length 
        self.rot = np.array([[cos(self.angle_r), sin(self.angle_r)], # rotation matrix
                             [-sin(self.angle_r), cos(self.angle_r)]])

        # import pdb; pdb.set_trace()
        self.create_sensor_plane()
        # self.create_aperture()

        # sensor_ray = Ray(s_x, y1, sensor_plane_dir)
        pixel_to_ray_viz = {}
        # import pdb; pdb.set_trace()
        for i in range(self.num_pixels):
            # 1. sample self.num_rays_per_pixel from all points (these are the starting points)
            pixel_to_ray_viz[i] = {'rays': [], # rays that make it through to the world
                                   'render': [], # render the ray?
                                   }
            sensor_pts_x = self.sensor_line[0][ i*self.num_rays_per_pixel : i*self.num_rays_per_pixel + self.num_rays_per_pixel]
            sensor_pts_y = self.sensor_line[1][ i*self.num_rays_per_pixel : i*self.num_rays_per_pixel + self.num_rays_per_pixel]
            # 2. for each point on sensor plane, create a ray from sensor to principal point
            first_point_x = sensor_pts_x[0]
            first_point_y = sensor_pts_y[0]
            last_point_x = sensor_pts_x[-1]
            last_point_y = sensor_pts_y[-1]
            total_sensor_distance = math.dist([first_point_x, first_point_y], [last_point_x, last_point_y]) # euclidean
            # faster method that only shoots rays out of visible poins 
            visible_pts=[]
            for l in range(len(sensor_pts_x)):
                x,y = sensor_pts_x[l], sensor_pts_y[l]
                length = math.dist([first_point_x, first_point_y], [x, y]) # euclidean
                bin_idx = map_one_range_to_other(length, 0, len(self.aperture_mask)-1, 0, total_sensor_distance)
                bin_idx = int(bin_idx)
                # 3. Index the bin to find collisions
                if bin_idx < len(self.aperture_mask):
                    if self.aperture_mask[bin_idx] > 0: 
                        collision = True
                    else: 
                        collision = False
                        # ray is good 
                        w = Wall((x,y), (self.x, self.y))
                        if self.f_dir[0] <= 0.:
                            ray_angle = w.angle
                        else:
                            ray_angle = w.angle + math.radians(180.)
                            #ray_angle = w.angle + math.radians(180.)
        
                        r = Ray(x,y, ray_angle)
                        visible_pts.append([x,y])
                        pixel_to_ray_viz[i]['rays'].append(r)
                        pixel_to_ray_viz[i]['render'].append(not collision)

            # for k in range(len(sensor_pts_x)):
            #     x,y = sensor_pts_x[k], sensor_pts_y[k]
            #     length = math.dist([first_point_x, first_point_y], [x, y]) # euclidean
            #     # ray r
            #     w = Wall((x,y), (self.x, self.y))
            #     if self.f_dir == -1.0:
            #         ray_angle = w.angle
            #     else:
            #         ray_angle = w.angle + math.radians(180.)

            #     r = Ray(x,y, ray_angle)
            #     # map ray starting point -> aperture bin 
            #     # bin_idx = map_one_range_to_other(length, 0, len(self.aperture_mask), 0, total_sensor_distance)
            #     # bin_idx = int(bin_idx)-1
            #     bin_idx = map_one_range_to_other(length, 0, len(self.aperture_mask)-1, 0, total_sensor_distance)
            #     bin_idx = np.clip(bin_idx, 0, len(self.aperture_mask)-1) # length at wall collision maps to bin
            #     bin_idx = int(bin_idx)
            #     # 3. Index the bin to find collisions
            #     if bin_idx < len(self.aperture_mask):
            #         if self.aperture_mask[bin_idx] > 0: 
            #             collision = True
            #         else: 
            #             collision = False
            #     else:
            #         raise ValueError(self.aperture_wall.length, length, bin_idx)
                
            #     """
            #     # 3. check if ray has collided
            #     collision, pos, bin = self.check_ray_aperture_intersection(r, self.aperture_mask)                
            #     """
            #     pixel_to_ray_viz[i]['rays'].append(r)
            #     pixel_to_ray_viz[i]['render'].append(not collision)

        # print('num rays: {}'.format(len(pixel_to_ray_viz[i]['rays'])))
        return pixel_to_ray_viz

    def create_sensor_plane(self, ):
        sensor_x = self.f * np.ones(self.num_rays_per_pixel * self.num_pixels) # will use these points to create a line 
        sensor_y = np.linspace(-self.sensor_size/2, self.sensor_size/2, self.num_rays_per_pixel * self.num_pixels)
        sensor_line = np.vstack([sensor_x, sensor_y])
        self.sensor_line = self.pos  + np.matmul(self.rot, sensor_line)

    #     # sensor wall for viz purposes 
    #     x = self.f * np.ones(2) + _APERTURE_SENSOR_DISTANCE_
    #     y = np.linspace(-self.sensor_size/2, self.sensor_size/2, 2)
    #     line = np.vstack([x, y])
    #     line = self.pos + np.matmul(self.rot, line)
    #     self.sensor_wall = Wall(line[0], line[1], type="sensor", color="magenta")

    # def create_aperture(self):
    #     aperture_x = (self.f + _APERTURE_SENSOR_DISTANCE_) * np.ones(2)
    #     # enlarge by a little so all rays intersect (math trick)
    #     aperture_y = np.linspace( -(self.sensor_size + _APERTURE_SENSOR_DISTANCE_)/2, (self.sensor_size + _APERTURE_SENSOR_DISTANCE_)/2, 2)
    #     aperture_line = np.vstack([aperture_x, aperture_y])
    #     self.aperture_line = self.pos + np.matmul(self.rot, aperture_line)
    #     self.aperture_wall = Wall(self.aperture_line[0], self.aperture_line[1], type="aperture", color="pink1")
        
    # def check_ray_aperture_intersection(self, ray, aperture_mask):
    #     """
    #     ray: Ray()
    #     aperture_mask: [Kx1] vector of K units 
    #     """
    #     ret = ray.checkCollision(self.aperture_wall)
    #     if ret is not None: 
    #         collidePos, wall = ret
    #     else:
    #         # print('no collision between aperture & Ray')
    #         return False, -1, 0
    #         # raise ValueError("No Intersection between Aperture & Ray")
    #     # 1. compute distance between collision point & start of wall 
    #     sp = np.array([self.aperture_wall.start_pos[0], self.aperture_wall.start_pos[1]])
    #     length = math.dist(sp, collidePos) # euclidean
    #     # 2. map collision point to a bin in the aperture mask
    #     # self.aperture_wall.length maps to size K 
    #     bin_idx = map_one_range_to_other(length, 0, len(aperture_mask), 0, self.aperture_wall.length)
    #     bin_idx = int(bin_idx)-1 # length at wall collision maps to bin

    #     # 3. Index the bin to find collisions
    #     if bin_idx < len(aperture_mask):
    #         if aperture_mask[bin_idx] > 0: 
    #             return True, collidePos, bin_idx # collides since aperture is 1
    #         else: 
    #             return False, collidePos, bin_idx
    #     else:
    #         raise ValueError(self.aperture_wall.length, length, bin_idx)

    def trace_camera_rays(self, rays, sample_num_rays=10):
        x1 = self.x
        y1 = self.y
        # sensor extents 1 
        # x2 = self.x + (self.dir[0] * np.cos(self.angle - self.fov/2.))
        # y2 = self.y + (self.dir[1] * np.sin(self.angle - self.fov/2.))
        angle_extents_1 = self.angle - self.fov_r/2. # np.tan((x2-x1)/(y2-y1)) 
        # print("angle1", angle_extents_1 * 180./np.pi)

        # sensor extents 2
        # x3 = self.x + (self.dir[0] * np.cos(self.angle + self.fov/2.))
        # y3 = self.y + (self.dir[1] * np.sin(self.angle + self.fov/2.))
        angle_extents_2 = self.angle + self.fov_r/2. # np.tan((x3-x1)/(y3-y1)) 
        # print("angle2", angle_extents_2*180./np.pi)

        if sample_num_rays > 0: 
            dthetas = np.linspace(angle_extents_1, angle_extents_2, sample_num_rays)
            for theta in dthetas:
                rays.append(Ray(x1,y1, theta))
        else:
            rays.append(Ray(x1, y1, angle_extents_1))
            rays.append(Ray(x1, y1, angle_extents_2))

        return rays
    
    def visualize_aperture(self, d='auto'):
        f = self.sensor_size / (2*np.sin(self.fov_r)) # focal_length 
        # create an aperture at f 
        a_x = self.x + (self.dir[0] * f)
        a_y = self.y + (self.dir[1] * f) # ray: o + td 

        # add two walls with some opening 
        if d == 'auto':
            D = self.sensor_size/11
        elif d == 'random':
            # r = random.choice([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
            r = random.randint(2,20)
            D = self.sensor_size/r
        else:
            D = float(d) # d must be a number

        aperture_length = self.sensor_size/4 # should cover all edge rays
        y_1 = a_y + D 
        y_2 = y_1 + aperture_length
        y_3 = a_y - D
        y_4 = y_3 - aperture_length

        aperture_walls = []
        aperture_walls.append(Wall((a_x, y_1), (a_x, y_2), type='aperture', color = 'red1'))
        aperture_walls.append(Wall((a_x, y_3), (a_x, y_4), type='aperture', color = 'red1'))
        return aperture_walls

