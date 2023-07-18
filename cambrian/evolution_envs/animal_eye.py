import random
import numpy as np
from pygame.locals import *
import math
from typing import List, Tuple

from geometry import Wall
from prodict import Prodict
from utils.renderer_utils import points_on_circumference


class Eye:
    def __init__(self, config):
        self.config = config
        self.num_rays = config.num_rays
        self.num_pixels = 1
        self.num_rays_per_pixel = self.num_rays/self.num_pixels
        
        # constants 
        self.radius = 20.
        self.max_num_pixels_per_side = 25
        self.visual_acuity_sigma = 5.
        self.sensor_size = 10 # doesn't really change much since fov accounts for it.

    def init_eye(self, mx, my):
        self.reset(mx, my)
        self.generate_pixel_pos()
        init_pixel = self.add_pixel()
        self.pixels = [init_pixel]

    def reset(self, mx, my):
        self.x = mx
        self.y = my
        self.position = np.array([self.x, self.y])
    
    def update(self, dx, dy):
        self.x += dx
        self.y += dy
        self.position = np.array([[self.x], [self.y]])

    def generate_pixel_pos(self,):
        self.right_eye_pixels = points_on_circumference(center = self.position, r= self.radius, n = self.max_num_pixels_per_side, direction='right')
        self.right_eye_pixels_occupancy = np.zeros(len(self.right_eye_pixels))
        self.left_eye_pixels = points_on_circumference(center = self.position, r= self.radius, n = self.max_num_pixels_per_side, direction='left')
        self.self.left_eye_pixels_occupancy = np.zeros(len(self.left_eye_pixels))

    def sample_new_pixel_position(self, direction='left'):

        def get_idx(arr):
            # start from the middle and iterate left and right to find the first empty index
            l = len(arr)
            _c = int(l/2)
            _dir = 1
            i = 0 
            empty = True
            while empty:
                idx = int(_c + (i * _dir))
                if idx < 0 or idx > len(arr):
                    return None
                e = arr[idx]
                if e == 0:
                    empty = False 
                    return idx
                else:
                    _dir *= -1
                    idx = int(_c + (i * _dir))
                    if idx < 0 or idx > len(arr)-1:
                        return None
                    e = arr[idx]
                    if e == 0:
                        empty = False 
                        return idx

                i +=1 
            
            return idx

        if direction == 'left':
            idx = get_idx(self.left_eye_pixels_occupancy)
            if idx is None: 
                return None 
            pixel_pos = self.left_eye_pixels[idx]
            self.left_eye_pixels_occupancy[idx] = 1
        else:
            idx = get_idx(self.right_eye_pixels_occupancy)
            if idx is None: 
                return None 
            pixel_pos = self.right_eye_pixels_[idx]
            self.right_eye_pixels_occupancy[idx] = 1

        return pixel_pos

    def add_pixel(self, direction='left'):
        self.num_pixels += 1
        self.num_rays_per_pixel = int(self.num_rays/self.num_pixels)
        pixel_config = self.generate_pixel_config(direction)
        pixel = SinglePixel(pixel_config)
        self.pixels.append(pixel)

    def remove_pixel(self, ):
        self.num_pixels -= 1
        self.num_rays_per_pixel = int(self.num_rays/self.num_pixels)
        self.pixels.pop()

    def generate_pixel_config(self, direction, fov, orientation):
        config = {}
        pixel_pos = self.sample_new_pixel_position(direction)
        config.x = pixel_pos[0]
        config.y = pixel_pos[1]
        config.sensor_size = self.sensor_size
        config.fov = fov
        config.orientation = orientation
        config.f_dir = name_to_fdir(direction)
        config.visual_acuity = self.num_rays_per_pixel/self.visual_acuity_sigma
        config = Prodict.from_dict(config)
        return config 

    def render_eye(self, geometry, dx, dy):
        eye_out = []
        for i in range(self.num_pixels):
            if self.total_intensity_output_only:
                intensity, _ = self.render_pixel(geometry, dx, dy, self.pixels[i])
                eye_out.append(intensity)
            else:
                _, raw_photoreceptor_output = self.render_pixel(geometry, dx, dy, self.pixels[i])
                eye_out.append(raw_photoreceptor_output)

        return eye_out

    def render_pixel(self, geometry, dx, dy, pixel: SinglePixel):
        # update location
        pixel.update(dx, dy)
        # render geometry 
        final_intensity = pixel.render_pixel(geometry, self.num_rays_per_pixel)
        return final_intensity

class SinglePixel:
    def __init__(self, config):
        self.config = config
        self.update_pixel_params(self.config)

    def update_pixel_params(self, config):
        self.config = config
        self.fov_r = math.radians(self.fov) # deg to radians 
        self.angle_r = math.radians(self.config.orientation) # deg to radians 
        self.sensor_size = self.config.sensor_size
        self.visual_acuity = self.config.visual_acuity # should be 4/5
        self.f_dir = self.config.f_dir

        self.f = self.f_dir * (self.sensor_size/2) / (np.tan(self.fov_r/2)) # focal length
        self.rot = np.array([[math.cos(self.angle_r), math.sin(self.angle_r)], # rotation matrix
                             [-math.sin(self.angle_r), math.cos(self.angle_r)]])

        # other constants:
        self.fixed_incident_photon_flux = int(self.num_rays/10) # 1/10th of the total flux
        self.reset(self.config.x, self.config.y)

    def reset(self, mx, my):
        self.x = mx
        self.y = my
        self.position = np.array([[self.x], [self.y]])

    def update(self, dx, dy):
        self.x += dx
        self.y += dy
        self.position = np.array([[self.x], [self.y]])

    def _create_sensor_plane(self, num_rays):
        sensor_x = self.f * np.ones(num_rays) # will use these points to create a line
        sensor_y = np.linspace(-self.sensor_size/2, self.sensor_size/2, num_rays)
        sensor_line = np.vstack([sensor_x, sensor_y])
        self.sensor_line = self.position + np.matmul(self.rot, sensor_line)

    def render(self, geometry):
        # set visual acuity field
        x = np.arange(0, self.num_rays)
        self.visual_field = gaussian(x, self.num_rays/2, self.visual_acuity)

        # sample rays from a sensor plane
        self._create_sensor_plane(self.num_rays)
        rays = []
        for i in range(self.num_rays):
            _ray = {}
            x, y = self.sensor_line[0][i], self.sensor_line[1][i]
            w = Wall((x,y), (self.x, self.y))
            if self.f_dir == -1.0:
                ray_angle = w.angle
            else:
                ray_angle = w.angle + math.radians(180.)

            r = Ray(x,y, ray_angle)
            intensity, distance = geometry.collide_ray([r]) # returns intensity [0,1], distance to wall
            _ray['ray'] = r
            _ray['intensity'] = intensity # should be 0, 1
            _ray['distance'] = distance

            rays.append(_ray)

        return rays

    def forward_model(self, rays, visual_accuity=True, distance_weight=False):
        """
        rays: list of N rays sorted from [0, sensor_size]
            each ray:
               {intensity, distance to wall, direction}
        visual_accuity:
            weight by visual_accuity
        distance_weight:
            weight by distance_weight
        """

        # get max distance to the wall
        if distance_weight:
            max_dist = max([ray['distance'] for ray in rays])

        for r in range(self.num_rays):
            intensity = rays[r]['intensity']
            if distance_weight:
                # weight by distance to the wall
                dist = rays[r]['distance']
                intensity = intensity * dist/max_dist
            if visual_accuity:
                # weight by visual acuity at that point.
                intensity = intensity * self.visual_field[r]

            rays[r]['intensity'] = intensity

        return rays

    def render_pixel(self, geometry, num_rays, photon_noise=True, visual_accuity=True, distance_weight=False):
        # set num rays
        self.num_rays = num_rays
        final_intensity = 0.
        raw_photoreceptor_output = []

        rays = self.render(geometry)
        rays = self.forward_model(rays)

        for r in rays:
            raw_photoreceptor_output.append(r['intensity'])
            final_intensity += r['intensity']

        if photon_noise:
            # add camera noise, each ray is a photon striking the eye
            final_intensity = self.add_camera_noise(final_intensity)

        # scale by num_rays (should be turned off with aperture)
        final_intensity = final_intensity/self.num_rays

        if not photon_noise:
            # add some noise, if photon noise is disabled
            final_intensity += np.random.normal(loc=0, scale=0.1)

        # clip to 0,1
        final_intensity = np.clip(final_intensity, 0., 1.0)

        # one idea could be to look at each ray as a photoreceptor -> instead of just outputing the final intensity! 
        return final_intensity, raw_photoreceptor_output

    # Function to add camera noise
    def add_camera_noise(self, ray_intensities, qe=0.69, sensitivity=5.88,
                     dark_noise=2.29, bitdepth=12, baseline=100,
                     rs=np.random.RandomState(seed=42)):
        # http://kmdouglass.github.io/posts/modeling-noise-for-image-simulations/

        # Add shot noise
        photons = rs.poisson(ray_intensities, size=ray_intensities.shape)

        # switching off the converion...
        # # Convert to electrons
        # electrons = qe * photons

        # # Add dark noise
        # electrons_out = rs.normal(scale=dark_noise, size=electrons.shape) + electrons

        # Convert to ADU and add baseline
        # max_adu     = int(2**bitdepth - 1)
        # adu         = (electrons_out * sensitivity).astype(int) # Convert to discrete numbers
        # adu += baseline
        # adu[adu > max_adu] = max_adu # models pixel saturation

        # return adu
        return photons
