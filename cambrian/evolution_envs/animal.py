from math import radians
from pathlib import Path
import numpy as np
from typing import List, Tuple
from prodict import Prodict
import yaml


from cambrian.evolution_envs.eye import SinglePixel
from cambrian.utils.renderer_utils import name_to_fdir
from utils.renderer_utils import points_on_circumference

class OculozoicAnimal:
    def __init__(self, config):
        self.config = config
        self.max_photoreceptors = config.max_photoreceptors
        self.mutation_count = 0 
        self.num_photoreceptors = self.config.init_photoreceptors

        # constants 
        self.radius = self.config.radius
        self.max_num_eyes_per_side = self.config.max_num_eyes_per_side
        self.visual_acuity_sigma = self.config.visual_acuity_sigma

        # we will sample pixel positions from the circle 
        self.right_eye_pixels = points_on_circumference(center = self.position, r= self.radius, n = self.max_num_eyes_per_side, direction='right')
        self.right_eye_pixels_occupancy = np.zeros(len(self.right_eye_pixels))
        self.left_eye_pixels = points_on_circumference(center = self.position, r= self.radius, n = self.max_num_eyes_per_side, direction='left')
        self.left_eye_pixels_occupancy = np.zeros(len(self.left_eye_pixels))

    def init_animal(self, mx, my):
        self.reset(mx, my)
        imaging_model = 'simple'
        fov = 150
        angle = 60
        self.sensor_size = self.config.init_sensor_size # large sensor size 
        self.left_eye_pixels_occupancy[-1] = 1
        pixel_pos = self.left_eye_pixels[-1]
        pixel_config = self.generate_pixel_config(imaging_model, fov, angle, pixel_pos=pixel_pos, direction='left') # should be looking left down
        init_pixel = self.add_pixel(pixel_config)
        self.pixels = [init_pixel]
        self.num_pixels = len(self.pixels)
        self.mutation_count += 1
        self.mutation_chain = []

    def observe_scene(self, dx, dy, geometry):
        self._update_position(dx, dy)
        eye_out = []
        for i in range(self.num_pixels):
            raw_photoreceptor_output = self.pixels[i].render(dx, dy, geometry, self.num_photoreceptprs_per_pixel)
            if self.config.total_intensity_output_only:
                intensity, _ = np.mean(raw_photoreceptor_output)
                eye_out.append(intensity)
            else:
                eye_out.append(raw_photoreceptor_output)

        return eye_out
    
    def reset_position(self, mx, my):
        self.x = mx
        self.y = my
        self.position = np.array([self.x, self.y])

    def mutate(self, mutation_type, mut_args=Prodict):
        self.mutation_count += 1
        _mut = Prodict()
        _mut.type = mutation_type

        if mutation_type == 'add_photoreceptor':
            self.num_photoreceptors += self.config.increment_photoreceptor
            self.num_photoreceptors = np.clip(self.num_photoreceptors, 0, self.max_photoreceptors)
            _mut.args = {'num_photoreceptors': self.num_photoreceptors}

        elif mutation_type == 'simple_to_lens':
            self.pixels[mut_args.pixel_idx].imaging_model = 'lens'
            _mut.args = {'cam_idx': mut_args.pixel_idx}
            # start from the first compound eye and incrementally add a lens, break after affing 
            # for i in range(self.num_pixels):
            #     if self.pixels[i].imaging_model == 'simple':
            #         # the default imaging model will be used (at init)
            #         self.pixels[i].imaging_model = 'lens'
            #         _mut.args = {'cam_idx': i}
            #         break

        elif mutation_type == 'add_pixel':
            _dir = np.random.choice('left', 'right')
            pixel_config = self.generate_pixel_config(mut_args.imaging_model, mut_args.fov, mut_args.angle, pixel_pos=None, direction=_dir)
            self.add_pixel(pixel_config)
            _mut.args = {'imaging_model': mut_args.imaging_model, 'fov': mut_args.fov, 'angle': mut_args.angle, 
                         'pixel_pos': None, 'direction': _dir}

        elif mutation_type == 'update_pixel':
            self.pixels[mut_args.pixel_idx].update_pixel_config(mut_args.fov_update, mut_args.angel_r_update, mut_args.fov_update)
            _mut.args = {'fov_update':mut_args.fov_update, 
                         'angel_r_update': mut_args.angel_r_update, 
                         'fov_update': mut_args.fov_update}
        else:
            raise ValueError("{} not found".format(mutation_type))
        
        self.mutation_chain.append(_mut)

    def add_pixel(self, pixel_config):
        self.num_photoreceptprs_per_pixel = int(self.num_rays/self.num_pixels)
        pixel = SinglePixel(pixel_config)
        self.pixels.append(pixel)
        self.num_pixels = len(self.pixels)

    def remove_pixel(self, ):
        self.num_pixels -= 1
        self.num_photoreceptprs_per_pixel = int(self.num_rays/self.num_pixels)
        self.pixels.pop()

    def generate_pixel_config(self, imaging_model, fov, angle, pixel_pos=None, direction='left'):
        config = {}
        if pixel_pos is None: 
            pixel_pos = self._sample_new_pixel_position(direction)
        config.x = pixel_pos[0]
        config.y = pixel_pos[1]
        config.sensor_size = self.sensor_size
        config.fov = fov
        config.angle = angle
        config.f_dir = name_to_fdir(direction)
        config.visual_acuity = self.num_photoreceptprs_per_pixel/self.visual_acuity_sigma
        config.imaging_model = imaging_model
        config = Prodict.from_dict(config)
        return config 

    def load_animal_from_state(self, state_config_file):

        with open(state_config_file, "r") as ymlfile:
            dict_cfg = yaml.load(ymlfile, Loader=yaml.Loader)
            _state = Prodict.from_dict(dict_cfg)

        # load animal constant
        self.num_photoreceptprs_per_pixel = _state.num_rays_per_pixel
        self.radius = _state.radius
        self.max_num_eyes_per_side = _state.max_num_pixels_per_side
        self.visual_acuity_sigma = _state.visual_acuity_sigma

        # load eye location
        self.right_eye_pixels = _state.right_eye_pixels
        self.right_eye_pixels_occupancy = _state.right_eye_pixels_occupancy
        self.left_eye_pixels = _state.left_eye_pixels
        self.left_eye_pixels_occupancy = _state.left_eye_pixels_occupancy

        # load pixel configuration
        self.num_pixels = _state.num_pixels
        self.pixels = []
        for p in _state.pixels: 
            self.pixels.append(p.from_dict())

    def save_animal_state(self, save_dir=None):
        """
        if save_dir is None, it just returns the state
        """
        _state = {}
        _state = Prodict.from_dict(_state)
        
        # save animal constant
        _state.num_rays_per_pixel = self.num_photoreceptprs_per_pixel
        _state.radius = self.radius
        _state.max_num_pixels_per_side = self.max_num_eyes_per_side
        _state.visual_acuity_sigma = self.visual_acuity_sigma

        # save eye location
        _state.right_eye_pixels = self.right_eye_pixels
        _state.right_eye_pixels_occupancy = self.right_eye_pixels_occupancy
        _state.left_eye_pixels = self.left_eye_pixels
        _state.left_eye_pixels_occupancy = self.left_eye_pixels_occupancy

        # save pixel configuration
        _state.num_pixels = len(self.pixels)
        _state.pixels = []
        for p in self.pixels: 
            _state.pixels.append(p.to_dict())

        if save_dir is not None: 
            target = str(Path.joinpath(save_dir,'mutation_{}.yml'.format(self.mutation_count)))
            with open(target, 'w') as outfile:
                _state = _state.to_dict()
                yaml.dump(_state, outfile, default_flow_style=False)

        return _state

    def _update_position(self, dx, dy):
        self.x += dx
        self.y += dy
        self.position = np.array([[self.x], [self.y]])
        # everything else will shift equally as well. 
        self.right_eye_pixels += np.array([dx, dy])
        self.left_eye_pixels += np.array([dx, dy])

    def _sample_new_pixel_position(self, direction='left'):
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
            pixel_pos = self.right_eye_pixels[idx]
            self.right_eye_pixels_occupancy[idx] = 1

        return pixel_pos


################ 
## Utils
################ 

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