import random
from cambrian.renderer.ray import Ray
from cambrian.renderer.wall import Wall
from cambrian.utils.pygame_colors import rgbfloat2grey
import numpy as np
import math
from typing import List, Tuple
from prodict import Prodict
import yaml 


from cambrian.utils.renderer_utils import gaussian, map_one_range_to_other

class SinglePixel:
    """
    A pixel is a single eye with a sensor plane that has 
    atleast two photoreceptors. 
    ----------------------------------------------
                * photoreceptors * 
        ()()()()()()        ()()()()()() 
                * sensor plane * 
        ===============        ===============
                * Modulation of light * 
            Simple                  Lens
                * incoming light * 
        ^^^^^^^^^^^         ^^^^^^^^^^^^
    ----------------------------------------------
    Description: 
        Simple Eye: Incoming light from all directions
            hits the photoreceptor which records the intensity 
            per ray which is summed for a 
            total photoreceptor output `raw_radiance`. 
            
        Lens Eye: Incoming light is modulated by a lens 
            through the focal point, and the 
            photoreceptor records the intensity. 
            Here one photoreceptor corresponds to one ray
            and makes the `raw_radiance`. 
        
    """
    def __init__(self, config=None):

        ##########################################
        ###### Some Constants 
        self.constraint_angles = False
        self.ALLOWED_ANGLES = [0. , 15., 30. , 45., 60. , 75., 120. , 135. , 150. , 180.]
        for i in range(len(self.ALLOWED_ANGLES)):
            self.ALLOWED_ANGLES[i] = math.radians(self.ALLOWED_ANGLES[i])
        ###### Some Constants
        ##########################################

        if config: 
            self.config = config
            self.init_pixel_from_config(self.config)


    def init_pixel_from_config(self, config):
        self.config = config
        ###################################
        ##### Configurable Parameters #####
        self.fov_r = math.radians(self.config.fov) # deg to radians 
        self.angle_r = math.radians(self.config.angle) # deg to radians 
        self.sensor_size = self.config.sensor_size
        self.visual_acuity = self.config.visual_acuity # should be 4/5
        self.f_dir = (np.cos(self.angle_r), np.sin(self.angle_r)) # tuple 
        self.imaging_model = self.config.imaging_model # one of ['simple', 'lens']
        ##### Configurable Parameters #####
        ###################################

        ##########################################################
        ##### Parameters wrt the animal stored here for ease #####
        self.animal_direction = self.config.animal_direction
        self.animal_idx = self.config.animal_idx
        ##### Parameters wrt the animal stored here for ease #####
        ##########################################################
        self.fov_r = self._update_fov(self.fov_r)
        self.angle_r = self._update_angles(self.angle_r)
        self.f = (self.sensor_size/2) / (np.tan(self.fov_r/2)) # focal length
        self.closed_pinhole_percentage = self.config.closed_pinhole_percentage # number between 0 and 1 
        self.rot = np.array([[math.cos(self.angle_r), math.sin(self.angle_r)], # rotation matrix
                             [-math.sin(self.angle_r), math.cos(self.angle_r)]])

        # other constants:
        self.DIFFUSE_SWEEP_RAYS = self.config.diffuse_sweep_rays
        # self.num_photoreceptors = 2 # start with two 
        self.reset_position(self.config.x, self.config.y)

    def get_config_state(self,):
        config = Prodict()
        config.fov_r = self.fov_r
        config.angle_r = self.angle_r
        config.sensor_size = self.sensor_size
        config.visual_acuity = self.visual_acuity
        config.imaging_model = self.imaging_model
        config.diffuse_sweep_rays = self.DIFFUSE_SWEEP_RAYS
        config.animal_direction = self.animal_direction
        config.animal_idx = self.animal_idx
        config.closed_pinhole_percentage = self.closed_pinhole_percentage
        return config 
    
    def load_from_config(self, config):
        """
        LoadFromConfig: Loads directly from a saved checkpoint of the eye's mutation. 
        """
        
        if isinstance(config, str):
            with open(config, "r") as ymlfile:
                dict_cfg = yaml.load(ymlfile, Loader=yaml.Loader)
                config = Prodict.from_dict(dict_cfg)

        self.fov_r = config.fov_r
        self.angle_r = config.angle_r
        self.sensor_size = config.sensor_size
        self.visual_acuity = config.visual_acuity
        self.imaging_model = config.imaging_model
        self.DIFFUSE_SWEEP_RAYS = config.diffuse_sweep_rays
        self.animal_direction = config.animal_direction
        self.animal_idx = config.animal_idx
        
        # update params
        self.fov_r = self._update_fov(self.fov_r)
        self.angle_r = self._update_angles(self.angle_r)
        self.f_dir = (np.cos(self.angle_r), np.sin(self.angle_r)) # tuple 
        self.f = (self.sensor_size/2) / (np.tan(self.fov_r/2)) # focal length
        self.rot = np.array([[math.cos(self.angle_r), math.sin(self.angle_r)], # rotation matrix
                             [-math.sin(self.angle_r), math.cos(self.angle_r)]])


    def render_pixel(self, dx, dy, geometry, num_photoreceptors, 
                     photon_noise=False, scale_final_intensity=False, 
                     visual_accuity=False, distance_weight=False):
        """
        final_intensity: 
            - Adds up radiance from each of the photoreceptors. 
        raw_photoreceptor_output: 
            - List of Dicts: 
                - Rays: [list of rays that make up the photoreceptor]
                - intensity: intensity of the photoreceptor
        """
        # set num photoreceptors
        self.num_photoreceptors = num_photoreceptors
        # self.fixed_incident_photon_flux = int(self.num_photoreceptors/10) # 1/10th of the total flux

        final_intensity = 0.
        raw_photoreceptor_output = self.render(dx, dy, geometry)

        if len(raw_photoreceptor_output) > 0: 
            if self.imaging_model == 'lens':
                raw_photoreceptor_output = self.forward_model(raw_photoreceptor_output, visual_accuity, distance_weight)

            for r in raw_photoreceptor_output:
                final_intensity += r.raw_radiance

            if photon_noise:
                # add camera noise, each ray is a photon striking the eye
                final_intensity = self.add_camera_noise(final_intensity)

            if not photon_noise:
                # add some noise, if photon noise is disabled
                final_intensity += np.random.normal(loc=0, scale=0.1)

            # scale by num_photoreceptors (should be turned off with aperture)
            if scale_final_intensity:
                # import pdb; pdb.set_trace()
                # final_intensity = final_intensity/self.num_photoreceptors
                final_intensity = final_intensity/len(raw_photoreceptor_output)
                final_intensity = np.clip(final_intensity, 0, 1)

            # clip to 0,1
            # final_intensity = np.clip(final_intensity, 0., 1.0)

        # one idea could be to look at each ray as a photoreceptor -> instead of just outputing the final intensity!
        return final_intensity, raw_photoreceptor_output

    def reset_position(self, mx, my):
        self.x = mx
        self.y = my
        self.position = np.array([[self.x], [self.y]])

    def _update(self, dx, dy):
        self.x += dx
        self.y += dy
        self.position = np.array([[self.x], [self.y]])

    def _update_fov(self, fov_r):
        # update fov 
        fov_r = np.clip(fov_r, math.radians(30), math.radians(150)) 
        # if self.imaging_model == 'simple':
        # elif self.imaging_model == 'lens':
        #     fov_r = np.clip(fov_r, math.radians(30), math.radians(110)) 
        # else: 
        #     ValueError("{} not found".format(self.imaging_model))
        return fov_r

    def _update_angles(self, angle_r):
        """
        clips angles to avoid rendering issues...
        """
        if self.constraint_angles: 
            if self.animal_direction == 'left':
                angle_r = np.clip(angle_r, math.radians(0), math.radians(80))
            elif self.animal_direction == 'right':
                angle_r = np.clip(angle_r, math.radians(110), math.radians(180))
            else:
                raise ValueError("{} not found.".format(self.animal_direction))
        return angle_r

    def update_pixel_config(self, dfov=None, dangle=None, dsensor_size=None):
        if dfov: 
            self.fov_r += dfov

        if dangle: 
            self.angle_r += dangle
            
        if dsensor_size: 
            self.sensor_size += dsensor_size
            self.sensor_size = np.clip(self.sensor_size, 10, 75)

        self.fov_r = self._update_fov(self.fov_r)
        self.angle_r = self._update_angles(self.angle_r)
        self.f_dir = (np.cos(self.angle_r), np.sin(self.angle_r)) # tuple 
        self.f = (self.sensor_size/2) / (np.tan(self.fov_r/2)) # focal length
        self.rot = np.array([[math.cos(self.angle_r), math.sin(self.angle_r)], # rotation matrix
                             [-math.sin(self.angle_r), math.cos(self.angle_r)]])

    def _create_sensor_plane(self, num_photoreceptors):
        sensor_x = self.f * np.ones(num_photoreceptors) # will use these points to create a line
        sensor_y = np.linspace(-self.sensor_size/2, self.sensor_size/2, num_photoreceptors)
        sensor_line = np.vstack([sensor_x, sensor_y])
        # import pdb; pdb.set_trace()
        self.sensor_line = self.position + np.matmul(self.rot, sensor_line)
        # just mute points from here that are ommitted due to the aperture

    def _create_aperture_plane(self, num_points_aperture):
        x = self.f * np.ones(num_points_aperture) # will use these points to create a line
        y = np.linspace(-self.sensor_size/2, self.sensor_size/2, num_points_aperture)
        line = np.vstack([x, y])
        # offset position by f * (dx, dy)
        # import pdb; pdb.set_trace()
        # print("fdir: {}, {}, {}".format(self.position, self.f, self.f_dir))
        # offset = self.position - np.array(self.f * np.array(self.f_dir)).reshape(2,1)
        # self.aperture_line = offset + np.matmul(self.rot, line)
        # self.aperture_line = offset + np.matmul(self.rot, line)

        ###
        x = self.position[0] * np.ones(num_points_aperture)
        y = self.position[1] + np.linspace(-self.sensor_size/2, self.sensor_size/2, num_points_aperture)
        line = np.vstack([x, y])
        self.aperture_line = line 

    def render(self, dx, dy, geometry):
        # update position
        self._update(dx, dy)
        return self._render_lens_eye(geometry)

    def _render_simple_eye(self, geometry):
        """
        If the photoreceptor are simple eyes then use this, 
        Without a lens, the photoreceptor just integrates light
        from all directions. 
        """
        # sample rays from a sensor plane
        # TODO: why do you need to sample from a sensor_plane for this? 
        if self.num_photoreceptors < 2: 
            self._create_sensor_plane(2)
        else:
            self._create_sensor_plane(self.num_photoreceptors)
        # angle between the first point and the last 
        # w1 = Wall(self.position, self.sensor_line[0])
        # w2 = Wall(self.position, self.sensor_line[-1])
        # start_angle = w1.angle
        # end_angle = w2.angle
        if self.animal_direction == 'right':
            start_angle = self.angle_r #+ self.fov_r/2 #+ np.pi
            end_angle = self.angle_r - self.fov_r #+ np.pi
        if self.animal_direction == 'left':
            start_angle = self.angle_r #+ self.fov_r/2 #+ np.pi
            end_angle = self.angle_r + self.fov_r #+ np.pi

        angles = np.random.uniform(start_angle, end_angle, size=[self.DIFFUSE_SWEEP_RAYS])
        photoreceptors = []
        for i in range(self.num_photoreceptors):
            _ray = {}
            sub_rays = []
            x, y = self.sensor_line[0][i], self.sensor_line[1][i]
            total_radiance = 0.
            for ray_angle in angles: 
                # hacky fix since this is annoying... 
                _min_angle_ = 220
                _max_angle_ = 300
                if ray_angle > _min_angle_ and ray_angle < _max_angle_:
                    if np.abs(_max_angle_ - ray_angle) < np.abs(_min_angle_ - ray_angle):
                        ray_angle = _max_angle_
                        continue 
                    else:
                        ray_angle = _min_angle_
                        continue 
                r = Ray(x,y, ray_angle)
                ret = geometry.check_ray_collision(r) # returns intensity [0,1], distance to wall
                if ret is not None: 
                    closest, closestPoint, rgb = ret
                    r.intensity = rgbfloat2grey(rgb)
                    r.rgb = rgb
                    r.collision_point = closestPoint
                    total_radiance += r.intensity # should be 0, 1
                    sub_rays.append(r)
                else: 
                    # don't append the ray.. just ignore it.
                    pass 

            _ray['raw_radiance'] = total_radiance/self.DIFFUSE_SWEEP_RAYS # intensity per photoreceptor
            # _ray['raw_radiance'] = total_radiance # intensity per photoreceptor
            _ray['rays'] = sub_rays
            photoreceptors.append(Prodict.from_dict(_ray))

        return photoreceptors

    def _render_lens_eye(self, geometry):
        """
        If the photoreceptor have 'lens' then use this! 
        Returns: 
            List of dicts of photoreceptors: 
                - ray: ray attributes
                - raw_radiance: radiance recorded at that photoreceptor
        """
        # set visual acuity field
        x = np.arange(0, self.num_photoreceptors)
        self.visual_field = gaussian(x, self.num_photoreceptors/2, self.visual_acuity)

        # sample rays from a sensor plane
        self._create_sensor_plane(self.num_photoreceptors)
        # create aperture 
        NUM_POINTS_APERTURE = self.sensor_size  * 5
        self._create_aperture_plane(NUM_POINTS_APERTURE)
        # this could be a coded aperture at some point
        self.aperture_mask = np.zeros(NUM_POINTS_APERTURE)
        self.closed_pinhole_percentage = np.clip(0, 1, self.closed_pinhole_percentage)
        num_closed = map_one_range_to_other(self.closed_pinhole_percentage, 0, int(NUM_POINTS_APERTURE/2), 0, 1)
        num_closed = int(np.round(num_closed))
        if num_closed > 0:
            # 1 means it's closed. 
            self.aperture_mask[0:num_closed] = 1
            self.aperture_mask[-num_closed:-1] = 1
            self.aperture_mask[-1] = 1 

        photoreceptors = []

        for i in range(self.num_photoreceptors):
            _ray = {}
            sub_rays = []
            total_radiance = 0.
            px, py = self.sensor_line[0][i], self.sensor_line[1][i]
            # Photoreceptor accepts light from all points... 
            for j in range(len(self.aperture_mask)):
                if self.aperture_mask[j] == 1: 
                    # aperture is closed
                    continue
            # for j in range(1):
                ap_x, ap_y = self.aperture_line[0][j], self.aperture_line[1][j]
                # ap_x, ap_y = self.x, self.y 
                w = Wall((px, py), (ap_x, ap_y))
                # w = Wall((px, py), (self.x, self.y))
                # angle = math.atan2( y - self.y, x - self.x)
                if self.f_dir[0] <= 0.:
                    ray_angle = w.angle
                    # ray_angle = angle
                else:
                    ray_angle = w.angle + math.radians(180.)

                    # ray_angle = angle + math.radians(180.)

                # hacky fix since this is annoying... 
                # _min_angle_ = 220
                # _max_angle_ = 300
                # if ray_angle > _min_angle_ and ray_angle < _max_angle_:
                #     if np.abs(_max_angle_ - ray_angle) < np.abs(_min_angle_ - ray_angle):
                #         ray_angle = _max_angle_
                #         # continue 
                #     else:
                #         ray_angle = _min_angle_
                        # continue 
                # if self.f_dir[0] <= 0. and self.f_dir[1] <= 0: # neg and neg
                #     print('neg and neg rendering', math.degrees(w.angle))
                #     ray_angle = w.angle
                # elif self.f_dir[0] <= 0. and self.f_dir[1] >= 0: # neg and pos
                #     print('neg and pos rendering', math.degrees(w.angle))
                #     ray_angle = w.angle 
                # elif self.f_dir[0] >= 0. and self.f_dir[1] >= 0: # pos and neg
                #     print('pos and neg rendering', math.degrees(w.angle))
                #     ray_angle = w.angle
                # else:
                #     # pos and pos
                #     ray_angle = w.angle #+ math.radians(90)
                #     ray_angle = math.radians(180) + w.angle 
                #     print('pos and pos rendering', math.degrees(w.angle))

                r = Ray(px,py, ray_angle)
                ret = geometry.check_ray_collision(r) # returns intensity [0,1], distance to wall
                if ret is not None: 
                    closest, closestPoint, rgb = ret
                    r.intensity = rgbfloat2grey(rgb)
                    ## add noise per ray
                    # r.intensity += 
                    r.rgb = rgb
                    r.collision_point = closestPoint
                    total_radiance += r.intensity # should be 0, 1
                    sub_rays.append(r)
                else: 
                    # don't append the ray.. just ignore it.
                    #  TODO(): if there is no intersection it probably intersects with the maze 
                    # boundary. we should just do line-line interesection with those values. 
                    pass 

            _ray['raw_radiance'] = total_radiance/np.maximum(1, len(sub_rays)) # intensity per photoreceptor
            # _ray['raw_radiance'] = total_radiance # intensity per photoreceptor
            _ray['rays'] = sub_rays
            photoreceptors.append(Prodict.from_dict(_ray))

        return photoreceptors
    
    def forward_model(self, photoreceptors, visual_accuity=False, distance_weight=False):
        """
        photoreceptors: list of N photoreceptors sorted from [0, sensor_size]
            each photoreceptors:
               {intensity, distance to wall, direction}
        visual_accuity:
            weight by visual_accuity
        distance_weight:
            weight by distance_weight
        """

        # get max distance to the wall
        if distance_weight:
            max_dist = max([ray.distance for ray in photoreceptors.rays])

        for i, r in enumerate(photoreceptors):
            if distance_weight:
                # weight by distance to the wall
                dist = r.distance
                r.raw_radiance = r.raw_radiance * dist/max_dist
            if visual_accuity:
                # weight by visual acuity at that point.
                r.raw_radiance = r.raw_radiance * self.visual_field[i]

        return photoreceptors

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

class SinglePixelApertureMask: 
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
