import random
from cambrian.renderer.ray import Ray
from cambrian.renderer.wall import Wall
import numpy as np
import math
from typing import List, Tuple
from prodict import Prodict


from cambrian.utils.renderer_utils import gaussian

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
            No Lens                Lens
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
    def __init__(self, config):
        self.config = config
        self.update_pixel_params(self.config)

    def update_pixel_params(self, config):
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

        self.f = (self.sensor_size/2) / (np.tan(self.fov_r/2)) # focal length
        self.rot = np.array([[math.cos(self.angle_r), math.sin(self.angle_r)], # rotation matrix
                             [-math.sin(self.angle_r), math.cos(self.angle_r)]])

        # other constants:
        self.DIFFUSE_SWEEP_RAYS = 90
        self.num_photoreceptors = 2 # start with two 
        self.reset(self.config.x, self.config.y)

    def render_pixel(self, dx, dy, geometry, num_photoreceptors, 
                     photon_noise=False, scale_final_intensity=False, 
                     visual_accuity=True, distance_weight=False):
        # set num photoreceptors
        self.num_photoreceptors = num_photoreceptors
        # self.fixed_incident_photon_flux = int(self.num_photoreceptors/10) # 1/10th of the total flux

        final_intensity = 0.

        raw_photoreceptor_output = self.render(dx, dy, geometry)
        if self.imaging_model == 'lens':
            raw_photoreceptor_output = self.forward_model(raw_photoreceptor_output, visual_accuity, distance_weight)

        for r in raw_photoreceptor_output:
            final_intensity += r.raw_radiance

        if photon_noise:
            # add camera noise, each ray is a photon striking the eye
            final_intensity = self.add_camera_noise(final_intensity)

        # scale by num_photoreceptors (should be turned off with aperture)
        if scale_final_intensity:
            final_intensity = final_intensity/self.num_photoreceptors

        if not photon_noise:
            # add some noise, if photon noise is disabled
            final_intensity += np.random.normal(loc=0, scale=0.1)

        # clip to 0,1
        final_intensity = np.clip(final_intensity, 0., 1.0)

        # one idea could be to look at each ray as a photoreceptor -> instead of just outputing the final intensity! 
        return final_intensity, raw_photoreceptor_output

    def reset(self, mx, my):
        self.x = mx
        self.y = my
        self.position = np.array([[self.x], [self.y]])

    def _update(self, dx, dy):
        self.x += dx
        self.y += dy
        self.position = np.array([[self.x], [self.y]])

    def update_pixel_config(self, dfov, dangle=None, dsensor_size=None):
        if dfov: 
            self.fov_r += dfov
        if dangle: 
            self.angle_r = dangle
        if dsensor_size: 
            self.sensor_size = dsensor_size

        self.f_dir = (np.cos(self.angle_r), np.sin(self.angle_r)) # tuple 
        self.f = (self.sensor_size/2) / (np.tan(self.fov_r/2)) # focal length
        self.rot = np.array([[math.cos(self.angle_r), math.sin(self.angle_r)], # rotation matrix
                             [-math.sin(self.angle_r), math.cos(self.angle_r)]])


    def _create_sensor_plane(self, num_photoreceptors):
        sensor_x = self.f * np.ones(num_photoreceptors) # will use these points to create a line
        sensor_y = np.linspace(-self.sensor_size/2, self.sensor_size/2, num_photoreceptors)
        sensor_line = np.vstack([sensor_x, sensor_y])
        self.sensor_line = self.position + np.matmul(self.rot, sensor_line)
        # just mute points from here that are ommitted due to the aperture

    def render(self, dx, dy, geometry):
        # update position
        self._update(dx, dy)
        
        if self.imaging_model == 'simple':
            return self._render_simple_eye(geometry)
        elif self.imaging_model == 'lens':
            return self._render_lens_eye(geometry)
        else:
            raise ValueError("{} not exist".format(self.imaging_model))

    def _render_simple_eye(self, geometry):
        """
        If the photoreceptor are simple eyes then use this, 
        Without a lens, the photoreceptor just integrates light
        from all directions. 
        """
        # sample rays from a sensor plane
        # TODO: why do you need to sample from a sensor_plane for this? 
        self._create_sensor_plane(self.num_photoreceptors)
        # angle between the first point and the last 
        w = Wall(self.sensor_line[0], self.sensor_line[-1])
        start_angle = w.angle
        end_angle = w.angle + np.pi # a + 180
        angles = np.random.uniform(start_angle, end_angle, size=[self.DIFFUSE_SWEEP_RAYS])
        photoreceptors = []
        for i in range(self.num_photoreceptors):
            _ray = {}
            distance_points, sub_rays = [], []
            x, y = self.sensor_line[0][i], self.sensor_line[1][i]
            total_radiance = 0.
            for ray_angle in angles: 
                r = Ray(x,y, ray_angle)
                ret = geometry.check_ray_collision(r) # returns intensity [0,1], distance to wall
                if ret is not None: 
                    closest, closestPoint, intensity = ret
                    r.intensity = intensity
                    r.collision_point = closestPoint
                    total_radiance += intensity # should be 0, 1
                    sub_rays.append(r)
                else: 
                    # don't append the ray.. just ignore it.
                    pass 

            _ray['raw_radiance'] = total_radiance # intensity per photoreceptor
            _ray['rays'] = sub_rays
            photoreceptors.append(Prodict.from_dict(_ray))

        return photoreceptors

    def _render_lens_eye(self, geometry):
        """
        If the photoreceptor have 'lens' then use this! 
        """
        # set visual acuity field
        x = np.arange(0, self.num_photoreceptors)
        self.visual_field = gaussian(x, self.num_photoreceptors/2, self.visual_acuity)

        # sample rays from a sensor plane
        self._create_sensor_plane(self.num_photoreceptors)
        photoreceptors = []
        total_radiance = 0.
        for i in range(self.num_photoreceptors):
            _ray = {}
            x, y = self.sensor_line[0][i], self.sensor_line[1][i]
            w = Wall((x,y), (self.x, self.y))
            if self.f_dir[0] <= 0.:
                ray_angle = w.angle
            else:
                ray_angle = w.angle + math.radians(180.)

            r = Ray(x,y, ray_angle)
            ret = geometry.check_ray_collision(r) # returns intensity [0,1], distance to wall
            if ret is not None: 
                closest, closestPoint, intensity = ret
                r.intensity = intensity
                r.collision_point = closestPoint
                total_radiance += intensity 
            else: 
                # don't append the ray.. just ignore it.
                pass 

            _ray['ray'] = [r] # ray or rays that compose the intensity
            _ray['raw_radiance'] = intensity # intensity per photoreceptor

            photoreceptors.append(Prodict.from_dict(_ray))

        return photoreceptors
    
    def forward_model(self, photoreceptors, visual_accuity=True, distance_weight=False):
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
            max_dist = max([ray.distance for ray in photoreceptors])

        for r in photoreceptors:
            intensity = r.intensity
            if distance_weight:
                # weight by distance to the wall
                dist = r.distance
                intensity = intensity * dist/max_dist
            if visual_accuity:
                # weight by visual acuity at that point.
                intensity = intensity * self.visual_field[r]

            photoreceptors[r]['intensity'] = intensity

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
