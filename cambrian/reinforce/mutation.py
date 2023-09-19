import numpy as np
from cambiran.reinforce.ppo import CambrianPPO

## todo: should we make it a gym env?
class MutationHelper():
    def __init__(self, config):
        super.__init__()

    def mutate(self, animal_config, mutation_type=None): 
        if mutation_type == None: 
            mutate_func = np.random.choice([self.add_photoreceptors, self.add_eye, self.update_config, self.add_lens])
        

    def add_photoreceptors(self, conf):
        conf.init_photoreceptors +=  conf.increment_photoreceptor
        conf.init_photoreceptors = np.clip(conf.init_photoreceptors, 10, conf.max_photoreceptors)
        return conf

    def add_eye(self, conf):
        ## adds eyes symmetrically on each side. 
        if (conf.num_pixels + 2) > conf.max_num_eyes_per_side:
            # there is no space for more eyes, increment count of max_num_eyes 
            conf.max_num_eyes_per_side += 2
            # increment receptor count when increasing max eyes? 
            conf.init_photoreceptors += conf.increment_photoreceptor
            conf.init_photoreceptors += conf.increment_photoreceptor
        
        # create config for left
        direction = 'left'
        ss = conf.sensor_size[-1]
        # ss = np.mean(conf.sensor_size)
        fov = np.mean(conf.fov) # just the average fov
        if conf.angle_overwrite_to_normal:
            angle = 0 # doesnt matter if angle_overwrite_to_normal = True
        else:
            raise ValueError("Please set angle_overwrite_to_normal to true...")
        
        simple_p = float(len(np.where(conf.imaging_model == 'simple')[0]))/len(conf.imaging_model)            
        imaging_model = np.random.choice(['simple', 'lens'], p=[simple_p, 1-simple_p])
        conf = add_eye_to_conf(conf, fov, angle, ss, direction, imaging_model)
        # add eye with the same configuration to the right
        direction = 'right'
        conf = add_eye_to_conf(conf, fov, angle, ss, direction, imaging_model)
        return conf

    def update_config(self, conf, update_method='random'):
        # update the fov 
        update_conf = np.random.uniform(-10, 10)
        for i in conf.fov: 
            conf.fov[i] += update_conf
            conf.fov[i] = np.clip(conf.fov[i], 30, 150)
        # update the visual acuity

        return conf

    def add_lens(self,  conf):
        pass


def add_eye_to_conf(conf, fov, angle, sensor_size, direction, imaging_model):
    conf.direction.append(direction)
    conf.sensor_size.append(sensor_size)
    conf.fov.append(fov)
    conf.angle.append(angle)
    conf.imaging_model.append(imaging_model)
    conf.num_pixels += 1
    return conf