import functools
from copy import deepcopy
from prodict import Prodict
from env_v2 import make_env, BeeEnv

# +
project_path = "/home/gridsan/bagrawalla/EyesOfCambrian-main/"
maze_folder_path = project_path + "new_mazes_paint/"

#train generation[i] in maze[i]
generation_wise_mazes = ["right_turn_constant_width.png",
                         "right_turn_changing_width.png",
                         "simple_forward.png"]
init_config_file = project_path + "configs_evo/debug.yaml"


# -

def set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)



class TestEvoRun:
    """
    TestEvoRun is a very simple evo run, where n agents mutate into kn children through asexual 
    after every evolution epoch and the best n children are selected for further mutation. 
    DERL is more complex than this.
    """
    
    def __init__(self, 
                 init_animal_config, 
                 generation_wise_mazes,
                 init_population_size = 5,
                 best_agents_per_generation = 5,
                 
                 num_mutations_per_agent = 2, 
                 max_population_size = 10,
                 max_generations = 5):
        
                
        # initialize init_population many zeroth generation animals.
        
        self.population = [init_animal_config for _ in range(init_population_size)]        
        self.max_population_size = max_population_size
        self.max_generations = max_generations
        self.num_mutations_per_agent = num_mutations_per_agent
        self.generation_wise_mazes = generation_wise_mazes
        
        
        assert len(self.generation_wise_mazes) == max_generations
        
        self.generation_wise_envs = generation_wise_envs
        self.best_agents_per_generation = best_agents_per_generation
    
    def save_prodict_as_yaml(prodict, yaml_file_path):
        #saves given prodict into the specified yaml file path
        
        
        raise NotImplementedError
        
        
        
    
    def mutate_single_agent(self, agent, maze_path, num_mutations = 2):
        
        # mutate a single agent through asexual reproduction, this also kills the parent. Puts the new agent in 
        # specified maze_path
        
        children = []
        
        for i in range(num_mutations):
            
            child_cfg = self.create_random_symmetric_mutation(agent, maze_path, i)
            child_cfg_save_path = "{}/{}/evo_epoch = {}/agent_id = {}/config.yml".format(logdir, 
                                                                                         exp_name, 
                                                                                         evo_epoch,
                                                                                         child_cfg.animal_config.id)
            save_prodict_as_yaml(child_cfg, child_cfg_save_path)
            
            children.append(child_cfg_save_path)
            
            
            
        
        return children
    
    def create_random_symmetric_mutation(self, agent, maze_path, i):
        
        #adds or subtracts pixels symmetrically and randomly from agent
        
        allowed_angles = [10.*i for i in range(18)] #allowed angles for pixel placement on either side
        
        max_pixels = 2*len(allowed_angles)
        

        
        agent_config_file = "{}/{}/evo_epoch = {}/agent_id = {}/config.yml".format(cfg.env_config.logdir, 
                                                                                  cfg.env_config.exp_name, 
                                                                                  evo_epoch,
                                                                                  agent_id)
        
        with open(agent_config_file, "r") as ymlfile:
                agent_dict_cfg = yaml.load(ymlfile, Loader=yaml.Loader)
                agent_cfg = Prodict.from_dict(self.dict_cfg)
        
        
        num_parent_pixels = agent_cfg.animal_config.init_configuration.num_pixels
        
        parent_fov = agent_cfg.animal_config.init_configuration.angle
        parent_sensor_size = agent_cfg.animal_config.init_configuration.angle
        parent_direction = agent_cfg.animal_config.init_configuration.angle
        parent_imaging_model = agent_cfg.animal_config.init_configuration.angle
        
        parent_angles = agent_cfg.animal_config.init_configuration.angle
        
        parent_id = agent_cfg.animal_config.id 
        
        
        child_cfg = deepcopy(agent_cfg)
        
        if len(parent_pixel_angles) == max_pixels:
            mutation_type = "subtract_pixel"
        
        elif len(parent_pixel_angles) == 2:
            mutation_type = "add_pixel"
        
        else:
            mutation_type = random.choice(["add_pixel", "subtract_pixel"])
        
        
        child_angles = get_child_angles(parent_angles, allowed_angles, mutation_type)
        
        child_cfg.scene_config.load_path = maze_path
        
        child_cfg.animal_config.init_configuration.angle = child_angles
        
        child_cfg.animal_config.init_configuration.fov = parent_fov + parent_fov[-2:]
        child_cfg.animal_config.init_configuration.sensor_size = parent_sensor_size + parent_sensor_size[-2:]
        child_cfg.animal_config.init_configuration.direction = parent_direction + parent_direction[-2:]
        child_cfg.animal_config.init_configuration.imaging_model = parent_imaging_model + parent_imaging_model[-2:]
        
        child_cfg.animal_config.init_configuration.num_pixels = len(child_angles)
        
        child_cfg.animal_config.id = parent_id + f"_{i}"
        
        
        return child_cfg
        
    
    def get_child_angles(parent_angles, allowed_angles, mutation_type):
        
        pixels_each_side = int(len(parent_angles)/2)
                
        if mutation_type == "add_pixel":
            
            permuted_angles = random.permutation(allowed_angles)
            
            for angle in permuted_angles:
                if angle not in parent_angles:
                    
                    child_angles = parent_angles + [angle, angle] #once for both left and right
                    
            
            
            
        
        elif mutation_type == "subtract_pixel":
            
            remove_index = random.choice([i for i in range(pixels_each_side)])
            
            remove_element = parent_angles[2*remove_index]
            
            child_angles = deepcopy(parent_angles)
            
            child_angles.remove(remove_element) #once for left
            child_angles.remove(remove_element) #again for right
            
            
        
        else:
            raise NotImplementedError
        
        return child_angles
        
        
    def mutate_current_generation(self, ):
        
        #mutates each agent in the current generation through asexual reproductions
        
        next_generation = []
        
        for agent in self.population:
            
            children = mutate_single_agent(agent)
            
            next_generation += children
        
        self.population = next_generation
        
    
    def train_single_agent(self, agent_config_file, evo_epoch, trainsteps_per_agent = 1e4):
        
        #trains and saves single agent in the evo_epoch folder
        
        with open(agent_config_file, "r") as ymlfile:
                agent_dict_cfg = yaml.load(ymlfile, Loader=yaml.Loader)
                cfg = Prodict.from_dict(agent_dict_cfg)
                
        agent_id = cfg.animal_config.id
        
        print("____________________________________________________________________")
        print("")
        print(f"Training agent with agent_id = {agent_id}")
        print("")
        print("_____________________________________________________________________")
        
        logdir = Path("{}/{}/evo_epoch = {}/agent_id = {}/".format(cfg.env_config.logdir, 
                                                                   cfg.env_config.exp_name, 
                                                                   evo_epoch,
                                                                   agent_id))
        if not logdir.exists():
            print("creating directory: {}".format(str(logdir)))
            Path.mkdir(logdir)
        
        
        ppo_path = Path("{}/{}/evo_epoch = {}/agent_id = {}/ppo/".format(cfg.env_config.logdir, 
                                                                         cfg.env_config.exp_name, 
                                                                         evo_epoch,
                                                                         agent_id))
        
        if not ppo_path.exists():
            print("creating directory: {}".format(str(ppo_path)))
            Path.mkdir(ppo_path)
            

        env = SubprocVecEnv([make_env(rank=i, seed=cfg.env_config.seed, config_file= agent_config_file, idx=i) 
                            for i in range(cfg.env_config.num_cpu)])
        env = VecMonitor(env, str(ppo_path))

        callback = SaveOnBestTrainingRewardCallback(check_freq=cfg.env_config.check_freq, log_dir=str(ppo_path), verbose=2)
        
        print("Init policy from Scratch")
        policy_kwargs = dict(features_extractor_class=MultiInputFeatureExtractor,
                             features_extractor_kwargs=dict(features_dim = 256),)
        
        model = PPO("MultiInputPolicy",
                env,
                n_steps=cfg.env_config.n_steps,
                batch_size=cfg.env_config.batch_size,
                policy_kwargs=policy_kwargs,
                verbose=2)
        
        timesteps = 1e3
        start = time.time()

        print("Training now...")
        model.learn(total_timesteps=int(timesteps), callback=callback)
        save_path = "{}/{}".format(logdir, "final_rl_model.pt")
        model.save(save_path)
        
        agent.reward_file = "{}/{}/evo_epoch = {}/agent_id = {}/ppo/monitor.csv".format(cfg.env_config.logdir, 
                                                                                        cfg.env_config.exp_name, 
                                                                                        evo_epoch,
                                                                                        agent_id)
        
        
    
    def train_new_generation(self, evo_epoch = 0):
        print("_____________________________________________________")
        print(f"Starting evo_epoch = {evo_epoch}")
        print("_____________________________________________________")
        
        env = self.generation_wise_envs[evo_epoch]
        
        for agent in self.population:
            self.train_single_agent(agent, env)
            
            
    def parse_reward_file(reward_file, num_last_episodes = 10, ignore = 2):
        
        x = [0]
        y = []
        with open(reward_file, 'r') as csvfile:
            
            episodes = csv.reader(csvfile, delimiter = ',')
            count = 0
            for ep in episodes:
                if count < ignore:
                    count += 1
                    continue
                    
                a,b = ep[0], ep[1]
                x.append(x[-1] + float(b))
                y.append(float(a))
                
        return np.mean(y[-1*num_last_episodes : ])
                
                
    
    def compare_agents(agent1, agent2):
        
        rew1 = parse_reward_file(agent1.reward_file)
        rew2 = parse_reward_file(agent2.reward_file)
        
        if rew1 > rew2:
            return 1
        
        elif rew1 < rew2:
            return -1
        
        else:
            return 0
            
    
    def select_fittest_agents(self,):
        
        #select top best_agents_per_generation agents
        
        self.population = sorted(self.population, key=functools.cmp_to_key(compare_agents), reverse = True)
        
        self.population = self.population[: self.best_agents_per_generation]
        
        
    
            
    def run(self,):
        
        #basic evo algorithm!
        
        for evo_epoch in range(self.max_generations):
            
            self.mutate_current_generation()
            
            self.train_new_generation(evo_epoch)
            
            self.select_fittest_agents()     

if __name__ == "__main__":
    pass





