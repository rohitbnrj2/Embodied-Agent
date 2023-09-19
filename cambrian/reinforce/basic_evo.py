from copy import deepcopy
import functools

from cambrian.evolution_envs.animal import OculozoicAnimal


class TestEvoRun:
    """
    TestEvoRun is a very simple evo run, where n agents mutate into kn children through asexual 
    after every evolution epoch and the best n children are selected for further mutation. 
    DERL is more complex than this.
    """
    
    def __init__(self, 
                 evo_config,
                 init_animal_config, 
                 generation_wise_envs,
                 init_population_size = 5,
                 best_agents_per_generation = 5,
                 
                 num_mutations_per_agent = 2, 
                 max_population_size = 10,
                 max_generations = 5):
        
        # initialize init_population many first generation animals.
        
        self.population = []
        
        for _ in range(init_population_size):
            animal = OculozoicAnimal(init_animal_config)
            self.population.append(animal)
        
        self.max_population_size = max_population_size
        self.max_generations = max_generations
        self.num_mutations_per_agent = num_mutations_per_agent
        
        
        assert len(self.generation_wise_mazes) == max_generations
        
        self.generation_wise_envs = generation_wise_envs
        self.best_agents_per_generation = best_agents_per_generation
    
    def mutate_single_agent(self, agent, mutation_type, mut_args, num_mutations = 2):
        
        # mutate a single agent through asexual reproduction, this also kills the parent.
        
        children = []
        
        for _ in range(num_mutations):
            
            agent_copy = deepcopy(agent)           
            agent_copy.mutate(mutation_type, mut_args)
            children.append(agent_copy)
        
        return children
    
    
    def mutate_current_generation(self, ):
        
        #mutates each agent in the current generation through asexual reproductions
        
        next_generation = []
        
        for agent in self.population:
            
            children = mutate_single_agent(agent)
            
            next_generation += children
        
        self.population = next_generation
        
    
    def train_single_agent(self, agent, env):
        
        #to do
        raise NotImplementedError
        
    
    def train_new_generation(self, evo_epoch = 0):
        
        env = self.generation_wise_envs[evo_epoch]
        
        for agent in self.population:
            self.train_single_agent(agent, env)
            
            
    def parse_reward_file(reward_file):
        #to do
        
        raise NotImplementedError
    
    
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
        
        
    
    def save_agent(agent, evo_epoch):
        # to do
        raise NotImplementedError
        
        
    def save_generation(self, evo_epoch):
        for agent in self.population:
            save_agent(agent, evo_epoch)
        
        
    def run(self,):
        
        #basic evo algorithm!
        
        for evo_epoch in range(self.max_generations):
            
            self.mutate_current_generation()
            
            self.train_new_generation(evo_epoch)
            
            self.save_generation(evo_epoch)
            
            self.select_fittest_agents()     

if __name__ == "__main__":
    pass



