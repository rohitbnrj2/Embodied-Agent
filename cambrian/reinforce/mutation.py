from cambiran.reinforce.ppo import CambrianPPO

## todo: should we make it a gym env?
class CambiranEvolution():
    def __init__(self, config):
        super.__init__()
        self.ppo = CambrianPPO(config)

    def mutate(self): 
        pass 

    def simulate(self, ):
        pass 

    def pick_fittest(self,):
        pass 

    
    def run(self,):
        pass 


if __name__ == "__main__":
    CambiranEvolution()