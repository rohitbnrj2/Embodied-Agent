from cambrian.utils.config import MjCambrianBaseConfig, config_wrapper

@config_wrapper
class MjCambrianEvoConfig(MjCambrianBaseConfig):

    num_nodes: int
    num_agents_per_node: int
    local_rank: int
    global_rank: int
    generation: int
    num_generations: int