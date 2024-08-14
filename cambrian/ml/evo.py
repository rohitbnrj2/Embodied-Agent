from cambrian.utils.config import MjCambrianBaseConfig, config_wrapper


@config_wrapper
class MjCambrianEvoConfig(MjCambrianBaseConfig):
    population_size: int
    num_generations: int

    rank: int
    generation: int
