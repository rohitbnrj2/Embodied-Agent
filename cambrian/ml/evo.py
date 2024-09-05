try:
    import nevergrad as ng  # noqa

    has_nevergrad = True
except ImportError:
    has_nevergrad = False

from cambrian.utils.config import MjCambrianBaseConfig, config_wrapper


@config_wrapper
class MjCambrianEvoConfig(MjCambrianBaseConfig):
    population_size: int
    num_generations: int

    rank: int
    generation: int


if has_nevergrad:
    ng.optimizers.EvolutionStrategy(
        recombination_ratio=0.25, only_offsprings=False, offsprings=8, popsize=16
    ).set_name("CambrianES2", register=True)
