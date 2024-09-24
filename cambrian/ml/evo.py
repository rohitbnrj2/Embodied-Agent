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
    CambrianES = ng.optimizers.EvolutionStrategy(
        recombination_ratio=0.25,
        only_offsprings=False,
        offsprings=8,
        popsize=16,
    ).set_name("CambrianES", register=True)
    ng.optimizers.Chaining(
        [
            ng.optimizers.RandomSearch,
            CambrianES,
        ],
        ["tenth"],
    ).set_name("CambrianESRS", register=True)

    CambrianCMA = ng.optimizers.ParametrizedCMA(
        scale=3.0,
        diagonal=True,
        fcmaes=False,
    ).set_name("CambrianCMA", register=True)

    ng.optimizers.Chaining(
        [
            ng.optimizers.RandomSearch,
            CambrianCMA,
        ],
        ["tenth"],
    ).set_name("CambrianCMARS", register=True)
