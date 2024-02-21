from typing import Dict, Tuple, List, Optional
from pathlib import Path
import os

import numpy as np

from cambrian.animal import MjCambrianAnimal
from cambrian.utils.config import (
    MjCambrianConfig,
)
from cambrian.utils.logger import get_logger

Fitness = float


class MjCambrianPopulation:
    """This class is an abstraction around a cambrian population.

    A population consists of the current animals actively being trained. At any one
    point, there are `config.size` number of animals being trained.

    Mutations are also implemented here, where mutations can be of two types: mutation
    or crossover. A mutation is asexual in that it takes a single animal and augments
    it's genes in some way. Crossover results from taking two animals and combining
    their genes to create a new animal; some random mutations may happen in crossover,
    as well.

    As an animal trains, it's fitness is monitored. This class is responsible for
    efficiently parsing all the outputs and keeping track of the animals with the
    highest fitness. Each animal saves a `monitor.csv` file that contains the fitness
    and a `config.yaml` file that contains the config used to train the animal.

    Currently, the fitness is calculated as the average reward over the last 1000 (or
    the all rewards if there are less than 1000) timesteps of training. This is
    calculated by parsing the `monitor.csv` file.

    Args:
        initial_config (MjCambrianPConfig): The initial config.
        logdir (Path | str): The path to the base directory animal logs should be saved
            to. It is assumed each animal config is saved to a subdirectory of this
            directory under `generation_{generation_num}/rank_{rank_num}`.
    """

    def __init__(self, initial_config: MjCambrianConfig, logdir: Path | str):
        self.initial_config = initial_config
        self.config = initial_config.evo_config.population_config
        self.logdir = Path(logdir)
        self.logger = get_logger()

        self._all_population: Dict[Path, Tuple[Fitness, MjCambrianConfig]] = {}
        self._top_performers: List[Path] = []

    def add_animal(
        self, path_or_config: Path | MjCambrianConfig, fitness: Optional[Fitness] = None
    ):
        """Add an animal to the population. This can be called internally during an
        update or externally, such as when adding the very first animal to the
        population.

        Args:
            path_or_config (Path | MjCambrianConfig): The path to the animal's config or
                the animal's config. If a config is passed, the fitness parameter must
                be provided. Furthermore, it assumed that the animal is being added for
                convenience (like when we need to add an initial animal to the
                population); therefore, the key is set to "placeholder".
            fitness (Optional[Fitness]): The animal's fitness. If None, the fitness is
                calculated from the animal's monitor.csv file.
        """
        if isinstance(path_or_config, Path):
            path = path_or_config
            assert (
                path / "config.yaml"
            ).exists(), f"{path} does not contain config.yaml"

            fitness = self._calculate_fitness(path) if fitness is None else fitness
            try:
                config = MjCambrianConfig.load(path / "config.yaml")
            except AttributeError:
                # This may happen if we try to read concurrently as it's being written
                # Let's just do nothing if this happens
                return
        else:
            assert fitness is not None, "Must provide fitness if config is provided"
            config = path_or_config
            path = "placeholder"
        self._all_population[path] = (fitness, config)

    def update(self):
        """Parse through the logdir and update the current state of the population.

        Each training session saves it's state to a subdirectory in
        `{logdir}/generation_{generation_num}/rank_{rank_num}`.

        TODO: Probably could be more performant
        """
        self.logger.debug("Updating population.")

        for root, dirs, files in os.walk(self.logdir):
            root = Path(root)
            if not root.stem.startswith("generation_"):
                continue

            for dir in dirs:
                dir = Path(dir)
                if not dir.stem.startswith("rank_"):
                    continue

                path = root / dir
                if not (path / "config.yaml").exists():
                    continue

                # Only add the animal if it's finished and not already in the population
                if (path / "finished").exists() and path not in self._all_population:
                    self.add_animal(path)

        # Sort the population and update the current population
        pop = sorted(self._all_population.items(), key=lambda x: x[1][0])
        self._top_performers = [i for i, _ in pop[-self.num_top_performers :]]

    def _calculate_fitness(self, path: Path):
        """Calculates the fitness of the given animal.

        Currently, the fitness is calculated as the reward received on the last
        evaluation episode. This is calculated by parsing the `evaluation.npz` file.
        """
        path /= "evaluations.npz"
        if not path.exists():
            return -np.inf

        with np.load(path) as data:
            rewards = np.mean(data["results"], axis=1)

        fitness = np.max(rewards)
        return fitness

    def select_animal(self) -> MjCambrianConfig:
        """Alias to `select_animals` that selects a single animal."""
        return self.select_animals(1)[0]

    def select_animals(self, num: int) -> List[MjCambrianConfig]:
        """Selects `num` animals from the current population.

        The animal is selected based on the fitness of the animal. The animal with the
        highest fitness is selected.
        """
        self.logger.info(f"Selecting {num} animals.")

        assert (
            num <= self.num_top_performers
        ), f"Cannot select more animals than exist: {num=} < {self.num_top_performers=}"
        animals = np.random.choice(self._top_performers, num, False)
        configs = [self._all_population[a][1].copy() for a in animals]

        return configs

    def spawn_animal(self, generation: int, rank: int) -> MjCambrianConfig:
        """Spawns a new animal based on the current population.

        TODO: add crossover
        """
        self.logger.info(f"Spawning animal with {generation=} and {rank=}.")

        # Select an animal to mutate
        # If generation is 0, we'll read the default config file, select otherwise
        config = self.select_animal() if generation != 0 else self.initial_config.copy()

        # Aliases
        evo_config = config.evo_config
        training_config = config.training_config
        animal_configs = config.env_config.animal_configs
        spawning_config = config.evo_config.spawning_config

        # Set the top performer list to be the current population
        evo_config.top_performers = [str(tp) for tp in self._top_performers]

        # TODO
        # replication_type = MjCambrianSpawningConfig.ReplicationType[
        #     spawning_config.replication_type
        # ]

        # If this is the first generation, we'll use init_num_mutations to facilitate a
        # diverse initial population. Either way, the total number of mutations is
        # randomly selected from 1 to num_mutations.
        if generation == 0:
            num_mutations = np.random.randint(1, spawning_config.init_num_mutations + 1)
        else:
            num_mutations = np.random.randint(1, spawning_config.num_mutations + 1)

        # Mutate the config
        parent_rank = evo_config.generation_config.rank
        self.logger.info(f"Mutating child of {parent_rank=} {num_mutations} times.")
        for animal_config in animal_configs.values():
            # For each animal, we'll mutate it `num_mutations` times
            for _ in range(num_mutations):
                # TODO: add crossover
                animal_configs[animal_config.name] = MjCambrianAnimal.mutate(
                    animal_config,
                    spawning_config.mutations,
                    spawning_config.mutation_options,
                )

        # Update the evo_config to reflect the new generation
        if generation != 0:
            evo_config.parent_generation_config = evo_config.generation_config.copy()
        evo_config.generation_config.rank = rank
        evo_config.generation_config.generation = generation
        generation_logdir = self.logdir / evo_config.generation_config.to_path()
        generation_logdir.mkdir(parents=True, exist_ok=True)

        # Update the training_config. The logdir points to the new generation's logdir,
        # so leave exp_name to empty string since trainer will append it to the logdir
        training_config.seed = self._calc_seed(generation, rank)
        training_config.logdir = str(generation_logdir)
        training_config.exp_name = ""

        # Load the parent's policy if it exists and the user wants to load it
        if (parent := evo_config.parent_generation_config) is not None:
            policy_path = self.logdir / parent.to_path() / "policy.pt"
            if policy_path.exists() and spawning_config.load_policy:
                training_config.policy_path = str(policy_path)

        # Set n_envs to be the max_n_envs divided by the population size
        n_envs = config.evo_config.max_n_envs // self.size
        training_config.n_envs = n_envs

        return config

    def _calc_seed(self, generation: int, rank: int) -> int:
        """Calculates a unique seed for each rank."""

        seed = self.initial_config.training_config.seed
        n_envs = self.initial_config.training_config.n_envs
        n_nodes = self.initial_config.evo_config.num_nodes
        n_ranks_per_node = self.initial_config.evo_config.population_config.size
        n_ranks_per_population = n_nodes * n_ranks_per_node

        return seed + n_envs * (generation * n_ranks_per_population + rank)

    # ========

    @property
    def size(self) -> int:
        return self.config.size

    @property
    def num_top_performers(self) -> int:
        return self.config.num_top_performers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataclass/YAML Tester")

    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument(
        "-o",
        "--override",
        "--overrides",
        dest="overrides",
        action="extend",
        nargs="+",
        type=str,
        help="Override config values. Do <config>.<key>=<value>",
        default=[],
    )
    parser.add_argument(
        "-n",
        "--num-nodes",
        type=int,
        help="Number of nodes to spawn",
        default=4,
    )

    args = parser.parse_args()
    config = MjCambrianConfig.load(args.config, overrides=args.overrides)

    pop = MjCambrianPopulation(config, "logs")
    config.training_config.n_envs = config.evo_config.max_n_envs // pop.size
    config.evo_config.num_nodes = args.num_nodes

    seeds = set()
    for generation in range(config.evo_config.num_generations):
        for node in range(config.evo_config.num_nodes):
            size = config.evo_config.population_config.size
            for rank in range(node * size, node * size + size):
                for env in range(config.evo_config.max_n_envs // size):
                    seed = pop._calc_seed(generation, rank) + env
                    print(
                        f"Generation {generation}, Node {node}, Rank {rank}, Env {env}: {seed}"
                    )
                    if seed in seeds:
                        print(f"Duplicate seed: {seed}")
                    seeds.add(seed)
