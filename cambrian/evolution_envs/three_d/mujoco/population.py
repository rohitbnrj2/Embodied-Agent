from typing import Dict, Tuple, List, Optional
from pathlib import Path
import os
from enum import Flag, auto

import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

from config import MjCambrianPopulationConfig, MjCambrianConfig
from animal import MjCambrianAnimal

Fitness = float
AnimalID = Path  # Path to the animal's directory, assumed to be unique


class MjCambrianReplicationType(Flag):
    """Use as bitmask to specify which type of replication to perform on the animal.

    Example:
    >>> # Only mutation
    >>> type = MjCambrianReplicationType.MUTATION
    >>> # Both mutation and crossover
    >>> type = MjCambrianReplicationType.MUTATION | MjCambrianReplicationType.CROSSOVER
    """

    MUTATION = auto()
    CROSSOVER = auto()


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
        config (MjCambrianPopulationConfig): The config to use for the population.
        logdir (Path | str): The path to the base directory animal logs should be saved
            to. It is assumed each animal config is saved to a subdirectory of this
            directory under `generation_{generation_num}/rank_{rank_num}`.
    """

    def __init__(self, config: MjCambrianPopulationConfig, logdir: Path | str):
        self.config = config
        self.logdir = Path(logdir)

        self._all_population: Dict[AnimalID, Tuple[Fitness, MjCambrianConfig]] = {}
        self._top_performers: List[AnimalID] = []

        self._replication_type = MjCambrianReplicationType[self.config.replication_type]

    def add_animal(self, path_or_config: AnimalID | MjCambrianConfig, fitness: Optional[Fitness] = None):
        if isinstance(path_or_config, AnimalID):
            path = path_or_config
            assert (path / "config.yaml").exists(), f"{path} does not contain config.yaml"

            fitness = self._calculate_fitness(path) if fitness is None else fitness
            try:
                config = MjCambrianConfig.load(path / "config.yaml")
            except AttributeError:
                # This may happen if we try to read concurrently as it's being written
                # Let's just continue
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
        for root, dirs, files in os.walk(self.logdir):
            root = Path(root)
            if not root.stem.startswith("generation_"):
                continue

            for dir in dirs:
                dir = Path(dir)
                if not dir.stem.startswith("rank_"):
                    continue

                path = root / dir
                if path in self._all_population or not (path / "config.yaml").exists():
                    continue

                self.add_animal(path)

        # Sort the population and update the current population
        pop = sorted(self._all_population.items(), key=lambda x: x[1][0])
        self._top_performers = [i for i, _ in pop[-self.num_top_performers :]]

    def _calculate_fitness(self, path: Path):
        """Calculates the fitness of the given animal.

        Currently, the fitness is calculated as the average reward over the last 1000
        (or the all rewards if there are less than 1000) timesteps of training. This is
        calculated by parsing the `monitor.csv` file.
        """
        if not (path / "monitor.csv").exists():
            return -np.inf

        _, rewards = ts2xy(load_results(path), "timesteps")
        if len(rewards) < 1000:
            fitness = -np.inf
        else:
            fitness = rewards[-min(len(rewards) - 1, 1000) :].mean()

        return fitness

    # ========

    def spawn(self) -> MjCambrianConfig:
        """This method will spawn a new animal from the current population.

        Spawning can be simply a mutation or crossover between multiple animals in the
        current population. Crossover always comes first, followed by mutation. This
        means crossover can include mutations.
        """
        config: MjCambrianConfig = None
        if self._replication_type & MjCambrianReplicationType.CROSSOVER:
            config = self._crossover()

        if self._replication_type & MjCambrianReplicationType.MUTATION:
            config = self._mutate(config)

        return config

    def _crossover(self):
        parent1_uid, parent2_uid = np.random.choice(self._top_performers, 2, False)
        parent1 = self._all_population[parent1_uid][1].copy()
        parent2 = self._all_population[parent2_uid][1].copy()

        verbose = parent1.training_config.verbose
        parent1.animal_config = MjCambrianAnimal.crossover(
            parent1.animal_config, parent2.animal_config, verbose=verbose
        )
        return self._set_parent(parent1, parent1)

    def _mutate(self, config: Optional[MjCambrianConfig] = None):
        if config is None:
            animal_uid = np.random.choice(self._top_performers)
            config = self._all_population[animal_uid][1].copy()
        
        verbose = config.training_config.verbose
        animal_config = config.animal_config.copy()
        config.animal_config = MjCambrianAnimal.mutate(animal_config, verbose=verbose)

        return self._set_parent(config, config)

    def _set_parent(self, config: MjCambrianConfig, parent: MjCambrianConfig):
        print(
            f"Selected parent from generation "
            f"{parent.evo_config.generation_config.generation} and rank "
            f"{parent.evo_config.generation_config.rank}"
        )
        parent_generation_config = parent.evo_config.generation_config.copy()
        config.evo_config.parent_generation_config = parent_generation_config
        return config

    # ========

    @property
    def size(self) -> int:
        return self.config.size

    @property
    def num_top_performers(self) -> int:
        return self.config.num_top_performers