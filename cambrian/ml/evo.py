from typing import List
import os
import subprocess
import threading
from pathlib import Path
import random

from cambrian.population import MjCambrianPopulation
from cambrian.animal import MjCambrianAnimal
from cambrian.utils.config import (
    MjCambrianConfig,
    MjCambrianGenerationConfig,
    MjCambrianSpawningConfig,
)

ReplicationType = MjCambrianSpawningConfig.ReplicationType


class MjCambrianEvoRunner:
    """This is the evolutionary runner.

    This is the primary runner that dispatches training sessions. Each training batch
    is implemented in `runner.py` and is run as a separate process. The mutations and
    configuration writing occurs in this class.

    We run training as separate processes to avoid memory leaks. In this way,
    each training session essentially starts from scratch. Additionally, we then have
    two levels of parallelism: the number of training sessions and the number of
    environments per training session. This allows us to scale up the training to
    allow multiple parallel environments to be run at the same time on the same node.
    """

    def __init__(
        self,
        config: MjCambrianConfig,
        rank: int = 0,
        generation: int = 0,
        *,
        dry_run: bool = False,
    ):
        self.config = config
        self.dry_run = dry_run

        generation_config = MjCambrianGenerationConfig(generation=generation, rank=rank)
        self.config.evo_config.generation_config = generation_config

        self.verbose = self.config.training_config.verbose

        self.logdir = Path(
            Path(self.config.training_config.logdir)
            / self.config.training_config.exp_name
        )
        self.logdir.mkdir(parents=True, exist_ok=True)

        population_config = self.config.evo_config.population_config
        self.population = MjCambrianPopulation(population_config, self.logdir)

        trainer_py = Path(__file__).parent / "trainer.py"
        self.python_cmd = f"python {trainer_py}"

    def evo(self):
        """This method run's evolution.

        The evolution loop does the following:
            1. Updates the population
            2. Spawns a new animal
            3. Trains the animal
            4. Repeat

        To reduce the amount of time any one process is waiting, we'll spawn a new
        training immediately after it finishes training. Training stops when the total
        number of generations across all processes reaches
        num_generations * population_size.

        Animal selection logic is provided by the MjCambrianPopulation subclass and
        mutation is performed by MjCambrianAnimal.
        """

        def _loop(rank: int):
            generation = 0
            while generation < self.config.evo_config.num_generations:
                self.population.update()

                config = self.spawn_animal(generation, rank)
                process = self.train_animal(config)
                if not self.dry_run:
                    process.wait()

                generation += 1

        threads: List[threading.Thread] = []
        init_rank = self.config.evo_config.generation_config.rank
        population_size = self.config.evo_config.population_config.size
        for rank in range(init_rank, init_rank + population_size):
            thread = threading.Thread(target=_loop, args=(rank,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def spawn_animal(self, generation: int, rank: int) -> MjCambrianConfig:
        """Spawns a new animal.
        
        NOTE: The default eye config is selected randomly from the animal's eye configs.

        # TODO: move all of the mutation logic to MjCambrianPopulation or some other
        # class.
        """
        if self.verbose > 1:
            print(f"Spawning animal with generation {generation} and rank {rank}.")


        # Get the spawning config
        config: MjCambrianConfig  # the parent config, going to mutate
        num_mutations: int  # number of mutations to perform
        replication_type: ReplicationType  # type of replication
        if generation == 0:
            # If it's the first generation, we'll mutate the original config n number of
            # times to facilitate a diverse initial population.
            config = self.config.copy()
            num_mutations = random.randint(
                1, config.evo_config.spawning_config.init_num_mutations
            )
            replication_type = ReplicationType.MUTATION
        else:
            config = self.population.select_animal()
            num_mutations = random.randint(
                1, config.evo_config.spawning_config.num_mutations
            )
            replication_type = ReplicationType[
                config.evo_config.spawning_config.replication_type
            ]

        # Mutate the config
        if self.verbose > 1:
            print(
                f"Mutating child of parent with rank {config.evo_config.generation_config.rank} {num_mutations} times."
            )
        animal_configs = config.env_config.animal_configs
        spawning_config = config.evo_config.spawning_config
        for _ in range(num_mutations):
            for animal_config in animal_configs.values():
                if ReplicationType.MUTATION in replication_type:
                    animal_configs[animal_config.name] = MjCambrianAnimal.mutate(
                        animal_config,
                        random.choice(list(animal_config.eye_configs.values())),
                        mutations=spawning_config.mutation_options,
                        verbose=self.config.training_config.verbose,
                    )
                elif ReplicationType.CROSSOVER in replication_type:
                    raise NotImplementedError
                else:
                    raise ValueError(f"Invalid replication type {replication_type}")

        evo_config = config.evo_config
        evo_config.parent_generation_config = evo_config.generation_config.copy()
        evo_config.generation_config.rank = rank
        evo_config.generation_config.generation = generation
        generation_logdir = self.logdir / evo_config.generation_config.to_path()
        generation_logdir.mkdir(parents=True, exist_ok=True)

        config.training_config.seed = self._calc_seed(generation, rank)
        config.training_config.logdir = str(generation_logdir)
        config.training_config.exp_name = ""
        if (parent := config.evo_config.parent_generation_config) is not None:
            parent_logdir = self.logdir / parent.to_path()
            if (policy_path := parent_logdir / "policy.pt").exists():
                config.training_config.policy_path = str(policy_path)

        # Set n_envs to be the max_n_envs divided by the population size
        n_envs = config.evo_config.max_n_envs // self.population.size
        if self.verbose > 1:
            print(f"Setting n_envs to {n_envs}")
        config.training_config.n_envs = n_envs

        # Save the config
        config.save(generation_logdir / "config.yaml")

        return config

    def train_animal(self, config: MjCambrianConfig) -> subprocess.Popen | None:
        config_yaml = Path(config.training_config.logdir) / "config.yaml"
        cmd = f"{self.python_cmd} {config_yaml} --train"
        env = dict(os.environ, **self.config.evo_config.environment_variables)
        if self.verbose > 1:
            print(f"Running command: {cmd}")
        if not self.dry_run:
            stdin = subprocess.PIPE if self.verbose <= 1 else None
            stdout = subprocess.PIPE if self.verbose <= 1 else None
            stderr = subprocess.PIPE if self.verbose <= 1 else None
            return subprocess.Popen(
                cmd.split(" "), env=env, stdin=stdin, stdout=stdout, stderr=stderr
            )

    # ========

    def _calc_seed(self, generation: int, rank: int) -> int:
        """Calculates a unique seed for each environment.

        Equation is as follows:
            i * population_size * num_generations + seed + generation
        """
        # fmt: off
        return (
            (generation + 1) * (rank + 1) 
            + self.config.training_config.seed 
            * self.config.evo_config.population_config.size 
            * self.config.evo_config.num_generations 
        )
        # fmt: on


if __name__ == "__main__":
    from cambrian.utils.utils import MjCambrianArgumentParser

    parser = MjCambrianArgumentParser()

    parser.add_argument(
        "--dry-run", action="store_true", help="Don't actually run the training"
    )
    parser.add_argument(
        "-r", "--rank", type=int, help="Rank of this process", default=0
    )

    parser.add_argument("--no-egl", action="store_true", help="Disable EGL rendering")

    args = parser.parse_args()

    config = MjCambrianConfig.load(args.config, overrides=args.overrides)
    config.training_config.setdefault("exp_name", Path(args.config).stem)
    config.evo_config.setdefault("environment_variables", {})
    if not args.no_egl:
        config.evo_config.environment_variables["MUJOCO_GL"] = "egl"

    rank = config.evo_config.population_config.size * args.rank
    runner = MjCambrianEvoRunner(config, rank=rank, dry_run=args.dry_run)
    runner.evo()
