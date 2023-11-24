from typing import List
import os
import subprocess
from pathlib import Path

from stable_baselines3.common.utils import set_random_seed

from config import MjCambrianConfig, MjCambrianGenerationConfig
from population import MjCambrianPopulation


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

        # Initialize the population
        self.population.add_animal(self.config.copy(), -float("inf"))

    def evo(self):
        """This method run's evolution.

        The evolution loop is as follows:
            1. Select an animal
            2. Mutation the animal
            3. Train the animal
            4. Repeat

        Animal selection logic is provided by the MjCambrianAnimalPool subclass which
        is selected.
        """

        while self.generation < self.config.evo_config.num_generations:
            print(f"Starting generation {self.generation}...")

            init_rank = self.rank
            self._processes: List[subprocess.Popen] = []
            for _ in range(self.config.evo_config.population_config.size):
                self.update()

                config = self.population.spawn()
                self.train_animal(config)

                self.rank += 1

            for process in self._processes:
                process.wait()

            self.rank = init_rank
            self.generation += 1

    def update(self):
        # Set seed
        if self.verbose > 1:
            print(f"Setting seed for generation {self.generation}...")
        seed = self._calc_seed(0)
        set_random_seed(seed)
        if self.verbose > 1:
            print(f"Seed set to {seed}.")

        # Update logdir
        if self.verbose > 2:
            print(f"Updating logdir for generation {self.generation}...")
        generation_config = self.config.evo_config.generation_config
        self.generation_logdir = self.logdir / generation_config.to_path()
        self.generation_logdir.mkdir(parents=True, exist_ok=True)

        # Update the population
        self.population.update()

    def train_animal(self, config: MjCambrianConfig):
        if self.verbose > 1:
            print(f"Training animal for generation {self.generation}...")

        config.training_config.seed = self._calc_seed(0)
        config.training_config.logdir = str(self.generation_logdir)
        config.training_config.exp_name = ""
        config.evo_config.generation_config = self.config.evo_config.generation_config
        if (parent := config.evo_config.parent_generation_config) is not None:
            parent_logdir = self.logdir / parent.to_path()
            if (policy_path := parent_logdir / "policy.pt").exists():
                config.training_config.checkpoint_path = str(policy_path)
        if (max_n_envs := self.config.evo_config.max_n_envs) is not None:
            n_envs = max_n_envs // self.population.size
            if self.verbose > 1:
                print(f"Setting n_envs to {n_envs}")
            config.training_config.n_envs = n_envs

        config_yaml = self.generation_logdir / "config.yaml"
        config.write_to_yaml(config_yaml)

        trainer_py = Path(__file__).parent / "trainer.py"
        cmd = f"python {trainer_py} {config_yaml} --train"
        env = dict(os.environ, **self.config.evo_config.environment_variables)
        if self.verbose > 1:
            print(f"Running command: {cmd}")
        if not self.dry_run:
            stdin = subprocess.PIPE if self.verbose <= 1 else None
            stderr = subprocess.PIPE if self.verbose <= 1 else None
            popen = subprocess.Popen(cmd.split(" "), env=env, stdin=stdin, stderr=stderr)
            self._processes.append(popen)

    # ========

    def _calc_seed(self, i: int) -> int:
        """Calculates a unique seed for each environment.

        Equation is as follows:
            i * population_size * num_generations + seed + generation
        """
        return (
            (self.generation + 1) * (self.rank + 1)
            + self.config.training_config.seed
            + i
            * self.config.evo_config.population_config.size
            * self.config.evo_config.num_generations
        )

    # ========

    @property
    def rank(self) -> int:
        return self.config.evo_config.generation_config.rank

    @rank.setter
    def rank(self, rank: int):
        self.config.evo_config.generation_config.rank = rank

    @property
    def generation(self) -> MjCambrianGenerationConfig:
        return self.config.evo_config.generation_config.generation

    @generation.setter
    def generation(self, generation: int):
        self.config.evo_config.generation_config.generation = generation


if __name__ == "__main__":
    from utils import MjCambrianArgumentParser

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
