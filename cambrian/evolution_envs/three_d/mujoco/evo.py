import os
import subprocess
from pathlib import Path

from stable_baselines3.common.utils import set_random_seed

from animal import MjCambrianAnimal
from config import MjCambrianConfig, MjCambrianGenerationConfig


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
        self, config: MjCambrianConfig, rank: int = 0, initial_generation: int = 0
    ):
        self.config = config
        self.rank = rank

        self.verbose = self.config.training_config.verbose

        self.logdir = Path(
            Path(self.config.training_config.logdir)
            / self.config.training_config.exp_name
        )
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.generation = MjCambrianGenerationConfig(
            rank=rank, generation=initial_generation
        )

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

            self.update()

            config = self.select_animal()
            config = self.mutate_animal(config)
            self.train_animal(config)

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
        self.generation_logdir = self.logdir / self.generation.to_path()
        self.generation_logdir.mkdir(parents=True, exist_ok=True)

    def select_animal(self) -> MjCambrianConfig:
        if self.verbose > 1:
            print(f"Selecting animal for generation {self.generation}...")
        return self.config.copy()

    def mutate_animal(self, config: MjCambrianConfig) -> MjCambrianConfig:
        if self.verbose > 1:
            print(f"Mutating animal for generation {self.generation}...")
        animal_config = config.animal_config.copy()
        animal_config = MjCambrianAnimal.mutate(animal_config, verbose=self.verbose)

        evo_config = config.evo_config.copy()
        if evo_config.generation is not None:
            evo_config.parent_generation = evo_config.generation.copy()
        evo_config.generation = self.generation

        return config.copy(animal_config=animal_config, evo_config=evo_config)

    def train_animal(self, config: MjCambrianConfig):
        if self.verbose > 1:
            print(f"Training animal for generation {self.generation}...")

        self.config = config
        self.config.training_config.logdir = str(self.generation_logdir)
        self.config.training_config.exp_name = ""
        if (parent_generation := self.config.evo_config.parent_generation) is not None:
            parent_logdir = self.logdir / parent_generation.to_path()
            if (policy_path := parent_logdir / "policy.pt").exists():
                self.config.training_config.checkpoint_path = str(policy_path)

        self.config.write_to_yaml(self.generation_logdir / "config.yaml")

        runner_py = Path(__file__).parent / "runner.py"
        cmd = f"python {runner_py} {self.generation_logdir / 'config.yaml'} -r {self.rank} --seed {self._calc_seed(0)} --train"
        subprocess.run(cmd.split(" "), env=dict(os.environ, **self.config.evo_config.training_env_vars))

    # ========

    def _calc_seed(self, i: int) -> int:
        """Calculates a unique seed for each environment.

        Equation is as follows:
            i * population_size * num_generations + seed + generation
        """
        return (
            self.generation * self.rank
            + self.config.training_config.seed
            + i
            * self.config.evo_config.population_size
            * self.config.evo_config.num_generations
        )

    # ========

    @property
    def generation(self) -> MjCambrianGenerationConfig:
        return self.config.evo_config.generation

    @generation.setter
    def generation(self, generation: MjCambrianGenerationConfig):
        self.config.evo_config.generation = generation


if __name__ == "__main__":
    from utils import MjCambrianArgumentParser

    parser = MjCambrianArgumentParser()

    parser.add_argument("--no-egl", action="store_true", help="Disable EGL rendering")

    args = parser.parse_args()

    config = MjCambrianConfig.load(args.config, overrides=args.overrides)
    config.training_config.setdefault("exp_name", Path(args.config).stem)
    config.evo_config.setdefault("training_env_vars", {})
    if not args.no_egl:
        config.evo_config.training_env_vars["MUJOCO_GL"] = "egl"

    runner = MjCambrianEvoRunner(config)
    runner.evo()
