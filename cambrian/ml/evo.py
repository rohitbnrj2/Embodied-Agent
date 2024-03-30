from typing import List, Optional, TYPE_CHECKING
import os
import subprocess
import threading
from pathlib import Path
import logging

from stable_baselines3.common.utils import set_random_seed

from cambrian.ml.population import (
    MjCambrianPopulation,
    MjCambrianPopulationConfig,
    MjCambrianSpawningConfig,
)
from cambrian.utils.logger import get_logger
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig

if TYPE_CHECKING:
    from cambrian.utils.config import MjCambrianConfig


@config_wrapper
class MjCambrianEvoConfig(MjCambrianBaseConfig):
    """Config for evolutions. Used for type hinting.

    Attributes:
        num_nodes (int): The number of nodes used for the evolution process. By default,
            this should be 1. And then if multiple evolutions are run in parallel, the
            number of nodes should be set to the number of evolutions.
        max_n_envs (int): The maximum number of environments to use for
            parallel training. Will set `n_envs` for each training process to
            `max_n_envs // population size`.
        num_generations (int): The number of generations to run for.

        population (MjCambrianPopulationConfig): The config for the population.
        spawning (MjCambrianSpawningConfig): The config for the spawning process.

        rank (int): The rank of the current evolution. Will be set by the evolution
            runner. A rank is essentially the id of the current evolution process.
        generation (int): The generation of the current evolution. Will be set by the
            evolution runner.

        generation (Optional[MjCambrianGenerationConfig]): The config for the
            current generation. Will be set by the evolution runner.
        parent_generation (Optional[MjCambrianGenerationConfig]): The config for
            the parent generation. Will be set by the evolution runner. If None, that
            means that the current generation is the first generation (i.e. no parent).

        top_performers (Optional[List[str]]): The top performers from the parent
            generation. This was used to select an animal to spawn an offspring from.
            Used for parsing after the fact.
    """

    num_nodes: int
    max_n_envs: int
    num_generations: int

    population: MjCambrianPopulationConfig
    spawning: MjCambrianSpawningConfig

    # rank: int
    # generation: int

    # parent_rank: int
    # parent_generation: int

    top_performers: Optional[List[str]] = None


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

    def __init__(self, config: "MjCambrianConfig", *, dry_run: bool = False):
        self.config = config
        self.evo_config = config.evo_config
        self.dry_run = dry_run

        # The node num is defined as the rank divided by the number of nodes
        # The rank should be initialized to the task id (i.e. slurm array job,
        # node/computer, etc).
        # This should be set prior to running the evo script
        node_num = self.evo_config.generation_config.rank
        self.evo_config.generation_config.rank *= self.evo_config.population_config.size

        # Create the logdir
        # Will also create the logdir if it doesn't exist
        (self.config.logdir / "logs").mkdir(parents=True, exist_ok=True)

        self.logger = get_logger()

        self.population = MjCambrianPopulation(self.config, self.logdir)

        # Save the initial config to the logdir
        (self.logdir / "initial_configs").mkdir(parents=True, exist_ok=True)
        self.config.save(self.logdir / "initial_configs" / f"config_{node_num}.yaml")

        # Update the seed to be unique between evos
        set_random_seed(self.config.training_config.seed + node_num)

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
            while generation < self.evo_config.num_generations:
                # Update the population. This will search the evo logdir for the latest
                # generation and update the population accordingly.
                self.population.update()

                # Spawn a new animal
                config = self.population.spawn_animal(generation, rank)

                # Save the config
                config.save(Path(config.training_config.logdir) / "config.yaml")

                # Finally, train the animal
                self.train_animal(config)

                generation += 1

        threads: List[threading.Thread] = []
        init_rank = self.evo_config.generation_config.rank
        population_size = self.evo_config.population_config.size
        for rank in range(init_rank, init_rank + population_size):
            thread = threading.Thread(target=_loop, args=(rank,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def train_animal(self, config: "MjCambrianConfig"):
        """Runs the training process for the given animal.

        Each trainer is launched using a separate process and waits until it completes.
        This method is blocking, so if multiple animals should be trained separately
        on the same computer, this method should be called from a thread.
        """

        # Create the cmd
        trainer_py = Path(__file__).parent / "trainer.py"
        python_cmd = f"python {trainer_py}"
        config_yaml = Path(config.training_config.logdir) / "config.yaml"
        cmd = f"{python_cmd} {config_yaml} --train"
        self.logger.debug(f"Running command: {cmd}")

        env = dict(os.environ, **self.evo_config.environment_variables)
        if not self.dry_run:
            process = subprocess.Popen(
                cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )

            # Log the output of the subprocess
            threading.Thread(
                target=self._log_subprocess_output,
                args=(
                    process.stdout,
                    logging.INFO,
                    config.evo_config.generation_config.rank,
                ),
            ).start()
            threading.Thread(
                target=self._log_subprocess_output,
                args=(
                    process.stderr,
                    logging.ERROR,
                    config.evo_config.generation_config.rank,
                ),
            ).start()

            process.wait()

    # ========

    def _log_subprocess_output(self, pipe, log_level: int, rank: int):
        """Logs the output of a subprocess to the logger with the given log level.

        The logged output will also be prepended with the rank of the process. The lines
        will only be logged if they are not empty.
        """
        with pipe:
            for line in iter(pipe.readline, b""):
                line = line.strip()
                if line:
                    self.logger.log(log_level, f"[{rank}]: {line}")


if __name__ == "__main__":
    from cambrian.utils.utils import MjCambrianArgumentParser

    parser = MjCambrianArgumentParser()

    parser.add_argument("--dry-run", action="store_true", help="Don't actually run")
    parser.add_argument("--no-egl", action="store_true", help="Disable EGL rendering")

    args = parser.parse_args()

    config = MjCambrianConfig.load(args.config, overrides=args.overrides)
    if not args.no_egl:
        # Set's EGL rendering for all trainer processes
        config.evo_config.environment_variables.setdefault("MUJOCO_EGL", "egl")

    runner = MjCambrianEvoRunner(config, dry_run=args.dry_run)
    runner.evo()
