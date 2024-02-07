from typing import List
import os
import subprocess
import threading
from pathlib import Path
import random
import logging

from cambrian.population import MjCambrianPopulation
from cambrian.animal import MjCambrianAnimal
from cambrian.utils.config import (
    MjCambrianConfig,
    MjCambrianGenerationConfig,
    MjCambrianSpawningConfig,
)
from cambrian.utils.logger import get_logger

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

        real_rank = int(rank / self.config.evo_config.population_config.size)
        generation_config = MjCambrianGenerationConfig(generation=generation, rank=rank)
        self.config.evo_config.generation_config = generation_config

        self.logdir = Path(
            Path(self.config.training_config.logdir)
            / self.config.training_config.exp_name
        )
        self.logdir.mkdir(parents=True, exist_ok=True)

        # Get the logger _after_ creating the logdir. Also, overwrite the filepath
        # only for this logger (subprocesses will use the default filepath).
        (self.logdir / "logs").mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(
            config,
            overwrite_filepath=self.logdir / "logs",
            overwrite_filename_suffix=f"_{real_rank}",
        )

        self.population = MjCambrianPopulation(self.config, self.logdir)

        # Save the initial config to the logdir
        (self.logdir / "initial_configs").mkdir(parents=True, exist_ok=True)
        self.config.save(self.logdir / "initial_configs" / f"config_{real_rank}.yaml")

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
        init_rank = self.config.evo_config.generation_config.rank
        population_size = self.config.evo_config.population_config.size
        for rank in range(init_rank, init_rank + population_size):
            thread = threading.Thread(target=_loop, args=(rank,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def train_animal(self, config: MjCambrianConfig):
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

        env = dict(os.environ, **self.config.evo_config.environment_variables)
        if not self.dry_run:
            process = subprocess.Popen(
                cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=1,
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

    parser.add_argument(
        "--dry-run", action="store_true", help="Don't actually run the training"
    )
    parser.add_argument(
        "-r", "--rank", type=int, help="Rank of this process", default=0
    )

    parser.add_argument("--no-egl", action="store_true", help="Disable EGL rendering")

    args = parser.parse_args()

    config: MjCambrianConfig = MjCambrianConfig.load(
        args.config, overrides=args.overrides
    )
    config.training_config.setdefault("exp_name", Path(args.config).stem)
    config.evo_config.setdefault("environment_variables", {})
    if not args.no_egl:
        config.evo_config.environment_variables["MUJOCO_GL"] = "egl"

    rank = config.evo_config.population_config.size * args.rank
    runner = MjCambrianEvoRunner(config, rank=rank, dry_run=args.dry_run)
    runner.evo()
