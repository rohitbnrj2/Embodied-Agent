from typing import Dict, Callable, TYPE_CHECKING
from pathlib import Path

from stable_baselines3.common.vec_env import (
    VecEnv,
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

from cambrian.envs.env import MjCambrianEnv, MjCambrianEnvConfig
from cambrian.ml.model import MjCambrianModel
from cambrian.utils import evaluate_policy, calculate_fitness
from cambrian.utils.config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.wrappers import make_wrapped_env
from cambrian.utils.logger import get_logger

if TYPE_CHECKING:
    from cambrian.utils.config.config import MjCambrianConfig


@config_wrapper
class MjCambrianTrainerConfig(MjCambrianBaseConfig):
    """Settings for the training process. Used for type hinting.

    Attributes:
        total_timesteps (int): The total number of timesteps to train for.
        max_episode_steps (int): The maximum number of steps per episode.
        n_envs (int): The number of parallel environments to use for training.

        model (MjCambrianModelType): The model to use for training.
        callbacks (Dict[str, BaseCallback]): The callbacks to use for training.
        wrappers (Dict[str, Callable[[VecEnv], VecEnv]] | None): The wrappers to use for
            training. If None, will ignore.
    """

    total_timesteps: int
    max_episode_steps: int
    n_envs: int

    model: MjCambrianModel
    callbacks: Dict[str, BaseCallback | Callable[[VecEnv], BaseCallback]]
    wrappers: Dict[str, Callable[[VecEnv], VecEnv] | None]


class MjCambrianTrainer:
    """This is the trainer class for running training and evaluation.

    Args:
        config (MjCambrianConfig): The config to use for training and evaluation.
    """

    def __init__(self, config: "MjCambrianConfig"):
        self.config = config
        self.trainer_config = config.trainer

        from hydra.core.hydra_config import HydraConfig
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(HydraConfig.get()))
        exit()

        self.config.logdir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger()
        self.logger.info(f"Logging to {self.config.logdir / 'logs'}...")

        self.logger.debug(f"Setting seed to {self.config.seed}...")
        set_random_seed(self.config.seed)

    def train(self) -> float:
        """Train the animal."""
        # Set to warn so we have something output to the error log
        self.logger.warning(f"Training the animal in {self.config.logdir}...")

        self.config.save(self.config.logdir / "config.yaml")

        # Setup the environment, model, and callbacks
        env = self._make_env(self.config.env, self.trainer_config.n_envs)
        eval_env = self._make_env(self.config.eval_env, 1, monitor="eval_monitor.csv")
        callback = self._make_callback(eval_env)
        model = self._make_model(env)

        # Save the eval environments xml
        # All xml's _should_ be the same
        xml_path = self.config.logdir / "env.xml"
        cambrian_env: MjCambrianEnv = eval_env.envs[0].unwrapped
        cambrian_env.xml.write(xml_path)

        # Start training
        total_timesteps = self.trainer_config.total_timesteps
        model.learn(total_timesteps=total_timesteps, callback=callback)
        self.logger.info("Finished training the animal...")

        # Save the policy
        self.logger.info(f"Saving model to {self.config.logdir}...")
        model.save_policy(self.config.logdir)
        self.logger.debug(f"Saved model to {self.config.logdir}...")

        # The finished file indicates to the evo script that the animal is done
        Path(self.config.logdir / "finished").touch()

        # Calculate fitness
        fitness = calculate_fitness(self.config.logdir / "evaluations.npz")
        self.logger.info(f"Final Fitness: {fitness}")

        return fitness

    def eval(self):
        self.config.save(self.config.logdir / "eval_config.yaml")

        eval_env = self._make_env(self.config.env, 1, monitor="eval_monitor.csv")
        model = self._make_model(eval_env)
        model = model.load(self.config.logdir / "best_model")

        n_runs = self.config.eval_env.n_eval_episodes
        filename = self.config.logdir / "eval"
        record_kwargs = dict(
            path=filename, save_mode=self.config.eval_env.renderer.save_mode
        )
        evaluate_policy(eval_env, model, n_runs, record_kwargs=record_kwargs)

    # ========

    def _calc_seed(self, i: int) -> int:
        return self.config.seed + i

    def _make_env(
        self,
        config: MjCambrianEnvConfig,
        n_envs: int,
        *,
        monitor: str | None = "monitor.csv",
    ) -> VecEnv:
        assert n_envs > 0, f"n_envs must be > 0, got {n_envs}."

        # Create the environments
        envs = []
        for i in range(n_envs):
            wrappers = [w for w in self.trainer_config.wrappers.values() if w]
            wrapped_env = make_wrapped_env(
                config=config.copy(),
                name=self.config.expname,
                wrappers=wrappers,
                seed=self._calc_seed(i),
            )
            envs.append(wrapped_env)

        # Wrap the environments
        vec_env = DummyVecEnv(envs) if n_envs == 1 else SubprocVecEnv(envs)
        if monitor is not None:
            vec_env = VecMonitor(vec_env, str(self.config.logdir / monitor))

        # Do an initial reset
        vec_env.reset()
        return vec_env

    def _make_callback(self, env: VecEnv) -> CallbackList:
        """Makes the callbacks."""
        from functools import partial

        callbacks = []
        for callback in self.trainer_config.callbacks.values():
            # TODO: is this a good assumption? is there a better way to do this?
            if isinstance(callback, partial):
                callback = callback(env)
            callbacks.append(callback)

        return CallbackList(callbacks)

    def _make_model(self, env: VecEnv) -> MjCambrianModel:
        """This method creates the model."""
        return self.trainer_config.model(env=env)


if __name__ == "__main__":
    import argparse

    from cambrian.utils.config import run_hydra
    from cambrian.utils.config.config import MjCambrianConfig

    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train the model")
    action.add_argument("--eval", action="store_true", help="Evaluate the model")

    def main(config: MjCambrianConfig, *, train: bool, eval: bool) -> float | None:
        """This method will return a float if training. The float represents the
        "fitness" of the agent that was trained. This can be used by hydra to
        determine the best hyperparameters during sweeps."""
        runner = MjCambrianTrainer(config)

        if train:
            return runner.train()
        elif eval:
            runner.eval()

    run_hydra(main, parser=parser)
