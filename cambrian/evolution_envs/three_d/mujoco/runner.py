from typing import List, Tuple, Any
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecEnv,
    DummyVecEnv,
    VecMonitor,
)
from stable_baselines3.common.callbacks import EvalCallback

from config import MjCambrianConfig, write_yaml
from wrappers import make_single_env
from cambrian.reinforce.evo.runner import _update_config_with_overrides


class MjCambrianRunner:
    def __init__(
        self,
        config: Path | str | MjCambrianConfig,
        overrides: List[Tuple[str, Any]],
    ):
        self.config = MjCambrianConfig.load(config)
        _update_config_with_overrides(self.config, overrides)

        self.training_config = self.config.training_config
        self.env_config = self.config.env_config

        self.logdir = Path(self.training_config.logdir) / self.training_config.exp_name
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.verbose = self.training_config.verbose

    def train(self):
        """Begin the training."""

        # Agent metadata
        ppodir = self.logdir / "ppo"
        ppodir.mkdir(parents=True, exist_ok=True)
        write_yaml(self.config, ppodir / "config.yaml")

        env = self._make_env(ppodir)

        # Callbacks
        check_freq = self.training_config.check_freq
        eval_cb = EvalCallback(
            env,
            best_model_save_path=ppodir,
            log_path=ppodir,
            eval_freq=check_freq,
            deterministic=True,
            render=False,
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=self.training_config.n_steps,
            batch_size=self.training_config.batch_size,
            verbose=self.verbose,
        )

        model.learn(
            total_timesteps=self.training_config.total_timesteps,
            callback=eval_cb,
            progress_bar=True,
        )

        model.save(ppodir / "best_model")

        env.close()

    def eval(self):
        """Evaluate the model."""
        ppodir = self.logdir / "ppo"
        env = self._make_env(ppodir)

        model = (
            PPO.load(ppodir / "best_model")
            if (ppodir / "best_model.zip").exists()
            else PPO(
                "MultiInputPolicy",
                env,
                n_steps=self.training_config.n_steps,
                batch_size=self.training_config.batch_size,
                verbose=self.verbose,
            )
        )

        obs = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            env.render()

        env.close()

    def _make_env(self, ppodir: Path) -> VecEnv:
        """Create the environment."""
        env = DummyVecEnv([make_single_env(0, 0, self.config)])
        env = VecMonitor(env, ppodir.as_posix())
        return env


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Runner for cambrian/mujoco envs")

    parser.add_argument("config", type=str, help="The config file to use.")
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action="append",
        type=lambda v: v.split("="),
        help="Override config values. Do <dot separated yaml config>=<value>",
        default=[],
    )

    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model.")

    args = parser.parse_args()

    runner = MjCambrianRunner(args.config, args.overrides)
    if args.train:
        runner.train()
    if args.eval:
        runner.eval()
