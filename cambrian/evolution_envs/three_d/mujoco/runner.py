from typing import List, Tuple, Any
from pathlib import Path
import cv2
import glfw

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecEnv,
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)
from stable_baselines3.common.callbacks import EvalCallback

from env import MjCambrianEnv
from config import MjCambrianConfig
from wrappers import make_single_env
from callbacks import (
    PlotEvaluationCallback,
    SaveVideoCallback,
)
from feature_extractors import MjCambrianCombinedExtractor
from cambrian.reinforce.evo.runner import _update_config_with_overrides


class MjCambrianRunner:
    def __init__(
        self,
        config: Path | str | MjCambrianConfig,
        overrides: List[Tuple[str, Any]],
    ):
        self.config = MjCambrianConfig.load(config)
        # _update_config_with_overrides(self.config, overrides)

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

        env = self._make_env(ppodir, n_envs=self.training_config.n_envs)
        eval_env = self._make_env(ppodir, n_envs=1)

        # Call reset and write the yaml
        env.reset()
        self.config.write_to_yaml(ppodir / "config.yaml")

        # Callbacks
        check_freq = self.training_config.check_freq
        eval_cb = EvalCallback(
            env,
            best_model_save_path=ppodir,
            log_path=ppodir,
            eval_freq=check_freq,
            deterministic=True,
            render=False,
            callback_on_new_best=SaveVideoCallback(
                eval_env, ppodir, self.training_config.max_episode_steps, verbose=1
            ),
            callback_after_eval=PlotEvaluationCallback(ppodir),
        )

        n_epochs = 10  # Default for PPO
        self.training_config.batch_size = (
            self.training_config.n_steps * self.training_config.n_envs
        ) // n_epochs
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=self.training_config.n_steps,
            batch_size=self.training_config.batch_size,
            verbose=self.verbose,
            policy_kwargs=dict(features_extractor_class=MjCambrianCombinedExtractor),
        )

        model.learn(
            total_timesteps=self.training_config.total_timesteps,
            callback=eval_cb,
            progress_bar=True,
        )

        model.save(ppodir / "best_model")

        env.close()

    def eval(self, random_actions: bool = False, fullscreen: bool = False):
        """Evaluate the model."""
        assert self.training_config.n_envs == 1, "Must have 1 env for evaluation."

        ppodir = self.logdir / "ppo"
        env = self._make_env(ppodir, n_envs=1)
        cambrian_env: MjCambrianEnv = env.envs[0]

        if not random_actions:
            print(f"Loading {ppodir / 'best_model.zip'}...")
            assert (ppodir / "best_model.zip").exists()
            model = PPO.load(ppodir / "best_model", env=env)

        obs = env.reset()

        window = cambrian_env.mujoco_renderer.viewer.window
        if fullscreen:
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                window,
                monitor,
                0,
                0,
                mode.size.width,
                mode.size.height,
                mode.refresh_rate,
            )
            glfw.focus_window(window)

        for animal_name in cambrian_env.animals:
            cv2.namedWindow(animal_name, cv2.WINDOW_NORMAL)

        done = True
        terminate = False
        for i in range(10000):
            if terminate:
                break
            if done:
                viewer = cambrian_env.mujoco_renderer.viewer
                viewer.cam.distance = 30
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -90

            if random_actions:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()

            for name, animal in cambrian_env.animals.items():
                if (composite_image := animal.create_composite_image()) is not None:
                    cv2.imshow(name, composite_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        terminate = True
                        break

        env.close()

    def _make_env(self, ppodir: Path, n_envs: int) -> VecEnv:
        """Create the environment."""
        assert n_envs > 0, "Must have at least one environment."
        if n_envs == 1:
            env = DummyVecEnv([make_single_env(0, 0, self.config)])
        else:
            env = SubprocVecEnv(
                [
                    make_single_env(i, i, self.config)
                    for i in range(n_envs)
                ]
            )
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
    parser.add_argument("--random", action="store_true", help="Use random actions.")
    parser.add_argument("--fullscreen", action="store_true", help="Use fullscreen.")

    args = parser.parse_args()

    runner = MjCambrianRunner(args.config, args.overrides)
    if args.train:
        runner.train()
    if args.eval:
        runner.config.training_config.n_envs = 1
        runner.config.env_config.render_mode = "human"
        runner.eval(args.random, args.fullscreen)
