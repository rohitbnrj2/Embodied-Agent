from typing import List, Tuple, Any
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecEnv,
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)
from stable_baselines3.common.callbacks import EvalCallback

from config import MjCambrianConfig, write_yaml
from wrappers import make_single_env
from callbacks import PlotEvaluationCallback
from feature_extractors import MjCambrianCombinedExtractor
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
        ppodir = self.logdir / "ppo"
        env = self._make_env(ppodir)

        if not random_actions:
            print(f"Loading {ppodir / 'best_model.zip'}...")
            assert (ppodir / "best_model.zip").exists()
            model = PPO.load(ppodir / "best_model", env=env)

        obs = env.reset()

        import cv2, glfw

        window = env.envs[0].mujoco_renderer.viewer.window
        # old_callback = glfw.get_key_callback(window)
        # def _key_callback(w, k, s, a, m):
        #     if k == glfw.KEY_Q and a == glfw.PRESS:
        #         glfw.set_window_should_close(w, True)
        #     else:
        #         old_callback(w, k, s, a, m)
        # glfw.set_key_callback(window, _key_callback)
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

        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)

        for i in range(10000):
            if random_actions:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()

            images = []
            for animal_name, animal in env.envs[0].animals.items():
                num_eyes_lat, num_eyes_lon = (
                    animal.config.num_eyes_lat,
                    animal.config.num_eyes_lon,
                )
                for i in range(num_eyes_lat):
                    images.append([])
                    for j in range(num_eyes_lon):
                        eye_name = f"{animal_name}_eye_{i * num_eyes_lat + j}"
                        eye_obs = obs[eye_name]
                        images[i].append(eye_obs.squeeze(0).transpose(1, 0, 2))
            # # Concat the image
            # image = cv2.vconcat(
            #     [cv2.hconcat(image_row) for image_row in reversed(images)]
            # )

            # cv2.imshow("image", image)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

        env.close()

    def _make_env(self, ppodir: Path) -> VecEnv:
        """Create the environment."""
        assert self.training_config.n_envs > 0, "Must have at least one environment."
        if self.training_config.n_envs == 1:
            env = DummyVecEnv([make_single_env(0, 0, self.config)])
        else:
            env = SubprocVecEnv(
                [
                    make_single_env(i, i, self.config)
                    for i in range(self.training_config.n_envs)
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
        runner.eval(args.random, args.fullscreen)
