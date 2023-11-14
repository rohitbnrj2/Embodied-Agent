from typing import List, Tuple, Any, Dict
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecEnv,
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.utils import set_random_seed

from env import MjCambrianEnv
from config import MjCambrianConfig
from wrappers import make_single_env
from callbacks import (
    PlotEvaluationCallback,
    SaveVideoCallback,
    CallbackListWithSharedParent,
)
from feature_extractors import MjCambrianCombinedExtractor
from renderer import MjCambrianRenderer
from animal_pool import MjCambrianAnimalPool


def _convert_overrides_to_dict(overrides: List[Tuple[str, Any]]):
    import yaml

    overrides_dict: Dict[str, Any] = {}
    for k, v in overrides:
        d = overrides_dict

        keys = k.split(".")
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = yaml.safe_load(v)

    return overrides_dict


class MjCambrianRunner:
    """This is the runner class for instantiating environments and running them.

    There are three main methods:
        - train: train the model from scratch or from an existing checkpoint
        - eval: evaluate the model using an existing checkpoint or a randomized model
    """

    def __init__(
        self,
        config_path: Path | str,
        *,
        overrides: Dict[str, Any] = {},
    ):
        config_path = Path(config_path)
        self.config: MjCambrianConfig = MjCambrianConfig.load(
            config_path, overrides=overrides
        )

        set_random_seed(self.config.training_config.seed)

        self.env_config = self.config.env_config
        self.training_config = self.config.training_config
        self.evo_config = self.config.evo_config
        self.verbose = self.training_config.verbose

        self.training_config.setdefault("exp_name", config_path.stem)
        self.logdir: Path = Path(self.training_config.logdir) / self.training_config.exp_name
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.ppodir = self.logdir / "ppo"
        self.ppodir.mkdir(parents=True, exist_ok=True)

    def evo(self, args):
        """Run the evolution.

        This does 4 things:
            1. Mutate the current animal
            2. Train the animal
            3. Select one of the best performing animals to train
            4. Repeat
        """
        animal_pool = MjCambrianAnimalPool(self.config, args.rank, verbose=self.verbose)
        animal_config = animal_pool.get_new_animal_config()

        generation = 0
        while generation < self.evo_config.num_generations:
            print(f"Starting generation {generation}...")

            # Mutate the current animal
            config = self.config.copy(animal_config=animal_config)

            generation += 1

    def train(self, args):
        """Train the model."""

        # Create a regular training environment and an evaluation environment
        # The eval env only has one environment, so that we can save videos
        env = self._make_env(self.training_config.n_envs)
        eval_env = self._make_env(1)

        # Call reset and write the yaml
        env.reset()
        self.config.write_to_yaml(self.ppodir / "config.yaml")

        # Create the model and train it
        model = self._create_model(env)
        model.learn(
            total_timesteps=self.training_config.total_timesteps,
            callback=self._create_callbacks(env, eval_env),
            progress_bar=True,
        )

        # Save the final model
        model.save(self.ppodir / "best_model")

        # Cleanup
        env.close()
        eval_env.close()

    def eval(self, args):
        """Evaluate the model."""
        self.env_config.renderer_config.fullscreen = args.fullscreen
        self.env_config.renderer_config.render_modes = ["rgb_array"] + (
            [] if args.no_human else ["human"]
        )
        if self.training_config.n_envs != 1:
            print("WARNING: n_envs is not set to 1!")
            self.training_config.n_envs = 1

        env = self._make_env(self.training_config.n_envs)
        cambrian_env: MjCambrianEnv = env.envs[0].unwrapped

        model = self._create_model(env, force=args.random_actions)

        obs = env.reset()

        renderer: MjCambrianRenderer = cambrian_env.renderer
        renderer.record = args.record

        run = 0
        done = True
        cumulative_reward = 0
        print("Starting evaluation...")
        while True:
            if not renderer.is_running:
                print("Renderer closed. Exiting...")
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cumulative_reward += reward[0]

            if done:
                print(f"Episode done. Cumulative reward: {cumulative_reward:.2f}")
                run += 1
                if run >= args.total_runs:
                    break
                print(f"Starting run {run}...")

            cambrian_env.rollout["Cumulative Reward"] = f"{cumulative_reward:.2f}"
            env.render()

        if args.record:
            renderer.save(self.ppodir / "eval")
            renderer.record = False

        env.close()

    def _create_model(self, env: VecEnv, *, force: bool = False) -> PPO:
        """Try to create the model. If loading is requested, do that. Otherwise, create
        a new model.

        Keyword Args:
            force (bool): If True, create a new model even if a checkpoint is found.
        """
        if not force and (path := self._get_ppo_checkpoint_path()) is not None:
            print(f"Loading {path}...")
            model = PPO.load(path, env=env)
        else:
            print("Creating new model...")
            policy_kwargs = dict(features_extractor_class=MjCambrianCombinedExtractor)
            model = PPO(
                "MultiInputPolicy",
                env,
                n_steps=self.training_config.n_steps,
                batch_size=self._calc_batch_size(),
                learning_rate=self.training_config.learning_rate,
                verbose=self.verbose,
                policy_kwargs=policy_kwargs,
            )
        return model

    def _create_callbacks(self, env: VecEnv, eval_env: gym.Env) -> BaseCallback:
        save_video_callback = SaveVideoCallback(
            eval_env,
            self.ppodir,
            self.training_config.max_episode_steps,
            verbose=self.verbose,
        )
        stop_training_on_no_model_improvement = StopTrainingOnNoModelImprovement(
            self.training_config.max_no_improvement_evals,
            self.training_config.min_no_improvement_evals,
            verbose=self.verbose,
        )
        callbacks_on_new_best = CallbackListWithSharedParent(
            [
                save_video_callback,
                stop_training_on_no_model_improvement,
            ]
        )

        eval_cb = EvalCallback(
            env,
            best_model_save_path=self.ppodir,
            log_path=self.ppodir,
            eval_freq=self.training_config.eval_freq,
            deterministic=True,
            render=False,
            callback_on_new_best=callbacks_on_new_best,
            callback_after_eval=PlotEvaluationCallback(self.ppodir),
        )

        return eval_cb

    def _calc_batch_size(self) -> int:
        """Calculates the batch size as n_steps * n_envs / n_epochs.

        n_epochs is set to 8 as a default since stable_baselines3 uses 10 and 8 is
        the closest factor of 2. Stable baselines recommends to use a batch size that is
        a factor of `n_steps * n_envs` which _should_ be a factor of 2 (and it's
        powers), but not of 10.

        NOTE: if `training_config.batch_size` is set, that will be returned.
        """
        if self.training_config.batch_size is None:
            self.training_config.batch_size = (
                self.training_config.n_steps * self.training_config.n_envs
            ) // self.training_config.n_epochs
        return self.training_config.batch_size

    def _make_env(self, n_envs: int) -> VecEnv:
        """Create the environment.

        NOTE: `use_renderer` for the first environment is set to True, and False for
        all others.
        """
        assert n_envs > 0, "Must have at least one environment."
        if n_envs == 1:
            env = DummyVecEnv([make_single_env(0, 0, self.config)])
        else:
            env = SubprocVecEnv(
                [
                    make_single_env(i, i, self.config, use_renderer=i == 0)
                    for i in range(n_envs)
                ]
            )
        env = VecMonitor(env, str(self.ppodir))
        return env

    def _get_ppo_checkpoint_path(self) -> Path | None:
        if self.training_config.ppo_checkpoint_path is None:
            return None

        ppo_checkpoint_path = Path(self.training_config.ppo_checkpoint_path)

        possible_paths = [
            ppo_checkpoint_path,
            self.logdir / ppo_checkpoint_path,
            self.logdir / "ppo" / ppo_checkpoint_path,
            Path(self.training_config.logdir) / ppo_checkpoint_path,
        ]
        for possible_path in possible_paths:
            if possible_path.exists():
                return possible_path

        raise FileNotFoundError(f"Could not find {ppo_checkpoint_path}.")


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

    actions = parser.add_subparsers(help="Actions to take.", required=True)

    train = actions.add_parser("train", help="Train the model.")
    train.set_defaults(cmd=MjCambrianRunner.train)

    eval = actions.add_parser("eval", help="Evaluate the model.")
    eval.add_argument(
        "--random-actions", action="store_true", help="Use random actions."
    )
    eval.add_argument("--fullscreen", action="store_true", help="Use fullscreen.")
    eval.add_argument("--record", action="store_true", help="Record a gif.")
    eval.add_argument(
        "--total-runs",
        "--num-runs",
        type=int,
        help="Number of runs. Default is 1.",
        default=1,
    )
    eval.add_argument(
        "--no-human", action="store_true", help="Don't render the human view."
    )
    eval.set_defaults(cmd=MjCambrianRunner.eval)

    args = parser.parse_args()

    overrides = _convert_overrides_to_dict(args.overrides)
    runner = MjCambrianRunner(args.config, overrides=overrides)

    args.cmd(runner, args)
