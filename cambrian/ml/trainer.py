from pathlib import Path

import torch
from stable_baselines3.common.vec_env import (
    VecEnv,
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)
from stable_baselines3.common.callbacks import (
    BaseCallback,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
    CallbackList,
)
from stable_baselines3.common.utils import set_random_seed

from cambrian.envs.env import MjCambrianEnv
from cambrian.ml.features_extractors import MjCambrianCombinedExtractor
from cambrian.ml.callbacks import (
    MjCambrianEvalCallback,
    MjCambrianPlotMonitorCallback,
    MjCambrianSavePolicyCallback,
    MjCambrianPlotEvaluationsCallback,
    MjCambrianGPUUsageCallback,
    MjCambrianProgressBarCallback,
    CallbackListWithSharedParent,
)
from cambrian.ml.model import MjCambrianModel
from cambrian.utils import (
    evaluate_policy,
    get_gpu_memory_usage,
    get_observation_space_size,
    setattrs_temporary,
)
from cambrian.utils.config import MjCambrianConfig
from cambrian.utils.wrappers import make_single_env
from cambrian.utils.logger import get_logger


class MjCambrianTrainer:
    """This is the trainer class for running training and evaluation.

    Args:
        config (MjCambrianConfig): The config to use for training and evaluation.
    """

    def __init__(self, config: MjCambrianConfig):
        self.config = config
        self.training_config = config.training

        self.logdir = Path(
            Path(self.training_config.logdir)
            / self.training_config.exp_name
        )
        self.logdir.mkdir(parents=True, exist_ok=True)

        (self.logdir / "logs").mkdir(exist_ok=True)
        self.logger = get_logger(
            self.config,
            overwrite_filepath=self.logdir / "logs",
            allow_missing_filepath=False,
        )
        self.logger.info(f"Logging to {self.logdir / 'logs'}...")

        self.logger.debug(f"Setting seed to {self.training_config.seed}...")
        set_random_seed(self.training_config.seed)

    def train(self):
        """Train the animal."""
        self.logger.debug(f"Training the animal in {self.logdir}...")

        self.config.save(self.logdir / "config.yaml")

        n_envs = self._calc_n_envs()
        self.logger.info(f"Using {n_envs} environments for training...")

        # Setup the environment, model, and callbacks
        env = self._make_env(n_envs)
        eval_env = self._make_env(1, "eval_monitor.csv")
        callback = self._make_callback(env, eval_env)
        model = self._make_model(env)

        # Save the eval environments xml
        # All xml's _should_ be the same
        xml_path = self.logdir / "env.xml"
        cambrian_env: MjCambrianEnv = eval_env.envs[0].unwrapped
        cambrian_env.xml.write(xml_path)

        # Start training
        total_timesteps = self.training_config.total_timesteps
        model.learn(total_timesteps=total_timesteps, callback=callback)
        self.logger.info("Finished training the animal...")

        # Save the policy
        self.logger.info(f"Saving model to {self.logdir}...")
        model.save_policy(self.logdir)
        self.logger.debug(f"Saved model to {self.logdir}...")

        # The finished file indicates to the evo script that the animal is done
        Path(self.logdir / "finished").touch()

    def eval(self, record: bool = False):
        self.config.save(self.logdir / "eval_config.yaml")

        # Update temporary attributes for evaluation
        temp_attrs = []
        if (eval_overrides := self.config.env_config.eval_overrides) is not None:
            temp_attrs.append((self.config.env_config, eval_overrides))

        with setattrs_temporary(*temp_attrs):
            env = self._make_env(1, None)
            model = self._make_model(env)

            n_runs = len(self.config.env_config.maze_configs)
            filename = self.logdir / "eval" if record else None
            record_kwargs = dict(path=filename, save_types=["webp", "png", "gif", "mp4"])
            evaluate_policy(env, model, n_runs, record_kwargs=record_kwargs)

    # ========

    def _calc_seed(self, i: int) -> int:
        return self.training_config.seed + i

    def _calc_n_envs(self) -> int:
        """Calculates the number of environments to use for training based on 
        the memory available.
        
        Will be (ideally) an overestimate. We'll check how much memory is used by a 
        single environment and then calculate the memory used by the model and it's 
        rollout buffer.

        Returns:
            int: The number of environments to use for training. Will be at most the
                number of environments specified in the config.
        """
        if not torch.cuda.is_available():
            return self.training_config.n_envs

        # Get the memory usage before creating the environment
        memory_usage_before, total_memory = get_gpu_memory_usage()

        # We'll create a single environment and then calculate the memory usage
        # after a single step
        eval_env = self._make_env(1, None)
        model = self._make_model(eval_env)
        with torch.no_grad():
            eval_env.reset()
            eval_env.step(model.predict(eval_env.observation_space.sample()))

        # Get the memory usage and the size in bytes of the observation space
        memory_usage_after = get_gpu_memory_usage(return_total_memory=False)
        observation_space_size = get_observation_space_size(eval_env.observation_space)

        # Calculate the memory usage per environment, which is the difference in memory
        # usage after and before creating the environment plus the size of the
        # observation space times the batch size (which is the size of the rollout 
        # buffer)
        memory_usage_per_env = (
            memory_usage_after
            - memory_usage_before
            + observation_space_size * self.training_config.batch_size
        )

        return min(self.training_config.n_envs, int(total_memory // memory_usage_per_env) - 1)


    def _make_env(self, n_envs: int, monitor_csv: str | None = "monitor.csv") -> VecEnv:
        assert n_envs > 0, f"n_envs must be > 0, got {n_envs}."

        config = self.config.copy()
        envs = [make_single_env(config, self._calc_seed(i)) for i in range(n_envs)]

        if n_envs == 1:
            vec_env = DummyVecEnv(envs)
        else:
            vec_env = SubprocVecEnv(envs)
        if monitor_csv is not None:
            vec_env = VecMonitor(vec_env, str(self.logdir / monitor_csv))
        return MjCambrianModel._wrap_env(vec_env)

    def _make_callback(self, env: VecEnv, eval_env: VecEnv) -> BaseCallback:
        """Makes the callbacks."""
        from functools import partial 

        for i, callback in enumerate(self.training_config.callbacks):
            # TODO: is this a good assumption? is there a better way to do this?
            if isinstance(callback, partial):
                self.training_config.callbacks[i] = callback(eval_env)

        return callback

    def _make_model(self, env: VecEnv) -> MjCambrianModel:
        """This method creates the model.

        If available, the weights of a previously trained model are loaded into the new
        model. See `MjCambrianModel` for more details, but because the shape of the
        output may be different between animals, the weights with different shapes
        are ignored.
        """
        model = self.config.training.model(env=env)
        if (checkpoint_path := self.training_config.checkpoint_path) is not None:
            model = MjCambrianModel.load(checkpoint_path, env=env)

        if (policy_path := self.training_config.policy_path) is not None:
            policy_path = Path(policy_path)
            assert (
                policy_path.exists()
            ), f"Checkpoint path {policy_path} does not exist."
            self.logger.info(f"Loading model weights from {policy_path}...")
            model.load_policy(policy_path.parent)
        return model


if __name__ == "__main__":
    from cambrian.utils import MjCambrianArgumentParser

    parser = MjCambrianArgumentParser()

    parser.add_argument(
        "-ao",
        "--animal-overrides",
        nargs="+",
        action="extend",
        type=str,
        help="Override animal config values. Do <config>.<key>=<value>. These are applied to _all_ animals.",
        default=[],
    )
    parser.add_argument(
        "-eo",
        "--eye-overrides",
        nargs="+",
        action="extend",
        type=str,
        help="Override eye config values. Do <config>.<key>=<value>. These are applied to _all_ eyes for _all_ animals.",
        default=[],
    )
    parser.add_argument(
        "-mo",
        "--maze-overrides",
        nargs="+",
        action="extend",
        type=str,
        help="Override maze config values. Do <config>.<key>=<value>. These are applied to _all_ mazes.",
        default=[],
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train the model")
    action.add_argument("--eval", action="store_true", help="Evaluate the model")

    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the evaluation. Only used if `--eval` is passed.",
    )

    args = parser.parse_args()

    config = MjCambrianConfig.load(args.config, overrides=args.overrides, defaults=args.defaults)

    animal_configs = config.env_config.animal_configs
    for animal_name, animal_config in animal_configs.items():
        animal_config = animal_config.merge_with_dotlist(args.animal_overrides)

        eye_configs = animal_config.eye_configs
        for eye_name, eye_config in eye_configs.items():
            eye_config = eye_config.merge_with_dotlist(args.eye_overrides)
            eye_configs[eye_name] = eye_config
        animal_configs[animal_name] = animal_config

    if args.maze_overrides:
        maze_configs_store = config.env_config.maze_configs_store
        maze_configs = config.env_config.maze_configs
        if eval_overrides := config.env_config.eval_overrides:
            if eval_maze_configs := eval_overrides.get("maze_configs"):
                maze_configs += eval_maze_configs
        for maze_name in maze_configs:
            maze_config = maze_configs_store[maze_name]
            maze_config = maze_config.merge_with_dotlist(args.maze_overrides)
            maze_configs_store[maze_name] = maze_config

    runner = MjCambrianTrainer(config)

    if args.train:
        runner.train()
    elif args.eval:
        runner.eval(args.record)
