from pathlib import Path

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
    CallbackList,
)

from feature_extractors import MjCambrianCombinedExtractor
from config import convert_overrides_to_dict, MjCambrianConfig
from animal import MjCambrianAnimal
from animal_pool import MjCambrianAnimalPool
from wrappers import make_single_env
from callbacks import (
    PlotEvaluationCallback,
    SaveVideoCallback,
    CallbackListWithSharedParent,
    MjCambrianProgressBarCallback,
    MjCambrianAnimalPoolCallback,
)
from ppo import MjCambrianPPO


class MjCambrianEvoRunner:
    def __init__(self, config: MjCambrianConfig, rank: int):
        self.config = config
        self.rank = rank

        self.verbose = self.config.training_config.verbose

        self.logdir = (
            Path(self.config.training_config.logdir)
            / self.config.training_config.exp_name
        )
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.generation = 0
        self.animal_pool = MjCambrianAnimalPool.create(self.config, rank)

    # ========

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

            self.update_logdir()

            config = self.select_animal()
            config = self.mutate_animal(config)
            self.train_animal(config)

            self.generation += 1

    def update_logdir(self):
        generation = f"generation_{self.generation}"
        rank = f"rank_{self.rank}"
        self.generation_logdir = self.logdir / generation / rank
        self.generation_logdir.mkdir(parents=True, exist_ok=True)

    def select_animal(self) -> MjCambrianConfig:
        return self.animal_pool.get_new_config()

    def mutate_animal(self, config: MjCambrianConfig) -> MjCambrianConfig:
        animal_config = config.animal_config.copy()
        animal_config = MjCambrianAnimal.mutate(animal_config, verbose=self.verbose)
        return config.copy(animal_config=animal_config)

    def train_animal(self, config: MjCambrianConfig):
        self.config = config
        self.config.write_to_yaml(self.generation_logdir / "config.yaml")

        self.train()

    # ========

    def train(self):
        env = self._make_env(self.config.training_config.n_envs)
        eval_env = self._make_env(1)
        callback = self._make_callback(env, eval_env)
        model = self._make_model(env)

        print(f"Beginning training for generation {self.generation}...")
        total_timesteps = self.config.training_config.total_timesteps
        model.learn(total_timesteps=total_timesteps, callback=callback)
        print(f"Finished training for generation {self.generation}...")

        print(f"Saving model to {self.generation_logdir}...")
        model.save_policy(self.generation_logdir)

        env.close()
        eval_env.close()

    def eval(self):
        pass

    # ========

    def _make_env(self, n_envs: int) -> VecEnv:
        assert n_envs > 0, f"n_envs must be > 0, got {n_envs}."

        def calc_seed(i: int) -> int:
            """Calculates a unique seed for each environment.

            Equation is as follows:
                i * population_size * num_generations + seed + generation
            """
            return (
                i
                * self.config.evo_config.population_size
                * self.config.evo_config.num_generations
                + self.config.training_config.seed
                + self.generation
            )

        envs = [make_single_env(self.config, calc_seed(i)) for i in range(n_envs)]

        if n_envs == 1:
            vec_env = DummyVecEnv(envs)
        else:
            vec_env = SubprocVecEnv(envs)
        return VecMonitor(vec_env, str(self.generation_logdir / "monitor.csv"))

    def _make_callback(self, env: VecEnv, eval_env: VecEnv) -> BaseCallback:
        callbacks_on_new_best = []
        callbacks_on_new_best.append(
            SaveVideoCallback(
                eval_env,
                self.generation_logdir,
                self.config.training_config.max_episode_steps,
                verbose=self.verbose,
            )
        )
        callbacks_on_new_best.append(
            StopTrainingOnNoModelImprovement(
                self.config.training_config.max_no_improvement_evals,
                self.config.training_config.min_no_improvement_evals,
                verbose=self.verbose,
            )
        )
        if self.animal_pool is not None:
            callbacks_on_new_best.append(
                MjCambrianAnimalPoolCallback(
                    eval_env,
                    self.animal_pool,
                    verbose=self.verbose,
                )
            )
        callbacks_on_new_best = CallbackListWithSharedParent(callbacks_on_new_best)

        eval_cb = EvalCallback(
            env,
            best_model_save_path=self.generation_logdir,
            log_path=self.generation_logdir,
            eval_freq=self.config.training_config.eval_freq,
            deterministic=True,
            render=False,
            callback_on_new_best=callbacks_on_new_best,
            callback_after_eval=PlotEvaluationCallback(self.generation_logdir),
        )

        return CallbackList([eval_cb, MjCambrianProgressBarCallback()])

    def _make_model(self, env: VecEnv) -> MjCambrianPPO:
        policy_kwargs = dict(
            features_extractor_class=MjCambrianCombinedExtractor,
        )
        model = MjCambrianPPO(
            "MultiInputPolicy",
            env,
            n_steps=self.config.training_config.n_steps,
            batch_size=self.config.training_config.batch_size,
            learning_rate=self.config.training_config.learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=self.verbose,
        )
        if self.config.training_config.checkpoint_path is not None:
            path = self.logdir / self.config.training_config.checkpoint_path
            assert path.exists(), f"Checkpoint `{path}` doesn't exist."

            print(f"Loading model from {path}...")
            model = model.load_policy(path)
        return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("-r", "--rank", type=int, help="Rank of this process", default=0)
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action="append",
        nargs=2,
        help="Override config values. Do <dot separated yaml config> <value>",
        default=[],
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--evo", action="store_true", help="Run evolution")
    action.add_argument("--eval", action="store_true", help="Evaluate the model")

    args = parser.parse_args()

    overrides = convert_overrides_to_dict(args.overrides)
    config: MjCambrianConfig = MjCambrianConfig.load(args.config, overrides=overrides)
    config.training_config.setdefault("exp_name", Path(args.config).stem)

    runner = MjCambrianEvoRunner(config, args.rank)

    if args.evo:
        runner.evo()
    elif args.eval:
        runner.eval()
