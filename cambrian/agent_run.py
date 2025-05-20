"""
This is a rough draft code to build an embodied agent
using a URDF, eye config & randomnly initialized weights..
The agent will work in a MUJOCO environment and ouptut fitness
scores based on its performance in the environment.
"""

import argparse
import os.path as osp
from pathlib import Path
from typing import Dict, List, Optional

from hydra_config import run_hydra
from loguru import logger
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)
from tqdm import tqdm

from cambrian import MjCambrianConfig
from cambrian.envs.env import MjCambrianEnv, MjCambrianEnvConfig
from cambrian.language_models import OllamaManager
from cambrian.ml.model import MjCambrianModel
from cambrian.utils import evaluate_policy
from cambrian.utils.wrappers import make_wrapped_env

curr_dir = osp.dirname(osp.abspath(__file__))


class MjCambrianEvaluator:
    """
    Evaluate the agent in the environment & determine its
    fitness score based on its performance.
    """

    def __init__(self, cfg: "MjCambrianConfig"):
        super().__init__()

        self._cfg: MjCambrianConfig = cfg

        self._cfg.expdir.mkdir(parents=True, exist_ok=True)

    def eval(
        self,
        filename: Optional[Path | str] = None,
        record: bool = True,
        load_if_exists: bool = False,
        **callback_kwargs,
    ) -> Optional[float]:
        """
        Evaluate the agent in the environment & determine its fitness
        score based on its performance.
        """

        logger.debug("Starting agent evaluation in env..")
        self._cfg.save(self._cfg.expdir / "eval_cfgs.yaml")
        # Create the environment
        env = self._make_env(
            config=self._cfg.eval_env,
            n_envs=1,
            monitor="eval_monitor.csv",
        )

        # Create the model
        model = self._make_model(env)
        logger.debug("Environment & model created successfully.")

        # Load best model
        if load_if_exists and (self._cfg.expdir / "best_model.zip").exists():
            logger.info("Loading best model...")
            model = model.load(self._cfg.expdir / "best_model")

        # Save the eval environments xml
        cambrian_env: MjCambrianEnv = env.envs[0].unwrapped
        cambrian_env.xml.write(self._cfg.expdir / "eval_env.xml")
        with open(self._cfg.expdir / "compiled_eval_env.xml", "w") as f:
            f.write(cambrian_env.spec.to_xml())

        # Configs for policy evaluation
        n_runs: int = self._cfg.eval_env.n_eval_episodes
        filename: Optional[Path | str] = self._cfg.eval_env.save_filename

        record_kwargs = dict(
            path=self._cfg.expdir / filename,
            save_mode=self._cfg.eval_env.renderer.save_mode,
        )
        if not record:
            record_kwargs = None

        # Evaluate the policy
        evaluate_policy(
            env, model, n_runs, record_kwargs=record_kwargs, **callback_kwargs
        )

        # Calculate fitness
        fitness = self._cfg.trainer.fitness_fn(self._cfg)
        logger.info(f"Fitness score: {fitness / n_runs:.3f}")

        # Save the final fitness to a file
        with open(self._cfg.expdir / f"{filename}_fitness.txt", "w") as f:
            f.write(str(fitness))

        return fitness / n_runs

    def _calc_seed(self, i: int) -> int:
        return self._cfg.seed + i

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
            wrappers = [w for w in self._cfg.trainer.wrappers.values() if w]
            wrapped_env = make_wrapped_env(
                config=config.copy(),
                name=self._cfg.expname,
                wrappers=wrappers,
                seed=self._calc_seed(i),
            )
            envs.append(wrapped_env)

        # Wrap the environments
        # Explicitly set start_method to spawn to avoid using forkserver on mac
        vec_env = (
            DummyVecEnv(envs)
            if n_envs == 1
            else SubprocVecEnv(envs, start_method="spawn")
        )
        if monitor is not None:
            vec_env = VecMonitor(vec_env, str(self._cfg.expdir / monitor))

        # Do an initial reset
        vec_env.reset()
        return vec_env

    def _make_model(self, env: VecEnv) -> MjCambrianModel:
        """This method creates the model."""
        return self._cfg.trainer.model(env=env)


class EyeConfig:
    """
    The eye configuration variables for the agent, that can be altered
    by the language model.
    """

    fov: Optional[List[float]] = None
    num_eyes: Optional[List[int]] = None
    resolution: Optional[List[int]] = None
    lat_range: Optional[List[float]] = None
    lon_range: Optional[List[float]] = None

    def __init__(self, cfg: "MjCambrianConfig"):
        super().__init__()

        self._cfg: Optional[MjCambrianConfig] = None
        self.cfg = cfg

    def __call__(
        self,
    ) -> Dict[str, list]:
        """
        Returns the eye configuration as a dictionary.
        """

        return {
            "fov": self.cfg.fov,
            "num_eyes": self.cfg.num_eyes,
            "resolution": self.cfg.resolution,
            "lat_range": self.cfg.lat_range,
            "lon_range": self.cfg.lon_range,
        }

    def update_cfg(
        self,
        new_eyes: Dict[str, list],
    ) -> MjCambrianConfig:
        """
        Updates the eye configuration with the new values from the LLM.

        Args:
            new_eyes (Dict[str, list]) : New eye configuration from the LLM.

        Returns:
            MjCambrianConfig : Updated configuration for the agent.
        """

        # Get current eye config
        exp_cfg: MjCambrianConfig = MjCambrianConfig.compose(
            config_dir=osp.join(curr_dir, "configs"),
            config_name="base",
            overrides=[
                "example=detection",
                f'env.agents.agent.eyes.eye.resolution={new_eyes["resolution"]}',
                f'env.agents.agent.eyes.eye.fov={new_eyes["fov"]}',
                f'env.agents.agent.eyes.eye.lat_range={new_eyes["lat_range"]}',
                f'env.agents.agent.eyes.eye.lon_range={new_eyes["lon_range"]}',
                f'env.agents.agent.eyes.eye.num_eyes={new_eyes["num_eyes"]}',
            ],
        )

        # Update the eye config
        self.cfg = exp_cfg

        return exp_cfg

    @property
    def cfg(
        self,
    ) -> MjCambrianConfig:
        """
        Returns the current eye configuration.
        """
        return self._cfg

    @cfg.setter
    def cfg(self, config: MjCambrianConfig) -> None:
        """
        Sets the current eye configuration.
        """

        self._cfg = config.eval_env.agents["agent"].eyes["eye"]


def evaluate(config: MjCambrianConfig) -> None:
    """
    Evaluate the agent in the environment by altering the eye configuration
    using the language model.

    Args:
        config (MjCambrianConfig) : Configuration for the agent.
    """

    # Load the initial eye config
    eyes = EyeConfig(config)

    # Initialize the LLM
    llm = OllamaManager()

    # Run the evaluator
    pbar = tqdm(
        range(100),
        desc="Evaluating agent",
    )
    for step, _ in enumerate(pbar):
        # Run the evaluator
        try:
            evaluator = MjCambrianEvaluator(config)
            score: float = evaluator.eval()

        except AssertionError as e:
            logger.error(f"Error running the evaluator: {e}")
            score += -100.0

        pbar.set_postfix({"Score:": round(score, 3)})

        # Run the language model to generate new eye config
        new_eye_cfg: str = llm(fitness_score=score, eye_config=eyes())

        # Update the configurations
        config = eyes.update_cfg(new_eyes=new_eye_cfg)
        logger.debug(f"Eye config: {eyes()}")
        pbar.update(1)


@logger.catch
def main():
    """
    Just run the evaluator given the eye config from
    the language model.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluate the agent.")

    def _main(config: MjCambrianConfig, *, eval: bool):
        """
        Hydra entrypoint for the evaluator.
        """

        if eval:
            evaluate(config)

    # Specify the Hydra config path
    config_path: str = "pkg://cambrian/configs"
    run_hydra(
        _main,
        config_path=config_path,
        parser=parser,
    )


if __name__ == "__main__":
    main()
