from typing import Dict, Any, Tuple
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

from cambrian.env import MjCambrianEnv
from cambrian.utils import MjCambrianConfig


class MjCambrianSingleAnimalEnvWrapper(gym.Wrapper):
    """Wrapper around the MjCambrianEnv that acts as if there is a single animal.

    Will replace all multi-agent methods to just use the first animal.
    """

    def __init__(self, env: MjCambrianEnv):
        super().__init__(env)

        self.animal = next(iter(env.animals.values()))
        self.action_space = next(iter(env.action_spaces.values()))
        self.observation_space = next(iter(env.observation_spaces.values()))

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)

        return obs[self.animal.name], info[self.animal.name]

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        action = {self.animal.name: action}
        obs, reward, terminated, truncated, info = self.env.step(action)

        return (
            obs[self.animal.name],
            reward[self.animal.name],
            terminated[self.animal.name],
            truncated[self.animal.name],
            info[self.animal.name],
        )


def make_single_env(
    config: Path | str | MjCambrianConfig, seed: int, **kwargs
) -> MjCambrianSingleAnimalEnvWrapper:
    """Utility function for multiprocessed MjCambrianSingleAnimalEnvWrapper."""

    def _init():
        env = MjCambrianEnv(config, **kwargs)
        env = MjCambrianSingleAnimalEnvWrapper(env)
        env.reset(seed=seed)
        check_env(env)
        return env

    return _init
