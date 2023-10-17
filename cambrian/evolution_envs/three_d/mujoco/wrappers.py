from typing import Dict, Any, Tuple
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed

from env import MjCambrianEnv


def make_single_env(
    rank: int, seed: float, config_path: str | Path
) -> "MjCambrianSingleAnimalEnvWrapper":
    """Utility function for multiprocessed MjCambrianSingleAnimalEnvWrapper."""

    def _init():
        env = MjCambrianEnv(config_path)
        env = MjCambrianSingleAnimalEnvWrapper(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed + rank)
    return _init


class MjCambrianSingleAnimalEnvWrapper(gym.Wrapper):
    """Wrapper around the MjCambrianEnv that acts as if there is a single animal.

    Will replace all multi-agent methods to just use the first animal.
    """

    def __init__(self, env: MjCambrianEnv):
        super().__init__(env)

        self.animal = next(iter(env.animals.values()))
        self.action_space = self.animal.action_space
        self.observation_space = self.animal.observation_space

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)

        return obs[self.animal.name], info[self.animal.name]

    def step(
        self, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        action = {self.animal.name: action}
        obs, reward, terminated, truncated, info = self.env.step(action)

        return (
            obs[self.animal.name],
            reward[self.animal.name],
            terminated[self.animal.name],
            truncated[self.animal.name],
            info[self.animal.name],
        )
