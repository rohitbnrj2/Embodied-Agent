from typing import Dict, Any, Tuple, Optional, Callable, List

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

from cambrian.envs.env import MjCambrianEnv, MjCambrianEnvConfig


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

class MjCambrianConstantActionWrapper(gym.Wrapper):
    """This wrapper will apply a constant action at specific indices of the action 
    space.
    
    Args:
        constant_actions: A dictionary where the keys are the indices of the action 
            space and the values are the constant actions to apply.
    """

    def __init__(self, env: MjCambrianEnv, constant_actions: Dict[Any, Any]):
        super().__init__(env)

        self.constant_action_indices = list(constant_actions.keys())
        self.constant_action_values = list(constant_actions.values())

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        action[self.constant_action_indices] = self.constant_action_values

        return self.env.step(action)

def make_wrapped_env(
    config: MjCambrianEnvConfig, wrappers: List[Callable[[gym.Env], gym.Env]], seed: Optional[int] = None, **kwargs
) -> gym.Env:
    """Utility function for creating a MjCambrianEnv."""

    def _init():
        env = config.instance(config, **kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        env.reset(seed=seed)
        check_env(env, warn=False)
        return env

    return _init
