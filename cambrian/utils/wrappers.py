"""Wrappers for the MjCambrianEnv. Used during training."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

from cambrian.envs import MjCambrianEnv, MjCambrianEnvConfig
from cambrian.utils import is_integer


class MjCambrianSingleAgentEnvWrapper(gym.Wrapper):
    """Wrapper around the MjCambrianEnv that acts as if there is a single agent.

    Will replace all multi-agent methods to just use the first agent.

    Keyword Args:
        agent_name: The name of the agent to use. If not provided, the first agent
            will be used.
    """

    def __init__(
        self,
        env: MjCambrianEnv,
        *,
        agent_name: Optional[str] = None,
        combine_rewards: bool = True,
        combine_terminated: bool = True,
        combine_truncated: bool = True,
    ):
        super().__init__(env)

        self._combine_rewards = combine_rewards
        self._combine_terminated = combine_terminated
        self._combine_truncated = combine_truncated

        agent_name = agent_name or next(iter(env.agents.keys()))
        assert agent_name in env.agents, f"agent {agent_name} not found."
        self._agent = env.agents[agent_name]
        self.action_space = env.action_space(agent_name)
        self.observation_space = env.observation_space(agent_name)

    def reset(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)

        return obs[self._agent.name], info[self._agent.name]

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        action = {self._agent.name: action}
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = obs[self._agent.name]
        info = info[self._agent.name]

        if self._combine_rewards:
            reward = np.sum(list(reward.values()))
        else:
            reward = reward[self._agent.name]

        if self._combine_terminated:
            terminated = any(terminated.values())
        else:
            terminated = terminated[self._agent.name]

        if self._combine_truncated:
            truncated = any(truncated.values())
        else:
            truncated = truncated[self._agent.name]

        return obs, reward, terminated, truncated, info


class MjCambrianPettingZooEnvWrapper(gym.Wrapper):
    """Wrapper around the MjCambrianEnv that acts as if there is a single agent, where
    in actuality, there's multi-agents.

    SB3 doesn't support Dict action spaces, so this wrapper will flatten the action
    into a single space. The observation can be a dict; however, nested dicts are not
    allowed.

    Note:
        All agents must be trainable
    """

    def __init__(self, env: MjCambrianEnv):
        super().__init__(env)
        self.env: MjCambrianEnv

    def reset(self, *args, **kwargs) -> Dict[str, Any]:
        obs, info = self.env.reset(*args, **kwargs)

        # Flatten the observations
        flattened_obs: Dict[str, Any] = {}
        for agent_name, agent_obs in obs.items():
            if isinstance(agent_obs, dict):
                for key, value in agent_obs.items():
                    flattened_obs[f"{agent_name}_{key}"] = value
            else:
                flattened_obs[agent_name] = agent_obs

        return flattened_obs, info

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        # Convert the action back to a dict
        action = action.reshape(-1, len(self.env.agents))
        action = {
            agent_name: action[:, i]
            for i, agent_name in enumerate(self.env.agents.keys())
            if self.env.agents[agent_name].config.trainable
        }

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Accumulate the rewards, terminated, and truncated
        reward = sum(reward.values())
        terminated = any(terminated.values())
        truncated = any(truncated.values())

        # Flatten the observations
        flattened_obs: Dict[str, Any] = {}
        for agent_name, agent_obs in obs.items():
            if isinstance(agent_obs, dict):
                for key, value in agent_obs.items():
                    flattened_obs[f"{agent_name}_{key}"] = value
            else:
                flattened_obs[agent_name] = agent_obs

        return flattened_obs, reward, terminated, truncated, info

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """SB3 doesn't support nested Dict observation spaces, so we'll flatten it.
        If each agent has a Dict observation space, we'll flatten it into a single
        observation where the key in the dict is the agent name and the original space
        name."""
        observation_space: Dict[str, gym.Space] = {}
        for agent in self.env.agents.values():
            # Ignore non-trainable agents
            if not agent.config.trainable:
                continue

            agent_observation_space = self.env.observation_space(agent.name)
            if isinstance(agent_observation_space, gym.spaces.Dict):
                for key, value in agent_observation_space.spaces.items():
                    observation_space[f"{agent.name}_{key}"] = value
            else:
                observation_space[agent.name] = agent_observation_space
        return gym.spaces.Dict(observation_space)

    @property
    def action_space(self) -> gym.spaces.Box:
        """The only gym.Space that SB3 supports that's continuous for the action space
        is a Box. We can assume each agent's action space is a Box, so we'll flatten
        each action space into one Box for the environment.

        Assumptions:
            - All agents have the same number of actions
            - All actions have the same shape
            - All actions are continuous
            - All actions are normalized between -1 and 1
        """

        # Get the first agent's action space
        first_agent_name = next(iter(self.env.agents.keys()))
        first_agent_action_space = self.env.action_space(first_agent_name)

        # Check if the action space is continuous
        assert isinstance(first_agent_action_space, gym.spaces.Box), (
            "SB3 only supports continuous action spaces for the environment. "
            f"agent {first_agent_name} has a {type(first_agent_action_space)}"
            " action space."
        )

        # Get the shape of the action space
        shape = first_agent_action_space.shape
        low = first_agent_action_space.low
        high = first_agent_action_space.high

        # Check if all agents have the same number of actions
        for agent_name, agent_action_space in self.env.action_spaces.items():
            assert shape == agent_action_space.shape, (
                "All agents must have the same number of actions. "
                f"agent {first_agent_name} has {shape} actions, but {agent_name} "
                f"has {agent_action_space.shape} actions."
            )

            # Check if the action space is continuous
            assert isinstance(agent_action_space, gym.spaces.Box), (
                "SB3 only supports continuous action spaces for the environment. "
                f"agent {first_agent_name} has a "
                f"{type(first_agent_action_space)} action space."
            )

            assert all(low == agent_action_space.low), (
                "All actions must have the same low value. "
                f"agent {first_agent_name} has a low value of {low}, "
                f"but {agent_name} has a low value of {agent_action_space.low}."
            )

            assert all(high == agent_action_space.high), (
                "All actions must have the same high value. "
                f"agent {first_agent_name} has a high value of {high}, "
                f"but {agent_name} has a high value of {agent_action_space.high}."
            )

        low = np.tile(low, len(self.env.agents))
        high = np.tile(high, len(self.env.agents))
        shape = (shape[0] * len(self.env.agents),)
        return gym.spaces.Box(
            low=low, high=high, shape=shape, dtype=first_agent_action_space.dtype
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

        self._constant_action_indices = [
            int(k) if is_integer(k) else k for k in constant_actions.keys()
        ]
        self._constant_action_values = list(constant_actions.values())

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if isinstance(action, dict):
            assert all(idx in action for idx in self._constant_action_indices), (
                "The constant action indices must be in the action space."
                f"Indices: {self._constant_action_indices}, Action space: {action}"
            )
        action[self._constant_action_indices] = self._constant_action_values

        return self.env.step(action)


def make_wrapped_env(
    config: MjCambrianEnvConfig,
    wrappers: List[Callable[[gym.Env], gym.Env]],
    seed: Optional[int] = None,
    **kwargs,
) -> gym.Env:
    """Utility function for creating a MjCambrianEnv."""

    def _init():
        env = config.instance(config, **kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        # check_env will call reset and set the seed to 0; call set_random_seed after
        check_env(env, warn=False)
        env.unwrapped.set_random_seed(seed)
        return env

    return _init
