from typing import Dict, Any, Tuple, Optional, Callable, List, Final, SupportsFloat
from collections import deque
from functools import singledispatch

import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
from gymnasium.error import CustomSpaceError
from gymnasium.spaces.space import T_cov
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

    def __init__(self, env: MjCambrianEnv, *, agent_name: Optional[str] = None):
        super().__init__(env)

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

        return (
            obs[self._agent.name],
            reward[self._agent.name],
            terminated[self._agent.name],
            truncated[self._agent.name],
            info[self._agent.name],
        )


class MjCambrianPettingZooEnvWrapper(gym.Wrapper):
    """Wrapper around the MjCambrianEnv that acts as if there is a single agent, where
    in actuallity, there's multi-agents.

    SB3 doesn't support Dict action spaces, so this wrapper will flatten the action
    into a single space. The observation can be a dict; however, nested dicts are not
    allowed.

    NOTE: All agents must be trainable
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
            f"agent {first_agent_name} has a {type(first_agent_action_space)} action space."
        )

        # Get the shape of the action space
        shape = first_agent_action_space.shape
        low = first_agent_action_space.low
        high = first_agent_action_space.high

        # Check if all agents have the same number of actions
        for agent_name, agent_action_space in self.env.action_spaces.items():
            assert shape == agent_action_space.shape, (
                "All agents must have the same number of actions. "
                f"agent {first_agent_name} has {shape} actions, but {agent_name} has {agent_action_space.shape} actions."
            )

            # Check if the action space is continuous
            assert isinstance(agent_action_space, gym.spaces.Box), (
                "SB3 only supports continuous action spaces for the environment. "
                f"agent {first_agent_name} has a {type(first_agent_action_space)} action space."
            )

            assert all(low == agent_action_space.low), (
                "All actions must have the same low value. "
                f"agent {first_agent_name} has a low value of {low}, but {agent_name} has a low value of {agent_action_space.low}."
            )

            assert all(high == agent_action_space.high), (
                "All actions must have the same high value. "
                f"agent {first_agent_name} has a high value of {high}, but {agent_name} has a high value of {agent_action_space.high}."
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


# =============================================================================
# Classes below are copies from gymnasium=v1.0.0. stable baselines has not
# been updated to work with gymnasium=v1.0.0 yet.


@singledispatch
def create_zero_array(space: gym.Space[T_cov]) -> T_cov:
    """Creates a zero-based array of a space, this is similar to ``create_empty_array`` except all arrays are valid samples from the space.

    As some ``Box`` cases have ``high`` or ``low`` that don't contain zero then the ``create_empty_array`` would in case
    create arrays which is not contained in the space.

    Args:
        space: The space to create a zero array for

    Returns:
        Valid sample from the space that is as close to zero as possible
    """
    if isinstance(space, gym.Space):
        raise CustomSpaceError(
            f"Space of type `{type(space)}` doesn't have an registered `create_zero_array` function. Register `{type(space)}` for `create_zero_array` to support it."
        )
    else:
        raise TypeError(
            f"The space provided to `create_zero_array` is not a gymnasium Space instance, type: {type(space)}, {space}"
        )


@create_zero_array.register(gym.spaces.Dict)
def _create_dict_zero_array(space: gym.spaces.Dict):
    return {key: create_zero_array(subspace) for key, subspace in space.spaces.items()}


@create_zero_array.register(gym.spaces.Box)
def _create_box_zero_array(space: gym.spaces.Box):
    zero_array = np.zeros(space.shape, dtype=space.dtype)
    zero_array = np.where(space.low > 0, space.low, zero_array)
    zero_array = np.where(space.high < 0, space.high, zero_array)
    return zero_array


@create_zero_array.register(gym.spaces.Discrete)
def _create_discrete_zero_array(space: gym.spaces.Discrete):
    return 0


class FrameStackObservation(
    gym.Wrapper[WrapperObsType, ActType, ObsType, ActType],
    gym.utils.RecordConstructorArgs,
):
    """Stacks the observations from the last ``N`` time steps in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Users have options for the padded observation used:

     * "reset" (default) - The reset value is repeated
     * "zero" - A "zero"-like instance of the observation space
     * custom - An instance of the observation space

    No vector version of the wrapper exists.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FrameStackObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env = FrameStackObservation(env, stack_size=4)
        >>> env.observation_space
        Box(0, 255, (4, 96, 96, 3), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)

    Example with different padding observations:
        >>> env = gym.make("CartPole-v1")
        >>> env.reset(seed=123)
        (array([ 0.01823519, -0.0446179 , -0.02796401, -0.03156282], dtype=float32), {})
        >>> stacked_env = FrameStackObservation(env, 3)   # the default is padding_type="reset"
        >>> stacked_env.reset(seed=123)
        (array([[ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
              dtype=float32), {})


        >>> stacked_env = FrameStackObservation(env, 3, padding_type="zero")
        >>> stacked_env.reset(seed=123)
        (array([[ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
              dtype=float32), {})
        >>> stacked_env = FrameStackObservation(env, 3, padding_type=np.array([1, -1, 0, 2], dtype=np.float32))
        >>> stacked_env.reset(seed=123)
        (array([[ 1.        , -1.        ,  0.        ,  2.        ],
               [ 1.        , -1.        ,  0.        ,  2.        ],
               [ 0.01823519, -0.0446179 , -0.02796401, -0.03156282]],
              dtype=float32), {})

    Change logs:
     * v0.15.0 - Initially add as ``FrameStack`` with support for lz4
     * v1.0.0 - Rename to ``FrameStackObservation`` and remove lz4 and ``LazyFrame`` support
                along with adding the ``padding_type`` parameter

    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        stack_size: int,
        *,
        padding_type: str | ObsType = "reset",
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env: The environment to apply the wrapper
            stack_size: The number of frames to stack.
            padding_type: The padding type to use when stacking the observations, options: "reset", "zero", custom obs
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, stack_size=stack_size, padding_type=padding_type
        )
        gym.Wrapper.__init__(self, env)

        if not np.issubdtype(type(stack_size), np.integer):
            raise TypeError(
                f"The stack_size is expected to be an integer, actual type: {type(stack_size)}"
            )
        if isinstance(padding_type, str) and (
            padding_type == "reset" or padding_type == "zero"
        ):
            self.padding_value: ObsType = create_zero_array(env.observation_space)
        elif padding_type in env.observation_space:
            self.padding_value = padding_type
            padding_type = "_custom"
        else:
            if isinstance(padding_type, str):
                raise ValueError(  # we are guessing that the user just entered the "reset" or "zero" wrong
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r}"
                )
            else:
                raise ValueError(
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r} not an instance of env observation ({env.observation_space})"
                )

        self.observation_space = batch_space(env.observation_space, n=stack_size)
        self.stack_size: Final[int] = stack_size
        self.padding_type: Final[str] = padding_type

        self.obs_queue = deque(
            [self.padding_value for _ in range(self.stack_size)], maxlen=self.stack_size
        )
        self.stacked_obs = create_empty_array(env.observation_space, n=self.stack_size)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Reset the environment, returning the stacked observation and info.

        Args:
            seed: The environment seed
            options: The reset options

        Returns:
            The stacked observations and info
        """
        obs, info = self.env.reset(seed=seed, options=options)

        if self.padding_type == "reset":
            self.padding_value = obs
        for _ in range(self.stack_size - 1):
            self.obs_queue.append(self.padding_value)
        self.obs_queue.append(obs)

        updated_obs = concatenate(
            self.env.observation_space, self.obs_queue, self.stacked_obs
        )
        return updated_obs, info

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and info from the environment
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs_queue.append(obs)

        updated_obs = concatenate(
            self.env.observation_space, self.obs_queue, self.stacked_obs
        )
        return updated_obs, reward, terminated, truncated, info
