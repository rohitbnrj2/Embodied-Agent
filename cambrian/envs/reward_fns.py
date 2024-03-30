from typing import Dict, Any

import numpy as np

from cambrian.envs.env import MjCambrianEnv
from cambrian.envs.object_env import MjCambrianObjectEnv
from cambrian.animals.animal import MjCambrianAnimal

# =====================
# Common reward logic


def apply_termination_reward(
    reward: float, terminated: bool, termination_reward: float = 1.0
) -> float:
    """Terminated indicates that the episode was ended early in a success.
    Returns termination_reward if terminated, else reward."""
    return termination_reward if terminated else reward


def apply_truncation_reward(
    reward: float, truncated: bool, *, truncation_reward: float = -1.0
) -> float:
    """Truncated indicates that the episode was ended early in a failure.
    Returns truncation_reward if truncated, else reward."""
    return truncation_reward if truncated else reward


def postprocess_reward(
    reward: float,
    terminated: bool,
    truncated: bool,
    *,
    termination_reward: float = 1.0,
    truncation_reward: float = -1.0
) -> float:
    """Applies termination and truncation rewards to the reward."""
    reward = apply_termination_reward(
        reward, terminated, termination_reward=termination_reward
    )
    reward = apply_truncation_reward(
        reward, truncated, truncation_reward=truncation_reward
    )
    return reward


# =====================
# Reward functions


def termination_and_truncation_only(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    termination_reward: float = 1.0,
    truncation_reward: float = -1.0
) -> float:
    """Rewards the animal for reaching the target."""
    return postprocess_reward(
        0,
        terminated,
        truncated,
        termination_reward=termination_reward,
        truncation_reward=truncation_reward,
    )


def euclidean_delta_from_init(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
) -> float:
    """
    Rewards the change in distance over the previous step scaled by the timestep.
    """
    return calc_delta(animal, info, animal.init_pos)


def reward_if_close_to_object(
    env: MjCambrianObjectEnv, animal: MjCambrianAnimal, info: Dict[str, Any]
) -> float:
    """Terminates the episode if the animal is close to an object. Terminate is only
    true if the object is set to terminate_if_close = True."""
    reward = 0
    for obj in env.objects.values():
        if obj.is_close(animal.pos):
            reward += obj.config.reward_if_close
    return reward


def penalize_if_has_contacts(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    penalty: float = -1.0
) -> float:
    """Penalizes the animal if it has contacts with the ground."""
    return penalty if info["has_contacts"] else 0.0


def combined_reward(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    **reward_fns
) -> float:
    """Combines multiple reward functions into one."""
    reward = 0
    for fn in reward_fns.values():
        reward += fn(env, animal, terminated, truncated, info)
    return reward


# =====================
# Utility functions


def calc_delta(
    animal: MjCambrianAnimal, info: Dict[str, Any], point: np.ndarray = np.array([0, 0])
) -> np.ndarray:
    """Calculates the delta position of the animal from a point.

    NOTE: returns delta position of current pos from the previous pos to the point
    (i.e. current - prev)
    """

    current_distance = np.linalg.norm(animal.pos - point)
    prev_distance = np.linalg.norm(info["prev_pos"] - point)
    return current_distance - prev_distance
