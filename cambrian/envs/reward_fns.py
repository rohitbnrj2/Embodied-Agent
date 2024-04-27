from typing import Dict, Any, List, Optional

import numpy as np
import mujoco as mj

from cambrian.envs import MjCambrianEnv
from cambrian.envs.object_env import MjCambrianObjectEnv
from cambrian.animals import MjCambrianAnimal

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
    truncation_reward: float = -1.0,
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
    truncation_reward: float = -1.0,
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
    *,
    factor: float = 1.0,
) -> float:
    """
    Rewards the change in distance over the previous step scaled by the timestep.
    """
    return calc_delta(animal, info, animal.init_pos) * factor


def euclidean_delta_to_object(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    object: str,
    factor: float = 1.0,
):
    """
    Rewards the change in distance to an object over the previous step.
    """
    assert object in env.objects, f"Object {object} not found in environment."
    # Multiply by -1 to reward getting closer to the object
    return calc_delta(animal, info, env.objects[object].pos) * -factor


def reward_if_close_to_object(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    info: Dict[str, Any],
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
    penalty: float = -1.0,
) -> float:
    """Penalizes the animal if it has contacts with the ground."""
    return penalty if info["has_contacts"] else 0.0


def reward_if_animals_in_view(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward_in_view: float = 0.0,
    reward_not_in_view: float = 0.0,
    from_animals: Optional[List[str]] = None,
    to_animals: Optional[List[str]] = None,
    for_animals: Optional[List[str]] = None,
    hfov: float = 45,
    scale_by_distance: bool = False,
) -> float:
    """This reward function rewards the animal if it is in the view of other animals.

    Keyword Args:
        reward_in_view (float): The reward to give the animal if it is in view of
            another animal. Default is 0.
        reward_not_in_view (float): The reward to give the animal if it is not in view
            of another animal. Default is 0.
        from_animals (Optional[List[str]]): The names of the animals that the reward
            should be calculated from. If None, the reward will be calculated from all
            animals.
        to_animals (Optional[List[str]]): The names of the animals that the reward
            should be calculated to. If None, the reward will be calculated to all
            animals.
        for_animals (Optional[List[str]]): The names of the animals that the reward
            should be calculated for. If None, the reward will be calculated for all
            animals.
        hfov (float): The horizontal fov to check whether the to animal is within view
            of the from animal. Default is 45. This is in degrees.
        scale_by_distance (bool): Whether to scale the reward by the distance between
            the animals. Default is False.
    """
    # Early exit if the animal is not in the from_animals list
    if for_animals is not None and animal.name not in for_animals:
        return 0

    def maybe_scale(reward: float, distance: float) -> float:
        return reward * (1 / max(distance, 1)) if scale_by_distance else reward

    accumulated_reward = 0
    from_animals = from_animals or list(env.animals.keys())
    to_animals = to_animals or list(env.animals.keys())
    for from_animal in [env.animals[name] for name in from_animals]:
        for to_animal in [env.animals[name] for name in to_animals]:
            # Extract the position and rotation matrices for both animals
            # We'll use these to calculate the direction the from animal is facing
            # relative to the to animal. This will determine if the to animal is in
            # view of the from animal.
            vec = to_animal.pos - from_animal.pos
            dist_to_animal = np.linalg.norm(vec)
            yaw = np.arctan2(from_animal.mat[1, 0], from_animal.mat[0, 0])
            relative_yaw = np.arctan2(vec[1], vec[0]) - yaw

            # Early exit if the animal isn't within view of the from animal
            if np.abs(relative_yaw) > np.deg2rad(hfov) / 2:
                accumulated_reward += maybe_scale(reward_not_in_view, dist_to_animal)
                continue

            # Now we'll trace a ray between the two animals to make sure the animal is
            # within view
            geomid = np.zeros(1, np.int32)
            mj.mj_ray(
                env.model,
                env.data,
                from_animal.pos,
                vec,
                from_animal.geomgroup_mask,  # mask out this animal to avoid self-collision
                1,  # include static geometries
                -1,  # include all bodies
                geomid,
            )

            # Early exit again if the animal is occluded
            if geomid != to_animal.geom.id:
                accumulated_reward += maybe_scale(reward_not_in_view, dist_to_animal)
                continue

            accumulated_reward += maybe_scale(reward_in_view, dist_to_animal)
    return accumulated_reward


def combined_reward(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    **reward_fns,
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
