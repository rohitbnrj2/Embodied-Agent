from typing import Dict, Any, List, Optional

import numpy as np
import mujoco as mj

from cambrian.envs import MjCambrianEnv
from cambrian.envs.object_env import MjCambrianObjectEnv
from cambrian.animals import MjCambrianAnimal

# =====================
# Reward functions


def reward_for_termination(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
) -> float:
    """Terminated indicates that the episode was ended early in a success.
    Returns termination_reward if terminated, else reward."""
    return reward if terminated else 0.0


def reward_for_truncation(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
) -> float:
    """Truncated indicates that the episode was ended early in a failure.
    Returns truncation_reward if truncated, else reward."""
    return reward if truncated else 0.0


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
    if object not in env.objects:
        return 0.0
    # Multiply by -1 to reward getting closer to the object
    return -1 * calc_delta(animal, info, env.objects[object].pos) * factor


def reward_if_close_to_object(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    objects: Optional[List[str]] = None,
    distance_threshold: float = 2.0,
) -> float:
    """Rewards the animal if it is close to an object.

    Keyword Args:
        objects (Optional[List[str]]): List of object names to check for closeness.
            If None, all objects are checked. Defaults to None.
        distance_threshold (float): Distance threshold for closeness. Defaults to 2.0.
    """
    accumulated_reward = 0
    for obj in env.objects.values():
        if objects is not None and obj.name not in objects:
            continue

        if np.linalg.norm(obj.pos - animal.pos) < distance_threshold:
            accumulated_reward += reward
    return accumulated_reward


def penalize_if_has_contacts(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    penalty: float,
) -> float:
    """Penalizes the animal if it has contacts with the ground."""
    return penalty if info.get("has_contacts", False) else 0.0


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

    accumulated_reward = 0
    from_animals = from_animals or list(env.animals.keys())
    to_animals = to_animals or list(env.animals.keys())
    for from_animal in [env.animals[name] for name in from_animals]:
        for to_animal in [env.animals[name] for name in to_animals]:
            # Check if the to_animal is in view of the from_animal
            in_view = check_in_view(
                env.model,
                env.data,
                from_animal,
                to_animal.pos,
                to_animal.geom.id,
                hfov=hfov,
            )

            # Add the reward to the accumulated reward. It may be scaled by the distance
            # if scale_by_distance is True, but only if the animal is in view.
            if in_view:
                dist = np.linalg.norm(to_animal.pos - from_animal.pos)
                scale = 1 / max(dist, 1) if scale_by_distance else 1
                reward = reward_in_view * scale
            else:
                reward = reward_not_in_view
            accumulated_reward += reward

    return accumulated_reward


def reward_if_objects_in_view(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward_in_view: float = 0.0,
    reward_not_in_view: float = 0.0,
    from_animals: Optional[List[str]] = None,
    to_objects: Optional[List[str]] = None,
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
        to_objects (Optional[List[str]]): The names of the objects that the reward
            should be calculated to. If None, the reward will be calculated to all
            objects.
        for_animals (Optional[List[str]]): The names of the animals that the reward
            should be calculated for. If None, the reward will be calculated for all
            animals.
        hfov (float): The horizontal fov to check whether the to object is within view
            of the from animal. Default is 45. This is in degrees.
        scale_by_distance (bool): Whether to scale the reward by the distance between
            the animals. Default is False.
    """
    # Early exit if the animal is not in the from_animals list
    if for_animals is not None and animal.name not in for_animals:
        return 0

    accumulated_reward = 0
    from_animals = from_animals or list(env.animals.keys())
    to_objects = to_objects or list(env.objects.keys())
    for from_animal in [env.animals[name] for name in from_animals]:
        for to_object in [env.objects[name] for name in to_objects]:
            # Check if the to_object is in view of the from_animal
            in_view = check_in_view(
                env.model,
                env.data,
                from_animal,
                to_object.pos,
                to_object.geomid,
                hfov=hfov,
            )

            # Add the reward to the accumulated reward. It may be scaled by the distance
            # if scale_by_distance is True, but only if the object is in view.
            if in_view:
                dist = np.linalg.norm(to_object.pos - from_animal.pos)
                scale = 1 / max(dist, 1) if scale_by_distance else 1
                reward = reward_in_view * scale
            else:
                reward = reward_not_in_view
            accumulated_reward += reward

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


def check_in_view(
    model: mj.MjModel,
    data: mj.MjData,
    animal: MjCambrianAnimal,
    to_pos: np.ndarray,
    to_geomid: int,
    hfov: float = 45,
) -> bool:
    """Checks if the to_pos is in the field of view of the animal."""
    vec = to_pos - animal.pos
    yaw = np.arctan2(animal.mat[1, 0], animal.mat[0, 0])
    relative_yaw = np.arctan2(vec[1], vec[0]) - yaw

    # Early exit if the to_pos isn't within view of the from_pos
    if np.abs(relative_yaw) > np.deg2rad(hfov) / 2:
        return False

    # Now we'll trace a ray between the two points to check if there are any obstacles
    geomid = np.zeros(1, np.int32)
    mj.mj_ray(
        model,
        data,
        animal.pos,
        vec,
        animal.geomgroup_mask,  # mask out this animal to avoid self-collision
        1,  # include static geometries
        -1,  # include all bodies
        geomid,
    )

    # If the ray hit the to geom, then the to_pos is in view
    return geomid[0] == to_geomid
