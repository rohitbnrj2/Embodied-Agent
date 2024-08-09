"""Truncation indicates a failure before the episode is over."""

from typing import Any, Dict, Optional, List

import numpy as np

from cambrian.envs import MjCambrianEnv
from cambrian.envs.object_env import MjCambrianObjectEnv
from cambrian.animals import MjCambrianAnimal

# =====================
# Truncation functions


def never_truncates(
    env: MjCambrianEnv, animal: MjCambrianAnimal, info: Dict[str, Any]
) -> bool:
    """Never terminates the episode."""
    return False


def exceeds_max_episode_steps(
    env: MjCambrianEnv, animal: MjCambrianAnimal, info: Dict[str, Any]
) -> bool:
    """Truncates the episode if the episode step exceeds the max episode steps."""
    return env.episode_step >= (env.max_episode_steps - 1)


def truncate_if_close_to_object(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    info: Dict[str, Any],
    *,
    objects: Optional[List[str]] = None,
    distance_threshold: float = 2.0,
) -> bool:
    """Truncates the episode if the animal is close to an object.

    Keyword Args:
        objects (Optional[List[str]]): List of object names to check for closeness.
            If None, all objects are checked. Defaults to None.
        distance_threshold (float): Distance threshold for closeness. Defaults to 2.0.
    """
    for obj in env.objects.values():
        if objects is not None and obj.name not in objects:
            continue

        if np.linalg.norm(obj.pos - animal.pos) < distance_threshold:
            return True
    return False


def truncate_if_close_to_animal(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    info: Dict[str, Any],
    *,
    animal_name: str,
    distance_threshold: float,
) -> bool:
    """Truncates the episode if the animal is close to another animal."""
    pos = animal.pos
    other_pos = env.animals[animal_name].pos
    return True if np.linalg.norm(pos - other_pos) < distance_threshold else False


def truncate_if_has_contacts(
    env: MjCambrianEnv, animal: MjCambrianAnimal, info: Dict[str, Any]
) -> bool:
    """Truncates the episode if the animal has contacts."""
    return info["has_contacts"]


def truncate_if_animal_facing_objects(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    info: Dict[str, Any],
    *,
    from_animals: Optional[List[str]] = None,
    to_objects: Optional[List[str]] = None,
    for_animals: Optional[List[str]] = None,
    angle_threshold: float = 45,
    n_frames: int = 1,
) -> bool:
    """Truncates the episode if an animal is facing another object.

    Keyword Args:
        from_animals (Optional[List[str]]): List of animal names to check for facing.
            If None, all animals are checked. Defaults to None.
        to_objects (Optional[List[str]]): List of object names to check for facing.
            If None, all objects are checked. Defaults to None.
        for_animals (Optional[List[str]]): List of animal names to check for facing.
            If None, all animals are checked. Defaults to None.
        angle_threshold (float): Angle threshold for facing. Defaults to 45.
        n_frames (int): Number of frames to check for facing. Defaults to 1.
    """
    if for_animals is not None and animal.name not in for_animals:
        return False

    from_animals = from_animals or list(env.animals.keys())
    to_objects = to_objects or list(env.objects.keys())
    for from_animal in [env.animals[name] for name in from_animals]:
        for to_object in [env.objects[name] for name in to_objects]:
            vec = to_object.pos - from_animal.pos
            yaw = np.arctan2(animal.mat[1, 0], animal.mat[0, 0])
            relative_yaw = np.abs(np.arctan2(vec[1], vec[0]) - yaw)
            if relative_yaw < np.deg2rad(angle_threshold / 2):
                key = f"trunc_facing_obj_{to_object.name}_{from_animal.name}"
                info.setdefault(key, 0)
                info[key] += 1
                if info[key] > n_frames:
                    return True
    return False


def truncate_if_low_reward(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    info: Dict[str, Any],
    threshold: float,
) -> bool:
    """Truncates the episode if the reward is less than a certain threshold."""
    return env.cumulative_reward < threshold


def combined_truncation(
    env: MjCambrianEnv, animal: MjCambrianAnimal, info: Dict[str, Any], **truncation_fns
) -> bool:
    """Combines multiple truncation functions into one."""
    truncate = False
    for fn in truncation_fns.values():
        truncate |= fn(env, animal, info)
    return truncate
