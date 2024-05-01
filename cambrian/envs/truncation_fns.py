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
    distance_threshold: float = 2.0
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
    distance_threshold: float
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


def combined_truncation(
    env: MjCambrianEnv, animal: MjCambrianAnimal, info: Dict[str, Any], **truncation_fns
) -> bool:
    """Combines multiple truncation functions into one."""
    truncate = False
    for fn in truncation_fns.values():
        truncate |= fn(env, animal, info)
    return truncate
