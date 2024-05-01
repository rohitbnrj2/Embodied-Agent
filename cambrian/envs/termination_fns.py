"""Termination indicates a success before or when the episode is over."""

from typing import Any, Dict, Optional, List

import numpy as np

from cambrian.envs import MjCambrianEnv
from cambrian.envs.object_env import MjCambrianObjectEnv
from cambrian.animals import MjCambrianAnimal

# =====================
# Termination functions


def never_terminates(
    env: MjCambrianEnv, animal: MjCambrianAnimal, info: Dict[str, Any]
) -> bool:
    """Never terminates the episode."""
    return False


def terminate_if_close_to_object(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    info: Dict[str, Any],
    *,
    objects: Optional[List[str]] = None,
    distance_threshold: float = 2.0
) -> bool:
    """Terminates the episode if the animal is close to an object.

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


def terminate_if_close_to_animal(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    info: Dict[str, Any],
    *,
    animal_name: str,
    distance_threshold: float
) -> bool:
    """Terminates the episode if the animal is close to another animal."""
    pos = animal.pos
    other_pos = env.animals[animal_name].pos
    return True if np.linalg.norm(pos - other_pos) < distance_threshold else False


def combined_termination(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    info: Dict[str, Any],
    **termination_fns
) -> bool:
    """Combines multiple termination functions into one."""
    terminate = False
    for fn in termination_fns.values():
        terminate |= fn(env, animal, info)
    return terminate
