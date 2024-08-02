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
    pos = animal.pos
    for obj_name in objects if objects is not None else env.objects.keys():
        obj = env.objects[obj_name]
        if np.linalg.norm(obj.pos - pos) < distance_threshold:
            return True
    return False


def terminate_if_close_to_animal(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    info: Dict[str, Any],
    *,
    animals: Optional[List[str]] = None,
    distance_threshold: float
) -> bool:
    """Terminates the episode if the animal is close to another animal."""
    pos = animal.pos
    for animal_name in animals if animals is not None else env.animals.keys():
        if np.linalg.norm(env.animals[animal_name].pos - pos) < distance_threshold:
            return True
    return False


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
