from typing import Any, Dict

from cambrian.envs.env import MjCambrianEnv
from cambrian.envs.object_env import MjCambrianObjectEnv
from cambrian.animals.animal import MjCambrianAnimal

# =====================
# Termination functions


def never_terminates(
    env: MjCambrianEnv, animal: MjCambrianAnimal, info: Dict[str, Any]
) -> bool:
    """Never terminates the episode."""
    return False


def terminate_if_close_to_object(
    env: MjCambrianObjectEnv, animal: MjCambrianAnimal, info: Dict[str, Any]
) -> bool:
    """Terminates the episode if the animal is close to an object. Terminate is only
    true if the object is set to terminate_if_close = True."""
    for obj in env.objects.values():
        if obj.is_close(animal.pos) and obj.config.terminate_if_close:
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
