from typing import Any, Dict

from cambrian.envs.env import MjCambrianEnv
from cambrian.envs.object_env import MjCambrianObjectEnv
from cambrian.animals.animal import MjCambrianAnimal

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
    env: MjCambrianObjectEnv, animal: MjCambrianAnimal, info: Dict[str, Any]
) -> bool:
    """Truncates the episode if the animal is close to an object. Truncate is only
    true if the object is set to truncate_if_close = True."""
    for obj in env.objects.values():
        if obj.is_close(animal.pos) and obj.config.truncate_if_close:
            return True
    return False


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
