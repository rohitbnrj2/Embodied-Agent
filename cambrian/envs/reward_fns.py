from typing import Dict, Any, List, Optional

import numpy as np
import mujoco as mj

from cambrian.envs import MjCambrianEnv
from cambrian.envs.object_env import MjCambrianObjectEnv
from cambrian.envs.maze_env import MjCambrianMazeEnv
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


reward_for_quick_termination = reward_for_termination


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


reward_for_quick_truncation = reward_for_truncation


def euclidean_delta_from_init(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    factor: float = 1.0,
    only_best: bool = False,
) -> float:
    """
    Rewards the change in distance over the previous step scaled by the timestep.

    `only_best` will only reward the animal has moved further from the initial position
    than any previous position. This requires us to keep around a state of the best
    position.
    """
    if only_best:
        # Only reward if the current position is further from the initial position than
        # any previous position. This requires us to keep around a state of the best
        # position. If the current position is closer, we'll skip this object.
        closest_pos = info.setdefault("best_pos", animal.init_pos.copy())
        if check_if_larger(animal.pos, closest_pos, animal.init_pos):
            info["best_pos"] = animal.pos.copy()
        else:
            return 0.0

    return calc_delta(animal, info, animal.init_pos) * factor


def reward_euclidean_delta_to_objects(
    env: MjCambrianMazeEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    factor: float,
    objects: Optional[List[str]] = None,
    only_best: bool = False,
    min_delta_threshold: Optional[float] = None,
    max_delta_threshold: Optional[float] = None,
):
    """
    Rewards the change in distance to any enabled object over the previous step.
    Convention is that a positive reward indicates getting closer to the object.

    `only_best` will only reward the animal if it is closer to the object than any
    previous position. This requires us to keep around a state of the best position.
    """
    enabled_objects = env.maze.config.enabled_objects

    accumulated_reward = 0.0
    for obj in env.objects.values():
        if objects is not None and obj.name not in objects:
            continue
        elif enabled_objects is not None and obj.name not in enabled_objects:
            continue

        if only_best:
            # Only reward if the current position is closer to the object than any
            # previous position. This requires us to keep around a state of the best
            # position. If the current position is further, we'll skip this object.
            closest_pos = info.setdefault(f"best_pos_{obj.name}", animal.pos.copy())
            if check_if_larger(obj.pos, closest_pos, animal.pos):
                info[f"best_pos_{obj.name}"] = animal.pos.copy()
            else:
                continue

        # NOTE: calc_delta returns a positive value if the animal moves away from the
        # object. We'll multiple by -1 to flip the convention.
        delta = -factor * calc_delta(animal, info, obj.pos)
        if min_delta_threshold is not None and delta < min_delta_threshold:
            continue
        elif max_delta_threshold is not None and delta > max_delta_threshold:
            continue

        accumulated_reward = delta

    return accumulated_reward


def reward_euclidean_delta_to_animals(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    factor: float,
    animals: Optional[List[str]] = None,
    only_best: bool = False,
    min_delta_threshold: Optional[float] = None,
    max_delta_threshold: Optional[float] = None,
):
    """
    Rewards the change in distance to any enabled animal over the previous step.
    Convention is that a positive reward indicates getting closer to the animal.

    `only_best` will only reward the animal if it is closer to the animal than any
    previous position. This requires us to keep around a state of the best position.
    """
    accumulated_reward = 0.0
    for other_animal in env.animals.values():
        if animals is not None and other_animal.name not in animals:
            continue

        if only_best:
            # Only reward if the current position is closer to the animal than any
            # previous position. This requires us to keep around a state of the best
            # position. If the current position is further, we'll skip this object.
            closest_pos = info.setdefault(
                f"best_pos_{other_animal.name}", animal.pos.copy()
            )
            if check_if_larger(other_animal.pos, closest_pos, animal.pos):
                info[f"best_pos_{other_animal.name}"] = animal.pos.copy()
            else:
                continue

        # NOTE: calc_delta returns a positive value if the animal moves away from the
        # object. We'll multiple by -1 to flip the convention.
        delta = -factor * calc_delta(animal, info, other_animal.pos)
        if min_delta_threshold is not None and delta < min_delta_threshold:
            continue
        elif max_delta_threshold is not None and delta > max_delta_threshold:
            continue

        accumulated_reward = delta

    return accumulated_reward


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
    animals: Optional[List[str]] = None,
) -> float:
    """Penalizes the animal if it has contacts with the ground."""
    if animals is not None and animal.name not in animals:
        return 0.0
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
    scale_by_distance: bool = False,
    hfov: Optional[float] = None,
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
        scale_by_distance (bool): Whether to scale the reward by the distance between
            the animals. Default is False.
        hfov (float): The horizontal fov to check whether the to object is within view
            of the from animal. This is in degrees. If unset, will check the horizontal
            fov for all eyes of the animal.
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
            in_view = False
            if hfov is None:
                for eye in from_animal.eyes.values():
                    hfov, _ = eye.config.fov
                    yaw = np.arctan2(eye.mat[1, 0], eye.mat[0, 0])
                    if check_in_view(
                        env.model,
                        env.data,
                        eye.pos,
                        yaw,
                        to_object.pos,
                        to_object.geomid,
                        hfov=hfov,
                        geomgroup_mask=animal.geomgroup_mask,
                    ):
                        in_view = True
                        break
            else:
                yaw = np.arctan2(animal.mat[1, 0], animal.mat[0, 0]) + np.pi / 2
                in_view = check_in_view(
                    env.model,
                    env.data,
                    animal.pos,
                    yaw,
                    to_object.pos,
                    to_object.geomid,
                    hfov=hfov,
                    geomgroup_mask=animal.geomgroup_mask,
                )

            # Add the reward to the accumulated reward. It may be scaled by the distance
            # if scale_by_distance is True, but only if the object is in view.
            dist = np.linalg.norm(to_object.pos - from_animal.pos)
            scale = 1 / max(dist, 1) if scale_by_distance else 1
            if in_view:
                reward = reward_in_view * scale
            else:
                reward = reward_not_in_view * scale
            accumulated_reward += reward

    return accumulated_reward


25


def reward_if_facing_objects(
    env: MjCambrianObjectEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward_facing: float = 0.0,
    reward_not_facing: float = 0.0,
    from_animals: Optional[List[str]] = None,
    to_objects: Optional[List[str]] = None,
    for_animals: Optional[List[str]] = None,
    angle_threshold: float = 45,
    scale_by_angle: bool = False,
) -> float:
    """This reward function rewards the animal for minimizing the angle between the
    animal's yaw and the yaw of the vector between itself and the object.

    Keyword Args:
        reward_facing (float): The reward to give the animal if it is facing the object.
            Default is 0.
        reward_not_facing (float): The reward to give the animal if it is not facing the
            object. Default is 0.
        from_animals (Optional[List[str]]): The names of the animals that the reward
            should be calculated from. If None, the reward will be calculated from all
            animals.
        to_objects (Optional[List[str]]): The names of the objects that the reward
            should be calculated to. If None, the reward will be calculated to all
            objects.
        for_animals (Optional[List[str]]): The names of the animals that the reward
            should be calculated for. If None, the reward will be calculated for all
            animals.
    """
    # Early exit if the animal is not in the from_animals list
    if for_animals is not None and animal.name not in for_animals:
        return 0

    accumulated_reward = 0
    from_animals = from_animals or list(env.animals.keys())
    to_objects = to_objects or list(env.objects.keys())
    for from_animal in [env.animals[name] for name in from_animals]:
        for to_object in [env.objects[name] for name in to_objects]:
            vec = to_object.pos - from_animal.pos
            yaw = np.arctan2(animal.mat[1, 0], animal.mat[0, 0])
            relative_yaw = np.abs(np.arctan2(vec[1], vec[0]) - yaw)

            # Add the reward to the accumulated reward. If the relative yaw is within
            # the fov of the animal, reward the animal. Otherwise, penalize it.
            if relative_yaw < np.deg2rad(angle_threshold / 2):
                reward = reward_facing
            else:
                reward = reward_not_facing
            if scale_by_angle:
                reward *= 1 - relative_yaw / np.pi
            accumulated_reward += reward

    return accumulated_reward


def constant_reward(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
) -> float:
    """Returns a constant reward."""
    return reward


def combined_reward(
    env: MjCambrianEnv,
    animal: MjCambrianAnimal,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    exclusive_fns: List[str] = [],
    **reward_fns,
) -> float:
    """Combines multiple reward functions into one.

    Keyword Args:
        exclusive_fns (Optional[List[str]]): If provided, only the reward functions
            with this name will be used if it's non-zero. As in, in order, the first
            function to return a non-zero reward will be returned.
    """
    accumulated_reward = 0
    for name, fn in reward_fns.items():
        reward = fn(env, animal, terminated, truncated, info)

        if name in exclusive_fns and reward != 0:
            return reward
        accumulated_reward += reward
    return accumulated_reward


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


def check_if_larger(
    p1: np.ndarray, p2: np.ndarray, point: np.ndarray = np.array([0, 0])
) -> bool:
    """Checks if the distance from point to p1 is larger than the distance from point
    to p2."""
    return np.linalg.norm(p1 - point) > np.linalg.norm(p2 - point)


def check_in_view(
    model: mj.MjModel,
    data: mj.MjData,
    from_pos: np.ndarray,
    from_yaw: float,
    to_pos: np.ndarray,
    to_geomid: int,
    hfov: float = 45,
    geomgroup_mask: int = 0,
) -> bool:
    """Checks if the to_pos is in the field of view of the animal."""
    vec = to_pos - from_pos
    relative_yaw = np.arctan2(vec[1], vec[0]) - from_yaw - np.pi / 2
    relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi

    # Early exit if the to_pos isn't within view of the from_pos
    if np.abs(relative_yaw) > np.deg2rad(hfov) / 2:
        return False

    # Now we'll trace a ray between the two points to check if there are any obstacles
    geomid = np.zeros(1, np.int32)
    mj.mj_ray(
        model,
        data,
        from_pos,
        vec,
        geomgroup_mask,  # can use to mask out animals to avoid self-collision
        1,  # include static geometries
        -1,  # include all bodies
        geomid,
    )

    # If the ray hit the to geom, then the to_pos is in view
    return geomid[0] == to_geomid
