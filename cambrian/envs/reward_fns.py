from typing import Dict, Any

import numpy as np

from cambrian.animal import MjCambrianAnimal

# =====================
# Reward functions


def euclidean_delta_from_init(
    self,
    animal: MjCambrianAnimal,
    info: Dict[str, Any],
) -> float:
    """
    Rewards the change in distance over the previous step scaled by the timestep.
    """
    return calc_delta(animal, info, animal.init_pos)


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
