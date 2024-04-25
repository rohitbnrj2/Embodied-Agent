from typing import Tuple

import numpy as np


def prune_anatomically_infeasible_eyes(
    *,
    num_lon_eyes_to_generate: int,
    width: int,
    lon_range: Tuple[float, float],
    radius: float = 0.1,
    pixel_size: float = 5e-3
) -> bool:
    """This method will check whether the eye config, if placed num_lon_eyes along
    the longitude of the animal, would be anatomically feasible. Anatomically feasible
    in this approximated case is basically whether all the eyes would fit. There
    are two primary factors here: sensorsize and number of eyes. We want to make sure,
    along the horizontal axis, that the eyes don't overlap.

    Going to approximate the animal as a circle and the eyes as a line with a length
    equal to the sensorsize width. Then we'll check whether the eyes fit in the allowed
    longitude range.

    NOTE: there isn't a specific unit, but the units should be consistent.

    Args:
        num_lon_eyes_to_generate (int): The number of eyes to generate along the longitude.
        width (int): The width of the eye. This is used to calculate the total width of
            the eyes. In pixels.
        lon_range (Tuple[float, float]): The longitude range to generate the eyes in.
            In degrees.

    Keyword Args:
        radius (float): The radius of the animal. Default is 0.2.
        pixel_size (float): The pixel size of the eye. This is used to calculate the
            total width of the eyes. Default is 0.01.
    """

    # Calculate the total width of the eyes
    sensor_width = width * pixel_size
    total_width = sensor_width * num_lon_eyes_to_generate

    # Check whether the total width is less than the circumference of the animal
    # Only checked in the lon range
    circumference = (lon_range[1] - lon_range[0]) * np.pi / 180 * radius
    return total_width < circumference
