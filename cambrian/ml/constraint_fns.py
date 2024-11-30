"""These are constraint functions for the optimizers. These functions are used to prune
experiments from the search space."""

from typing import Any, Dict, Tuple

import numpy as np

from cambrian.utils import is_number


def nevergrad_constraint_fn(
    parameterization: Dict[str, Any], /, *, fn: str, **parameters
) -> bool:
    """This function is used to prune experiments for nevergrad sweepers. It will
    return False if the experiment should be pruned."""
    from hydra.utils import get_method

    arguments: Dict[str, Any] = {}
    for argument_key, key_or_value in parameters.items():
        if isinstance(key_or_value, str) and key_or_value in parameterization:
            arguments[argument_key] = parameterization[key_or_value]
        else:
            arguments[argument_key] = key_or_value
    return get_method(fn)(**arguments)


def constrain_total_pixels(
    *,
    num_eyes_to_generate: Tuple[int, int] | int,
    resolution: Tuple[int, int] | int,
    max_num_pixels: int,
):
    """This constraint method will check whether the total number of pixels generated
    is less than a certain threshold."""
    if is_number(num_eyes_to_generate):
        num_eyes_to_generate = (1, num_eyes_to_generate)
    if is_number(resolution):
        resolution = (resolution, 1)
    pixels_per_eye = resolution[0] * resolution[1]
    number_of_eyes = num_eyes_to_generate[0] * num_eyes_to_generate[1]
    return pixels_per_eye * number_of_eyes <= max_num_pixels


def constrain_total_memory_throughput(
    *,
    num_eyes_to_generate: Tuple[int, int] | int,
    resolution: Tuple[int, int] | int,
    stack_size: int,
    max_pixels_in_memory: int,
):
    """This constraint method will check whether the total number of pixels generated
    is less than a certain threshold."""
    if is_number(num_eyes_to_generate):
        num_eyes_to_generate = (1, num_eyes_to_generate)
    if is_number(resolution):
        resolution = (resolution, 1)
    pixels_per_eye = resolution[0] * resolution[1]
    number_of_eyes = num_eyes_to_generate[0] * num_eyes_to_generate[1]
    return pixels_per_eye * number_of_eyes * stack_size <= max_pixels_in_memory


def constrain_morphologically_feasible_eyes(
    *,
    num_eyes_to_generate: int,
    resolution: Tuple[int, int] | int,
    lon_range: Tuple[int, int] | int,
    radius: float = 0.1,
    pixel_size: float = 5e-3,
    **_,
):
    """This constraint method will check whether the eye config, if placed
    num_eyes_to_generate along the longitude of the agent, would be
    morphologically feasible. Morphologically feasible in this approximated case is
    basically whether all the eyes would fit. There are two primary factors here:

    1. sensorsize and number of eyes. We want to make sure, along the horizontal axis,
    that the eyes don't overlap.

    2. The total number of pixels. We want to make sure that the total number of pixels
    generated is less than a certain threshold.

    Going to approximate the agent as a circle and the eyes as a line with a length
    equal to the sensorsize width. Then we'll check whether the eyes fit in the allowed
    longitude range.


    Args:
        num_eyes_to_generate (int): The number of eyes to generate along
            the longitude of the agent.
        resolution (Tuple[int, int] | int): The resolution of the eye.
        lon_range (Tuple[int, int] | int): The range of longitudes in which to generate
            the eyes. This is in degrees.

    Keyword Args:
        radius (float): The radius of the agent. Default is 0.2.
        pixel_size (float): The pixel size of the eye. This is used to calculate the
            total width of the eyes. Default is 0.01.
    """
    if is_number(num_eyes_to_generate):
        num_eyes_to_generate = (1, num_eyes_to_generate)
    if is_number(resolution):
        resolution = (resolution, 1)
    if is_number(lon_range):
        lon_range = (-abs(lon_range), abs(lon_range))

    # Total width of each eye
    sensor_width = resolution[0] * pixel_size
    total_width = sensor_width * num_eyes_to_generate[1]

    # Check whether the total width is less than the circumference of the agent
    # Only checked in the lon range
    lon_circumference = (lon_range[1] - lon_range[0]) * np.pi / 180 * radius
    lon_feasibility = total_width < lon_circumference

    return lon_feasibility


def constrain_total_num_eyes(
    *,
    num_eyes_to_generate: Tuple[int, int],
    max_num_eyes: int,
):
    """This constraint method will check whether the total number of eyes generated
    is less than a certain threshold."""
    return num_eyes_to_generate[0] * num_eyes_to_generate[1] <= max_num_eyes
