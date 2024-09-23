from typing import Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

from cambrian.utils.logger import get_logger

from utils import extract_data
from parse_types import Rank, PlotData
from parse_helpers import parse_plot_data


def accumulation_line(ax: plt.Axes, accumulation_fn: Callable[[np.ndarray], float]):
    """Extracts the data from a figure and plots the accumulation line along with
    the standard deviation. NOTE: does not support 3d"""

    # Extract the data from the plot
    x_data, y_data, z_data = extract_data(ax, return_data=True)
    assert z_data is None, "Accumulation line does not support 3d plots."

    # Calculates the accumulation y value for each unique x value
    x, y, y_std = [], [], []
    for unique_x in np.unique(x_data):
        x.append(unique_x)
        y.append(accumulation_fn(y_data[x_data == unique_x]))
        y_std.append(np.std(y_data[x_data == unique_x]))
    x, y, y_std = np.array(x), np.array(y), np.array(y_std)

    # Plot the data
    ax.plot(x, y, "C0-", alpha=0.5)
    ax.fill_between(x, y - y_std, y + y_std, alpha=0.2, facecolor="C0")


def num_eyes_and_resolution_constraint(
    ax: plt.Axes,
    *,
    lon_range: Tuple[float, float],
    radius: float = 0.1,
    pixel_size: float = 5e-3,
    clip_line: bool = True,
):
    # Extract the data from the plot
    # Assumes num_eyes is the x axis and resolution is the y axis
    num_eyes, resolution, _ = extract_data(ax, return_data=True)
    num_eyes = np.linspace(num_eyes.min(), num_eyes.max(), 1000)

    # Calculate the maximum feasible resolution for each number of eyes
    circumference = (lon_range[1] - lon_range[0]) * np.pi / 180 * radius
    max_feasible_resolution = circumference / (num_eyes * pixel_size)

    if clip_line:
        # Clip the max feasible resolution to the maximum resolution
        # Delete all the resolutions/num eyes at that max value so we don't have a flat
        # line at the top
        mask = max_feasible_resolution <= np.max(resolution)
        num_eyes = num_eyes[mask]
        max_feasible_resolution = max_feasible_resolution[mask]

    # Plot the constraint as a red curve
    ax.plot(num_eyes, max_feasible_resolution, "r-", label="Morophological Constraint")


def num_eyes_and_resolution_pixels_constraint(
    ax: plt.Axes,
    *,
    max_total_pixels: int,
    clip_line: bool = True,
):
    # Extract the data from the plot
    # Assumes num_eyes is the x axis and resolution is the y axis
    num_eyes, resolution, _ = extract_data(ax, return_data=True)
    num_eyes = np.linspace(num_eyes.min(), num_eyes.max(), 1000)

    # Calculate the maximum feasible resolution for each number of eyes
    max_feasible_resolution = max_total_pixels / num_eyes

    if clip_line:
        # Clip the max feasible resolution to the maximum resolution
        # Delete all the resolutions/num eyes at that max value so we don't have a flat
        # line at the top
        mask = max_feasible_resolution <= np.max(resolution)
        num_eyes = num_eyes[mask]
        max_feasible_resolution = max_feasible_resolution[mask]

    # Plot the constraint as a red curve
    ax.plot(num_eyes, max_feasible_resolution, "r-", label="Morophological Constraint")


def connect_with_parent(ax: plt.Axes, plot_data: PlotData, rank_data: Rank, **kwargs):
    """This custom plot fn is called for each rank and plots a line between itself and
    it's parent. No line is plotted if the rank doesn't have a parent."""

    if (parent := rank_data.parent) is None:
        return

    try:
        x_data, y_data, z_data, _, _ = parse_plot_data(
            plot_data, rank_data.generation.data, rank_data.generation, rank_data
        )
        x_data_parent, y_data_parent, z_data_parent, _, _ = parse_plot_data(
            plot_data, parent.generation.data, parent.generation, parent
        )
    except AssertionError:
        get_logger().warning(f"Failed to load data for rank {rank_data.num}")

    # We'll plot a line between the current agent's fitness and the parent's
    if z_data is None:
        ax.plot(
            [x_data[0], x_data_parent[0]],
            [y_data[0], y_data_parent[0]],
            "k-",
            alpha=0.5,
            zorder=0,
        )
    else:
        ax.plot(
            [x_data[0], x_data_parent[0]],
            [y_data[0], y_data_parent[0]],
            [z_data[0], z_data_parent[0]],
            "k-",
            alpha=0.5,
            zorder=0,
        )
