from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from utils import extract_data

def average_line(ax: plt.Axes):
    """Extracts the data from a figure and plots the average line along with
    the standard deviation. NOTE: does not support 3d"""

    # Extract the data from the plot
    x_data, y_data, z_data = extract_data(ax, return_data=True)
    assert z_data is None, "Average line does not support 3d plots."

    # Calculates the average y value for each unique x value
    x, y, y_std = [], [], []
    for unique_x in np.unique(x_data):
        x.append(unique_x)
        y.append(np.average(y_data[x_data == unique_x]))
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
):
    # Extract the data from the plot
    # Assumes num_eyes is the x axis and resolution is the y axis
    num_eyes, resolution, _ = extract_data(ax, return_data=True)
    num_eyes = np.linspace(num_eyes.min(), num_eyes.max(), 1000)

    # Calculate the maximum feasible resolution for each number of eyes
    circumference = (lon_range[1] - lon_range[0]) * np.pi / 180 * radius
    max_feasible_resolution = circumference / (num_eyes * pixel_size)

    # Clip the max feasible resolution to the maximum resolution
    # Delete all the resolutions/num eyes at that max value so we don't have a flat
    # line at the top
    mask = max_feasible_resolution <= np.max(resolution)
    num_eyes = num_eyes[mask]
    max_feasible_resolution = max_feasible_resolution[mask]

    # Plot the constraint as a red curve
    ax.plot(num_eyes, max_feasible_resolution, "r-", label="Morophological Constraint")