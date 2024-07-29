from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from cambrian.utils import is_integer

from parse_types import (
    PlotData,
    SizeData,
    ColorData,
    CustomPlotFnType,
    SizeType,
)
from parse_helpers import get_size_label, get_color_label


def plot_helper(
    x_data: np.ndarray,
    y_data: np.ndarray,
    z_data: Optional[np.ndarray] = None,
    /,
    *,
    name: str,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    projection: str,
    dry_run: bool = False,
    **kwargs,
) -> plt.Figure | None:
    """This is a helper method that will be used to plot the data.

    NOTE: Saving the plots at the end depends on each plt figure having a unique name,
    which this method sets to the passed `title`.

    Keyword arguments:
        **kwargs -- Additional keyword arguments to pass to plt.plot.
    """
    if dry_run:
        return

    fig = plt.figure(name)
    if len(fig.get_axes()) == 0:
        fig.add_subplot(111, projection=projection)
    assert len(fig.get_axes()) == 1
    ax = fig.gca()
    if z_data:
        ax.scatter(x_data, y_data, z_data, **kwargs)
        ax.set_zlabel(zlabel)
    else:
        ax.scatter(x_data, y_data, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)

    return fig


def adjust_points(
    size_data: SizeData,
    ax: plt.Axes,
    sizes: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
):
    """This method will adjust the points depending on different conditions."""

    if size_data.type is SizeType.NUM:
        # In this case, update the size of the points to reflect the number of points
        # at the same location and set the color to the average of the colors that are
        # stacked together

        unique, counts = np.unique(points, axis=0, return_counts=True)
        for point, count in zip(unique, counts):
            if count <= 1:
                continue

            idx = np.where(np.all(points == point, axis=1))[0]

            for i in idx:
                ax.collections[i].set_sizes(ax.collections[i].get_sizes() * count)

            # Set the color to the average of the colors that are stacked together
            if colors is not None:
                color = np.mean(colors[idx], axis=0)
                # set that color to the largest size since we'll reorder the points
                ax.collections[sizes[idx].argmax()].set_array(color)

    # Reorder the points such that the largest points are in the front
    # Use sizes for the zorder unless the are all the same, then use color
    zorder_data = sizes if len(np.unique(sizes)) > 1 else colors
    if zorder_data is not None:
        for i in range(len(ax.collections)):
            ax.collections[i].set_zorder(zorder_data[i])

    sizes = sizes * size_data.factor
    if size_data.normalize and sizes.min() != sizes.max():
        # Normalize the sizes
        sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())
        sizes = size_data.size_min + sizes * (size_data.size_max - size_data.size_min)

    for scatter, size in zip(ax.collections, sizes):
        scatter.set_sizes(size)


def adjust_axes(ax: plt.Axes, plot_data: PlotData):
    if plot_data.x_data.lim is not None:
        ax.set_xlim(plot_data.x_data.lim)
    if plot_data.x_data.ticks is not None:
        ax.set_xticks(plot_data.x_data.ticks)
    if plot_data.x_data.tick_labels is not None:
        ax.set_xticklabels(plot_data.x_data.tick_labels)

    if plot_data.x_data.thetamin is not None:
        ax.set_thetamin(plot_data.x_data.thetamin)
    if plot_data.x_data.thetamax is not None:
        ax.set_thetamax(plot_data.x_data.thetamax)
    if plot_data.x_data.rmin is not None:
        ax.set_rmin(plot_data.x_data.rmin)
    if plot_data.x_data.rmax is not None:
        ax.set_rmax(plot_data.x_data.rmax)

    if plot_data.y_data.lim is not None:
        ax.set_ylim(plot_data.y_data.lim)
    if plot_data.y_data.ticks is not None:
        ax.set_yticks(plot_data.y_data.ticks)
    if plot_data.y_data.tick_labels is not None:
        ax.set_yticklabels(plot_data.y_data.tick_labels)
    
    if plot_data.y_data.thetamin is not None:
        ax.set_thetamin(plot_data.y_data.thetamin)
    if plot_data.y_data.thetamax is not None:
        ax.set_thetamax(plot_data.y_data.thetamax)
    if plot_data.y_data.rmin is not None:
        ax.set_rmin(plot_data.y_data.rmin)
    if plot_data.y_data.rmax is not None:
        ax.set_rmax(plot_data.y_data.rmax)

    if plot_data.z_data is not None:
        if plot_data.z_data.lim is not None:
            ax.set_zlim(plot_data.z_data.lim)
        if plot_data.z_data.ticks is not None:
            ax.set_zticks(plot_data.z_data.ticks)
        if plot_data.z_data.tick_labels is not None:
            ax.set_zticklabels(plot_data.z_data.tick_labels)


def run_custom_fns(ax: plt.Axes, plot_data: PlotData):
    for custom_fn in plot_data.custom_fns:
        if custom_fn.type is CustomPlotFnType.GLOBAL:
            custom_fn.fn(ax)


def add_legend(
    ax: plt.Axes,
    plot_data: PlotData,
    points: np.ndarray,
    sizes: np.ndarray,
    colors: np.ndarray,
    **kwargs,
):
    """Add a legend to the plot. The legend title will say 'Selected Agent' and if the
    sizes are different between the points, the legend will show 3 examples
    (min, mean, max)."""

    size_data = plot_data.size_data
    label = get_size_label(size_data)
    if plot_data.size_data.normalize and sizes.min() != sizes.max():
        # The sizes are really values, and they are scaled separately in adjust_points
        # We'll rescale the sizes again here for the legend, and set the label to
        # reflect the original value.
        original_sizes = sizes.copy()
        sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())
        sizes = size_data.size_min + sizes * (size_data.size_max - size_data.size_min)
        sizes /= size_data.size_min / 2  # TODO: Why do we have to do this line?

        def calc_value(s):
            """Get's the original value from the size"""
            s = s * size_data.size_min / 2
            norm_value = (s - size_data.size_min) / (
                size_data.size_max - size_data.size_min
            )
            unscaled_value = (
                norm_value * (original_sizes.max() - original_sizes.min())
                + original_sizes.min()
            )
            return unscaled_value

        # If the number of unique pts is less than 3, we'll only show the min and max
        # values. Otherwise we'll show the min, mean, and max values.
        if len(np.unique(sizes)) < 3:
            num = [int(calc_value(s)) for s in [sizes.max(), sizes.min()]]
        else:
            num = [
                int(calc_value(s))
                for s in [sizes.max(), (sizes.max() - sizes.min()) / 2, sizes.min()]
            ]

        scatter = ax.scatter(*points.T, s=sizes, c=colors)
        legend_items = scatter.legend_elements(prop="sizes", num=num, func=calc_value)
        ax.legend(*legend_items, title=label, **kwargs)

        scatter.remove()  # remove the scatter plot so it doesn't show up in the plot
    else:
        # Ensure all lagend items are unique
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        if unique_labels:
            legend = ax.legend(unique_labels.values(), unique_labels.keys(), **kwargs)
            # Explicitly add the legend to the axis to ensure it stays on the plot
            ax.add_artist(legend)


def add_colorbar(
    color_data: ColorData | None,
    ax: plt.Axes,
    colors: np.ndarray | None,
    *,
    only_unique: bool = True,
):
    """Normalize the colors and add a colorbar to the plot."""
    # Don't add a colorbar if there is only one color
    if colors is None or (len(np.unique(colors)) == 1 and only_unique):
        return

    # Normalize the colorbar first
    vmin, vmax = (
        color_data.clim
        if color_data.clim is not None
        else (np.min(colors), np.max(colors))
    )
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = ax.collections[0].get_cmap()
    if color_data is not None:
        if color_data.cmap is not None:
            cmap = color_data.cmap
        if color_data.start_color is not None and color_data.end_color is not None:
            cmap = LinearSegmentedColormap.from_list(
                "custom", [color_data.start_color, color_data.end_color]
            )
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([vmin, vmax])

    label = get_color_label(color_data)
    cbar = plt.colorbar(sm, ax=ax, label=label, **color_data.kwargs)

    # Update the colors to the colormap
    for scatter in ax.collections:
        scatter.set_cmap(cmap)
        scatter.set_norm(norm)

    # Set the colorbar to int if the colors are integers
    if is_integer(colors):
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
