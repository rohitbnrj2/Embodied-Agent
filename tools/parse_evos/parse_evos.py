"""This script will a saved evolution folder.

This file works as follows:
1. Parse the evolution folder. The evolution folder consists of many generations, each
    with many ranks. Parsing involves walking through the folder hierarchy and loading
    each config, evaluations, monitor and/or other available data.
2. Save the parsed data to a pickle file for easy loading. Walking through the folder
    hierarchy can be slow, so this is useful for quick loading. NOTE: If a pickle file
    already exists, it will be loaded instead of parsing the data again (unless
    `--force` is passed).
3. Plot the parsed data. This involves plotting the evaluations, monitor and/or other
    available data. This is useful for visualizing the evolution of the population.
"""

from typing import Dict, Union, Optional, Any, List, Tuple, TypeAlias, Callable, Self
from pathlib import Path
import os
from dataclasses import field
from enum import Enum

import cloudpickle as pickle
from omegaconf import OmegaConf
import tqdm.rich as tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from stable_baselines3.common.results_plotter import load_results, ts2xy
import hydra
import hydra.conf

from cambrian.utils import calculate_fitness, is_integer
from cambrian.utils.logger import get_logger
from cambrian.utils.config import (
    MjCambrianBaseConfig,
    MjCambrianConfig,
    run_hydra,
    config_wrapper,
)

# =======================================================

Color: TypeAlias = List[Tuple[float, float, float, float]] | List[float]

ParsedAxisData: TypeAlias = Tuple[np.ndarray | float | int, str]
ParsedColorData: TypeAlias = Tuple[Color, str]
ParsedPlotData: TypeAlias = Tuple[
    ParsedAxisData, ParsedAxisData, ParsedAxisData | None, Color
]
ExtractedData: TypeAlias = (
    np.ndarray | None
    | Tuple[np.ndarray, np.ndarray, np.ndarray | None]
    | Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
)

# =======================================================
# Dataclasses


@config_wrapper
class Rank:
    """A rank is a single run of an inner training loop. Like in a generation, you have
    many ranks which in themselves have many parallel environments to speed up training.
    The terminology comes from MPI.
    """

    path: Path

    num: int

    config: Optional[MjCambrianConfig] = None
    evaluations: Optional[Dict[str, Any]] = None
    fitness: Optional[float] = None
    monitor: Optional[Dict[str, Any]] = None

    # Defauls to True
    ignored: bool = True


@config_wrapper
class Generation:
    """A generation is a collection of ranks. Throughout evolution (outer loop), you
    have many generations where each generation consists of many ranks which
    train (inner loop) in parallel.
    """

    path: Path

    num: int
    ranks: Dict[int, Rank] = field(default_factory=dict)

    # Defaults to True
    ignored: bool = True


@config_wrapper
class Data:
    """This is the primary data storage class. It contains all the generations and ranks
    and is used to store the parsed data. It also can accumulated arbitrary data
    which is used for plotting."""

    path: Path

    generations: Dict[int, Generation]

    accumulated_data: Dict[str, Any] = field(default_factory=dict)


class AxisDataType(Enum):
    GENERATION = "generation"
    CONFIG = "config"
    MONITOR = "monitor"
    WALLTIME = "walltime"
    EVALUATION = "evaluation"
    CUSTOM = "custom"


@config_wrapper
class AxisData:
    type: AxisDataType

    label: Optional[str] = None

    # CONFIG
    pattern: Optional[str] = None

    # MONITOR and WALLTIME
    window: Optional[int] = None

    # CUSTOM
    custom_fn: Optional[Callable[[Self, Data, Generation, Rank], ParsedAxisData]] = None


class ColorType(Enum):
    SOLID = "solid"
    GENERATION = "generation"
    RANK = "rank"
    MONITOR = "monitor"


@config_wrapper
class ColorData:
    type: ColorType

    label: Optional[str] = None

    # SOLID
    color: Optional[Tuple[float, float, float, float]] = None

    # GENERATION or RANK
    cmap: Optional[str] = None
    start_color: Optional[Tuple[float, float, float, float]] = None
    end_color: Optional[Tuple[float, float, float, float]] = None


@config_wrapper
class PlotData:
    xdata: AxisData
    ydata: AxisData
    zdata: Optional[AxisData] = None

    title: Optional[str] = None
    color_data: Optional[ColorData] = None
    use_average_line: Optional[bool] = None


@config_wrapper
class ParseEvosConfig(MjCambrianBaseConfig):
    """Config for the parse_evos script.

    Attributes:
        folder (Path): The folder to parse.
        output (Path): The output folder.
        plots_folder (Path): The folder to save the plots.
        evals_folder (Path): The folder to save the evaluations.
        hydra_folder (Optional[Path]): The sweep folder where the hydra config is saved.
            If specified, additional plots or plot features may be enabled.

        force (bool): Force loading of the data. If not passed, this script will try to
            find a parse_evos.pkl file and load that instead.
        no_save (bool): Do not save the parsed data.
        no_check_finished (bool): Don't check if a file called `finished` has been
            written.
        quiet (bool): Quiet mode. Set's the logger to warning.
        debug (bool): Debug mode. Set's the logger to debug and disables tqdm.

        ranks (Optional[List[int]]): The rank to use. If not passed, all ranks are
            used.
        generations (Optional[List[int]]): The generation to use. If not passed, all
            are used.

        plot (bool): Plot the data.
        eval (bool): Evaluate the data.

        dry_run (bool): Dry run.

        plots (Dict[str, PlotData]): The plots to create.
        overrides (List[str]): Overrides for the config.
        hydra_config (Optional[hydra.conf.HydraConf]): The hydra config.
    """

    folder: Path
    output: Path
    plots_folder: Path
    evals_folder: Path
    hydra_folder: Optional[Path] = None

    force: bool
    no_save: bool
    no_check_finished: bool
    quiet: bool
    debug: bool

    ranks: Optional[List[int]] = None
    generations: Optional[List[int]] = None

    plot: bool
    eval: bool

    dry_run: bool

    plots: Dict[str, PlotData]
    overrides: List[str]
    hydra_config: Optional[Dict[str, Any]] = None


# =======================================================
# Data loaders


def save_data(config: ParseEvosConfig, generations: Dict[int, Generation]):
    """Save the parsed data to a pickle file."""
    pickle_file = config.output / "data.pkl"
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_file, "wb") as f:
        pickle.dump(generations, f)
    get_logger().info(f"Saved parsed data to {pickle_file}.")


def try_load_pickle_data(folder: Path) -> Union[None, Dict]:
    """Try to load the data from the pickle file."""
    pickle_file = folder / "parse_evos" / "data.pkl"
    if pickle_file.exists():
        get_logger().info(f"Loading parsed data from {pickle_file}...")
        with open(pickle_file, "rb") as f:
            generations = pickle.load(f)
        get_logger().info(f"Loaded parsed data from {pickle_file}.")
        return generations

    get_logger().warning(f"Could not load {pickle_file}.")
    return None


def get_generation_file_paths(folder: Path) -> Data:
    """Create the initial storage dict for parsing and get the paths to all the
    generation/rank folders.

    The generation data is stored as follows:
    - <exp_name>
        - generation_0
            - rank_0
            - rank_1
            - ...
        - generation_1
            - rank_0
            - rank_1
            - ...
        - ...

    This function will parse the folder heirarchy as a preprocessing step and returns
    the file paths to all the generation/rank folders.
    """

    generations = dict()
    for root, dirs, files in os.walk(folder):
        root = Path(root)
        # Only parse the generation folders
        if not root.stem.startswith("generation_"):
            continue

        # Grab the generation number
        generation = int(root.stem.split("generation_", 1)[1])

        ranks = dict()
        for dir in dirs:
            dir = Path(dir)
            # Only parse the rank folders
            if not dir.stem.startswith("rank_"):
                continue

            # Grab the rank number
            rank = int(dir.stem.split("rank_", 1)[1])
            ranks[rank] = Rank(path=root / dir, num=rank)

        # Sort the ranks by rank number
        ranks = dict(sorted(ranks.items()))
        generations[generation] = Generation(path=root, num=generation, ranks=ranks)

    # Sort the generations by generation number
    generations = dict(sorted(generations.items()))
    return Data(path=folder, generations=generations)


def load_data(config: ParseEvosConfig) -> Data:
    """Load the data from the generation/rank folders. This function will walk through
    the folder heirarchy and load the config, evaluations and monitor data for each
    rank."""
    get_logger().info(f"Loading data from {config.folder}...")

    # Get the file paths to all the generation/rank folders
    data = get_generation_file_paths(config.folder)

    # Convert the overrides to a list
    overrides = config.overrides.to_container()

    # Walk through each generation/rank and parse the config, evaluations and monitor
    # data
    for generation, generation_data in data.generations.items():
        # Only load the generation we want, if specified
        if config.generations is not None and generation not in config.generations:
            continue

        get_logger().info(f"Loading generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            # Only load the rank we want, if specified
            if config.ranks is not None and rank not in config.ranks:
                continue

            get_logger().info(f"\tLoading rank {rank}...")

            # Check if the `finished` file exists.
            # If not, don't load the data. Sorry for the double negative.
            if (
                not (rank_data.path / "finished").exists()
                and not config.no_check_finished
            ):
                get_logger().warning(
                    f"\t\tSkipping rank {rank} because it is not finished."
                )
                continue

            # Get the config file
            if (config_file := rank_data.path / "config.yaml").exists():
                get_logger().debug(f"\tLoading config from {config_file}...")
                rank_data.config = MjCambrianConfig.load(config_file, instantiate=False)
                rank_data.config.merge_with_dotlist(overrides)
                rank_data.config.resolve()

            # Get the evaluations file
            evaluations_file = rank_data.path / "evaluations.npz"
            if evaluations_file.exists():
                rank_data.fitness = calculate_fitness(evaluations_file)
                with np.load(evaluations_file) as eval_data:
                    rank_data.evaluations = {k: eval_data[k] for k in eval_data.files}

            # Get the monitor file
            monitor_file = rank_data.path / "monitor.csv"
            if monitor_file.exists():
                rank_data.monitor = load_results(rank_data.path)

            # Set ignored to False
            rank_data.ignored = False

        # Set ignored to False
        generation_data.ignored = False

    return data


def get_axis_label(axis_data: AxisData, label: str = "") -> str:
    if axis_data.type is AxisDataType.GENERATION:
        label = "Generation"
    elif axis_data.type is AxisDataType.CONFIG:
        label = axis_data.label or axis_data.pattern
    elif axis_data.type is AxisDataType.MONITOR:
        label = "Training Reward"
    elif axis_data.type is AxisDataType.WALLTIME:
        label = "Walltime (minutes)"
    elif axis_data.type is AxisDataType.EVALUATION:
        label = "Fitness"
    return axis_data.label or label


def parse_axis_data(
    axis_data: AxisData, data: Data, generation_data: Generation, rank_data: Rank
) -> ParsedAxisData:
    data: Any
    if axis_data.type is AxisDataType.GENERATION:
        data = generation_data.num
    elif axis_data.type is AxisDataType.CONFIG:
        assert rank_data.config is not None, "Config is required for CONFIG."
        data = rank_data.config.glob(axis_data.pattern, flatten=True)
        pattern = axis_data.pattern.split(".")[-1]
        assert pattern in data, f"Pattern {axis_data.pattern} not found."
        data = data[pattern]
        if isinstance(data, list):
            data = np.average(data)
    elif axis_data.type is AxisDataType.MONITOR:
        assert rank_data.monitor is not None, "Monitor is required."
        # Grab the data sorted by timesteps
        _, y = ts2xy(rank_data.monitor, "timesteps")
        if axis_data.window is not None:
            y = moving_average(y, axis_data.window)
        data = y[-1]
    elif axis_data.type is AxisDataType.WALLTIME:
        assert rank_data.monitor is not None, "Monitor is required."
        t = ts2xy(rank_data.monitor, "walltime_hrs")[0] * 60  # convert to minutes
        window = axis_data.window or 100
        if len(t) < window:
            t = moving_average(t, axis_data.window)
        data = t[-1]
    elif axis_data.type is AxisDataType.EVALUATION:
        assert rank_data.fitness is not None, "Evaluations is required."
        data = rank_data.fitness
    elif axis_data.type is AxisDataType.CUSTOM:
        assert axis_data.custom_fn is not None, "Custom function is required."
        return axis_data.custom_fn(axis_data, data, generation_data, rank_data)
    else:
        raise ValueError(f"Unknown data type {axis_data.type}.")

    return data, get_axis_label(axis_data)


def get_color_label(color_data: ColorData | None) -> str:
    label: str
    if color_data is None or color_data.type is ColorType.SOLID:
        label = "Color"
    elif color_data.type is ColorType.GENERATION:
        label = "Generation"
    elif color_data.type is ColorType.RANK:
        label = "Rank"
    elif color_data.type is ColorType.MONITOR:
        label = "Training Reward"
    else:
        raise ValueError(f"Unknown color type {color_data.type}.")
    return color_data.label or label if color_data is not None else label


def parse_color_data(
    color_data: ColorData, data: Data, generation_data: Generation, rank_data: Rank
) -> ParsedColorData:
    color: Color
    if color_data is None:
        # Solid blue
        color = (0.65490196, 0.78039216, 0.90588235, 0.75)
        # Early return if no color data is provided
        return [color], "Color"
    elif color_data.type is ColorType.SOLID:
        assert color_data.color is not None, "Color is required for solid color."
        color = color_data.color
    elif color_data.type is ColorType.GENERATION:
        assert color_data.cmap is not None or (
            color_data.start_color is not None and color_data.end_color is not None
        ), "Cmap or start_color and end_color is required."

        # NOTE: The cmap is generated later and is normalized to the number of
        # generations.
        color = generation_data.num
    elif color_data.type is ColorType.RANK:
        assert color_data.cmap is not None or (
            color_data.start_color is not None and color_data.end_color is not None
        ), "Cmap or start_color and end_color is required."

        # NOTE: The cmap is generated later and is normalized to the number of ranks.
        color = rank_data.num
    elif color_data.type is ColorType.MONITOR:
        # NOTE: The color is set to the last value of the monitor data and is normalized
        # later relative to the reward of the other "colors".
        assert rank_data.monitor is not None, "Monitor is required."
        _, y = ts2xy(rank_data.monitor, "timesteps")
        y = moving_average(y, 100)
        color = y[-1]
    else:
        raise ValueError(f"Unknown color type {color_data.type}.")

    # Convert the color to a list if it's a single value
    if isinstance(color, (int, float)):
        color = [color]

    return color, get_color_label(color_data)


def parse_plot_data(
    plot_data: PlotData, data: Data, generation_data: Generation, rank_data: Rank
) -> ParsedPlotData:
    """Parses the plot data object and grabs the relevant x and y data."""

    xdata = parse_axis_data(plot_data.xdata, data, generation_data, rank_data)
    ydata = parse_axis_data(plot_data.ydata, data, generation_data, rank_data)
    zdata = (
        parse_axis_data(plot_data.zdata, data, generation_data, rank_data)
        if plot_data.zdata
        else None
    )
    color = parse_color_data(plot_data.color_data, data, generation_data, rank_data)
    return xdata, ydata, zdata, color


def extract_data(
    ax: plt.Axes, *, return_data: bool = False, return_color: bool = False
) -> ExtractedData:
    """Extracts the data from a figure."""
    assert return_data or return_color, "Must return either data or color."

    if return_data:
        x_data, y_data, z_data = [], [], []
        for collection in ax.collections:
            offset: np.ndarray = collection.get_offsets()
            if offset.shape[-1] == 2:
                x, y = offset.T
                x_data.append(x)
                y_data.append(y)
            else:
                x, y, z = offset.T
                x_data.append(x)
                y_data.append(y)
                z_data.append(z)
        x_data, y_data = np.array(x_data), np.array(y_data)
        z_data = None if not z_data else np.array(z_data)

    if return_color:
        c_data = []
        for collection in ax.collections:
            c = collection.get_array()
            c_data.append(c)
        c_data = np.array(c_data)

        # All the colors are None if the color is set to a solid color (i.e. SOLID)
        # We'll return None in that case
        if np.all([c is None for c in c_data]):
            c_data = None

    if return_data and return_color:
        return x_data, y_data, z_data, c_data
    elif return_data:
        return x_data, y_data, z_data
    else:
        return c_data


# =======================================================
# Plotters


def plot_helper(
    xdata: ParsedAxisData,
    ydata: ParsedAxisData,
    zdata: Optional[ParsedAxisData] = None,
    /,
    *,
    name: str,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
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
    if zdata:
        ax: Axes3D = fig.add_subplot(111, projection="3d")
        ax.scatter(xdata, ydata, zdata, **kwargs)
        ax.set_zlabel(zlabel)
    else:
        ax = fig.gca()
        ax.scatter(xdata, ydata, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)

    return fig


def adjust_points(
    ax: plt.Axes, xdata: np.ndarray, ydata: np.ndarray, zdata: np.ndarray | None, cdata: np.ndarray | None,
):
    """This method will adjust the points depending on different conditions.

    Current conditions:
        - non-unique x,y,z values: If non-unique points are detected, they are resized
            proportionally to the number of points at that location.
    """

    if zdata is not None:
        data = np.column_stack((xdata, ydata, zdata))
    else:
        data = np.column_stack((xdata, ydata))

    # Check for non-unique points
    unique, counts = np.unique(data, axis=0, return_counts=True)
    for point, count in zip(unique, counts):
        if count > 1:
            idx = np.where(np.all(data == point, axis=1))[0]

            for i in idx:
                ax.collections[i].set_sizes(ax.collections[i].get_sizes() * count)

            # Set the color to the average of the colors that are stacked together
            if cdata is not None:
                color = np.mean(cdata[idx], axis=0)
                ax.collections[idx[-1]].set_array(color) # the last point is in front


def add_legend(ax: plt.Axes):
    """Add a legend to the plot."""
    (selected_agent,) = ax.plot([], [], ".", color="k", label="Selected Agent")
    ax.legend(handles=[selected_agent])


def add_colorbar(
    color_data: ColorData | None,
    ax: plt.Axes,
    colors: np.ndarray | None,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    only_unique: bool = True,
):
    """Normalize the colors and add a colorbar to the plot."""
    # Don't add a colorbar if there is only one color
    if colors is None or (len(np.unique(colors)) == 1 and only_unique):
        return

    # Normalize the colorbar first
    vmin = vmin or np.min(colors)
    vmax = vmax or np.max(colors)
    norm = Normalize(vmin=vmin, vmax=vmax)
    for scatter in ax.collections:
        scatter.set_norm(norm)

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
    cbar = plt.colorbar(sm, ax=ax, label=label)

    # Set the colorbar to int if the colors are integers
    if is_integer(colors):
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_average_line(ax: plt.Axes):
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
    plt.plot(x, y, "C0-", alpha=0.5)
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.2, facecolor="C0")

def plot_constraints(ax: plt.Axes, hydra_config: hydra.conf.HydraConf):
    print(hydra_config.sweeper.parameterization)


def run_plot(config: ParseEvosConfig, data: Data):
    get_logger().info("Plotting data...")

    # Update the rcparams.figure.max_open_warning to suppress the warning
    get_logger().debug(f"Setting max open warning to {len(config.plots)}.")
    plt.rcParams["figure.max_open_warning"] = len(config.plots)

    for generation, generation_data in data.generations.items():
        # Only plot the generation we want, if specified
        if generation_data.ignored:
            continue
        elif config.generations is not None and generation not in config.generations:
            continue

        get_logger().info(f"Plotting generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            # Only plot the rank we want, if specified
            if rank_data.ignored:
                continue
            elif config.ranks is not None and rank not in config.ranks:
                continue

            get_logger().info(f"\tPlotting rank {rank}...")

            for plot_name, plot in config.plots.items():
                try:
                    xdata, ydata, zdata, cdata = parse_plot_data(
                        plot, data, generation_data, rank_data
                    )
                except AssertionError as e:
                    if rank_data.ignored:
                        title = f" {plot.title}" if plot.title else ""
                        get_logger().debug(f"Skipping plot{title}: {e}")
                        continue
                    raise ValueError(f"Error parsing plot {plot_name}: {e}")

                xdata, xlabel = xdata
                ydata, ylabel = ydata
                zdata, zlabel = zdata or (None, None)
                color, _ = cdata

                default_title = f"{ylabel} vs {xlabel}"
                if zlabel:
                    default_title = f"{zlabel} vs {default_title}"
                title = plot.title or default_title

                # Plot the data
                get_logger().debug(f"\t\tPlotting {title}...")

                plot_helper(
                    xdata,
                    ydata,
                    zdata,
                    name=plot_name,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    zlabel=zlabel,
                    marker=".",
                    c=color,
                    dry_run=config.dry_run,
                )

    # Now save the plots
    if not config.dry_run:
        fignums = plt.get_fignums()
        progress_bar = tqdm.tqdm(
            zip(config.plots.items(), fignums),
            total=len(fignums),
            desc="Saving plots...",
            disable=config.debug,
        )
        for (plot_name, plot_data), fig in progress_bar:
            fig = plt.figure(fig)
            ax = fig.gca()

            try:
                # Extract the data from the plot
                xdata, ydata, zdata, cdata = extract_data(
                    ax, return_data=True, return_color=True
                )
                is_3d = zdata is not None

                # We'll ignore any plots which don't have unique data along any axis
                # These plots aren't really useful as there is no independent variable.
                if np.all(xdata == xdata[0]):
                    get_logger().debug(f"Skipping plot {plot_name}: no unique xdata.")
                    continue
                elif np.all(ydata == ydata[0]):
                    get_logger().debug(f"Skipping plot {plot_name}: no unique ydata.")
                    continue
                elif is_3d and np.all(zdata == zdata[0]):
                    get_logger().debug(f"Skipping plot {plot_name}: no unique zdata.")
                    continue

                # Adjust the points, if necessary
                adjust_points(ax, xdata, ydata, zdata, cdata)

                # Plot the average line
                if plot_data.use_average_line:
                    plot_average_line(ax)

                # Add a legend entry for the blue circles
                # add_legend(ax)

                # Normalize the colorbar
                add_colorbar(plot_data.color_data, ax, cdata)

                # If all the values of an axis are integers, set the axis to integer
                if is_integer(xdata):
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if is_integer(ydata):
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                if is_3d and is_integer(zdata):
                    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
            except (ValueError, TypeError) as e:
                # Get the line number of the error
                if config.debug:
                    raise e
                _fn_name = e.__traceback__.tb_frame.f_code.co_name
                _line_number = e.__traceback__.tb_lineno
                get_logger().error(
                    f"{_fn_name}:{_line_number}: "
                    f"Error extracting data from plot {plot_name}: {e}"
                )
                continue

            # Set the aspect ratio
            ax.set_box_aspect(1 if not is_3d else [1, 1, 1])
            fig.tight_layout()

            # Save the plot
            filename = f"{plot_name}.png"
            plt.savefig(
                config.plots_folder / filename,
                dpi=500,
                bbox_inches="tight",
                transparent=False,
            )

            get_logger().debug(f"Saved plot to {config.plots_folder / filename}.")

            plt.close(fig)


# =======================================================


def run_eval(config: ParseEvosConfig, data: Data):
    from cambrian.ml.trainer import MjCambrianTrainer

    # # Have to update the sys.modules to include the features_extractors module
    # # features_extractors is saved in the model checkpoint
    # import sys
    # from cambrian.ml import features_extractors

    # sys.modules["features_extractors"] = features_extractors

    get_logger().info("Evaluating model...")

    for generation, generation_data in data.generations.items():
        if config.generations is not None and generation not in config.generations:
            continue

        get_logger().info(f"Evaluating generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            if config.ranks is not None and rank not in config.ranks:
                continue
            elif rank_data.config is None:
                continue

            get_logger().info(f"\tEvaluating rank {rank}...")

            if not config.dry_run:
                trainer = MjCambrianTrainer(rank_data.config)
                trainer.eval()

            get_logger().info(f"\t\tDone evaluating.")


# =======================================================
# Random helpers


def moving_average(values, window, mode="valid"):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, mode=mode)


# =======================================================


def main(config: ParseEvosConfig):
    assert not (config.debug and config.quiet), "Cannot be both debug and quiet."
    if config.debug:
        get_logger().setLevel("DEBUG")
    elif config.quiet:
        get_logger().setLevel("WARNING")

    assert config.folder.exists(), f"Folder {config.folder} does not exist."
    config.plots_folder.mkdir(parents=True, exist_ok=True)
    config.evals_folder.mkdir(parents=True, exist_ok=True)

    if (hydra_folder := config.hydra_folder) is not None:
        assert hydra_folder.exists(), f"Folder {hydra_folder} does not exist."
        hydra_config = OmegaConf.load(hydra_folder / ".hydra" / "hydra.yaml")
        with config.set_readonly_temporarily(False):
            config.hydra_config = hydra_config

        print(config.hydra_config.sweeper.parameterization)
        exit()

    if config.force or (data := try_load_pickle_data(config.folder)) is None:
        data = load_data(config)

        if not config.no_save:
            save_data(config, data)

    if config.plot:
        run_plot(config, data)
    if config.eval:
        run_eval(config, data)


if __name__ == "__main__":
    run_hydra(main, config_name="tools/parse_evos/parse_evos")
