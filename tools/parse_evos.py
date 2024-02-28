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

import argparse
from typing import Dict, Union, Optional, Any, List, Tuple
from pathlib import Path
import pickle
import os
from dataclasses import dataclass, field
import csv
from functools import partial

import mujoco as mj
import tqdm.rich as tqdm
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy

from cambrian.env import MjCambrianEnv
from cambrian.utils import evaluate_policy, setattrs_temporary
from cambrian.utils.wrappers import make_single_env
from cambrian.utils.config import MjCambrianConfig
from cambrian.ml.model import MjCambrianModel

# =======================================================
# Dataclasses


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class Rank:
    """A rank is a single run of an inner training loop. Like in a generation, you have
    many ranks which in themselves have many parallel environments to speed up training.
    The terminology comes from MPI.
    """

    path: Path

    num: int

    config: Optional[MjCambrianConfig] = None
    evaluations: Optional[Dict[str, Any]] = None
    monitor: Optional[Dict[str, Any]] = None


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class Generation:
    """A generation is a collection of ranks. Throughout evolution (outer loop), you
    have many generations where each generation consists of many ranks which
    train (inner loop) in parallel."""

    path: Path

    num: int
    ranks: Dict[int, Rank] = field(default_factory=dict)


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class Data:
    """This is the primary data storage class. It contains all the generations and ranks
    and is used to store the parsed data. It also can accumulated arbitrary data
    which is used for plotting."""

    path: Path

    generations: Dict[int, Generation]

    accumulated_data: Dict[str, Any] = field(default_factory=dict)


# =======================================================
# Data loaders


def save_data(generations: Dict, folder: Path):
    """Save the parsed data to a pickle file."""
    pickle_file = folder / "parse_evos" / "data.pkl"
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_file, "wb") as f:
        pickle.dump(generations, f)
    print(f"Saved parsed data to {pickle_file}.")


def try_load_pickle_data(folder: Path) -> Union[None, Dict]:
    """Try to load the data from the pickle file."""
    pickle_file = folder / "parse_evos" / "data.pkl"
    if pickle_file.exists():
        with open(pickle_file, "rb") as f:
            generations = pickle.load(f)
        print(f"Loaded parsed data from {pickle_file}.")
        return generations

    print(f"Could not load {pickle_file}.")
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


def load_data(
    folder: Path,
    no_check_finished: bool = False,
    *,
    overrides: List[List[str]] = [],
    **kwargs,
) -> Data:
    """Load the data from the generation/rank folders. This function will walk through
    the folder heirarchy and load the config, evaluations and monitor data for each
    rank."""
    print(f"Loading data from {folder}...")

    # Get the file paths to all the generation/rank folders
    data = get_generation_file_paths(folder)

    # We can ignore certain data if we want
    ignore = kwargs.get("ignore", [])

    # Walk through each generation/rank and parse the config, evaluations and monitor
    # data
    for generation, generation_data in data.generations.items():
        print(f"Loading generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            print(f"\tLoading rank {rank}...")

            # Check if the `finished` file exists.
            # If not, don't load the data. Sorry for the double negative.
            if not no_check_finished and not (rank_data.path / "finished").exists():
                print(f"\t\tSkipping rank {rank} because it is not finished.")
                continue

            # Get the config file
            if (config_file := rank_data.path / "config.yaml").exists():
                if "config" not in ignore:
                    print(f"\tLoading config from {config_file}...")
                    rank_data.config = MjCambrianConfig.load(
                        config_file,
                        overrides=overrides,
                    )

            # Get the evaluations file
            if (evaluations_file := rank_data.path / "evaluations.npz").exists():
                if "evaluations" not in ignore:
                    with np.load(evaluations_file) as evaluations_data:
                        evaluations = {
                            k: evaluations_data[k] for k in evaluations_data.files
                        }
                    rank_data.evaluations = evaluations

            # Get the monitor file
            if (rank_data.path / "monitor.csv").exists():
                if "monitor" not in ignore:
                    rank_data.monitor = load_results(rank_data.path)

            # Get the gpu_usage file
            # Will save it to config.custom
            if (rank_data.path / "gpu_usage.csv").exists() and rank_data.config:
                if "gpu_usage" not in ignore:
                    with open(rank_data.path / "gpu_usage.csv", "r") as f:
                        reader = csv.reader(f)
                        headers = next(reader)
                        gpu_usage = {header: [] for header in headers}
                        for row in reader:
                            for i, header in enumerate(headers):
                                gpu_usage[header].append(float(row[i]) / 1e9)  # GB
                    rank_data.config.custom["gpu_usage"] = gpu_usage

    return data


# =======================================================
# Plotters


def plot_helper(
    *args, title: str, xlabel: str, ylabel: str, dry_run: bool = False, **kwargs
) -> plt.Figure | None:
    """This is a helper method that will be used to plot the data.

    NOTE: Saving the plots at the end depends on each plt figure having a unique name,
    which this method sets to the passed `title`.

    Keyword arguments:
        **kwargs -- Additional keyword arguments to pass to plt.plot.
    """
    if dry_run:
        return

    fig = plt.figure(title)
    plt.plot(*args, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(title)

    return fig


def plot_config(
    config: MjCambrianConfig,
    xvalues: Any,
    pattern: str,
    *,
    dry_run: bool = False,
    **kwargs,
):
    """Plot the config data."""

    def plot_config_helper(attr: str, values: Any, **kwargs):
        # if values is a list, we'll average it
        if isinstance(values, list):
            values = np.average(values)

        # Plot the values
        xlabel = kwargs.pop("xlabel").title().replace("_", " ")
        title = f"{attr.title().replace('_', ' ')} vs {xlabel}"  # xlabel required by plot_helper
        ylabel = attr.split(".")[-1].replace("_", " ").title()
        plot_helper(
            xvalues,
            values,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            dry_run=dry_run,
            **kwargs,
        )

    # Get the config attributes to plot
    data = config.glob(pattern, flatten=True)

    # Now loop through each key and plot the values
    for attr, values in data.items():
        plot_config_helper(attr, values, **kwargs)


def plot_evaluations(
    evaluations: Dict[str, Any],
    xvalues: Any,
    *,
    dry_run: bool = False,
    **kwargs,
):
    """Plot the evaluations data."""
    assert "xlabel" in kwargs, "xlabel is required."
    assert "title" not in kwargs, "title is not allowed."
    xlabel = kwargs["xlabel"]

    key = "mean_results"
    if key not in evaluations:
        key = "results"

    title = f"eval_rewards_vs_{xlabel}"
    plot_helper(
        xvalues,
        np.max(evaluations[key]),
        title=title,
        ylabel="rewards",
        dry_run=dry_run,
        **kwargs,
    )


def plot_monitor(
    monitor: Dict[str, Any],
    xvalues: Any,
    *,
    dry_run: bool = False,
    window: int = 100,
    **kwargs,
):
    """Plot the monitor data."""
    assert "xlabel" in kwargs, "xlabel is required."
    assert "title" not in kwargs, "title is not allowed."
    xlabel = kwargs["xlabel"]

    # Grab the data sorted by timesteps
    x, y = ts2xy(monitor, "timesteps")
    t = ts2xy(monitor, "walltime_hrs")[0] * 60  # convert to minutes
    if len(y) < window:
        # Early exit if not enought data was recorded yet
        return

    # Calculate the moving average
    y = moving_average(y, window)
    x = x[window - 1 :].astype(np.int64)  # adjust the length of x to match y
    t = t[window - 1 :]  # adjust the length of t to match y

    # Now plot the data
    title = f"monitor_rewards_vs_{xlabel}"
    plot_helper(
        xvalues,
        y[-1],
        ylabel="rewards",
        dry_run=dry_run,
        title=title,
        **kwargs,
    )

    # Plot the walltime if requested
    title = f"monitor_walltime_vs_{xlabel}"
    plot_helper(
        xvalues,
        t[-1],
        ylabel="walltime (minutes)",
        dry_run=dry_run,
        title=title,
        **kwargs,
    )


def plot_monitor_and_config(
    monitor: Dict[str, Any],
    config: MjCambrianConfig,
    pattern: str,
    *,
    dry_run: bool = False,
    window: int = 100,
    **kwargs,
):
    """Plot the monitor data."""

    # Get the config attributes to plot
    data = config.glob(pattern, flatten=True)

    # Now loop through each key and plot the values
    for attr, values in data.items():
        # If xvalues is a list, we'll average it
        if isinstance(values, list):
            values = np.average(values)

        # Now plot the monitor data
        plot_monitor(
            monitor,
            values,
            dry_run=dry_run,
            window=window,
            xlabel=attr,
            **kwargs,
        )


def plot_evaluations_and_config(
    evaluations: Dict[str, Any],
    config: MjCambrianConfig,
    pattern: str,
    *,
    dry_run: bool = False,
    **kwargs,
):
    """Plot the evaluations data."""

    # Get the config attributes to plot
    data = config.glob(pattern, flatten=True)

    # Now loop through each key and plot the values
    for attr, values in data.items():
        # If xvalues is a list, we'll average it
        if isinstance(values, list):
            values = np.average(values)

        # Now plot the evaluations data
        plot_evaluations(
            evaluations,
            values,
            dry_run=dry_run,
            xlabel=attr,
            **kwargs,
        )


def plot_average_line(ax: plt.Axes):
    """Extracts the data from a figure and plots the average line along with
    the standard deviation."""

    # Extract the data from the plot
    x_data, y_data = [], []
    for line in ax.lines:
        x, y = line.get_xydata().T
        x_data.append(x)
        y_data.append(y)
    x_data, y_data = np.array(x_data), np.array(y_data)

    # Calculates the average y value for each unique x value
    x, y, y_std = [], [], []
    for unique_x in np.unique(x_data):
        x.append(unique_x)
        y.append(np.average(y_data[x_data == unique_x]))
        y_std.append(np.std(y_data[x_data == unique_x]))
    x, y, y_std = np.array(x), np.array(y), np.array(y_std)

    # Plot the data
    plt.plot(x, y, "C0-", alpha=0.5)
    plt.fill_between(x, y - y.std(), y + y.std(), alpha=0.2, facecolor="C0")


def plot(
    data: Data,
    output_folder: Path,
    *,
    rank_to_use: Optional[int] = None,
    generation_to_use: Optional[int] = None,
    dry_run: bool = False,
    verbose: int = 0,
    **kwargs,
):
    if verbose > 0:
        print("Plotting data...")

    # First, create a matplotlib colormap so each rank has a unique color + marker
    num_ranks = max(
        len(generation.ranks)
        for generation in data.generations.values()
        if generation.ranks
    )
    colors = plt.cm.tab20(np.linspace(0, 1, num_ranks))
    markers = [".", ",", "o", "v", "^", "<", ">", "s", "p", "*", "h", "+", "x"]
    markers = ["."]

    # set the colors to be a pastel blue with alpha of 0.1
    colors = np.array([[0.65490196, 0.78039216, 0.90588235, 0.75]])

    # We can ignore certain data if we want
    ignore = kwargs.get("ignore", [])

    output_folder.mkdir(parents=True, exist_ok=True)
    for generation, generation_data in data.generations.items():
        # Only plot the generation we want, if specified
        if generation_to_use is not None and generation != generation_to_use:
            continue

        if verbose > 0:
            print(f"Plotting generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            # Only plot the rank we want, if specified
            if rank_to_use is not None and rank != rank_to_use:
                continue

            if verbose > 0:
                print(f"\tPlotting rank {rank}...")

            # The color + marker of each rank is unique
            color = colors[rank % len(colors)]
            marker = markers[rank % len(markers)]

            # Plot config data
            if rank_data.config is not None and "config" not in ignore:
                if verbose > 1:
                    print("\t\tPlotting config data...")

                # Build the glob pattern for the config attributes
                # * indicates anything (like names which we don't know beforehand) and (|) indicates
                # an OR operation (i.e. (resolution|fov) matches either resolution or fov)
                pattern = build_pattern(
                    "training_config.seed",
                    "env_config.animal_configs.*.eye_configs.*.aperture_open",
                    "env_config.animal_configs.*.eye_configs.*.aperture_radius",
                    "env_config.animal_configs.*.eye_configs.*.resolution",
                    "env_config.animal_configs.*.eye_configs.*.fov",
                    "env_config.animal_configs.*.num_eyes",
                    "custom.gpu_usage.*",
                )
                plot_config(
                    rank_data.config,
                    generation,
                    pattern,
                    color=color,
                    marker=marker,
                    xlabel="generation",
                    dry_run=dry_run,
                )

            # Plot evaluations data
            if rank_data.evaluations is not None and "evaluations" not in ignore:
                if verbose > 1:
                    print("\t\tPlotting evaluations data...")

                plot_evaluations(
                    rank_data.evaluations,
                    generation,
                    color=color,
                    marker=marker,
                    xlabel="generation",
                    label=f"Rank {rank}",
                    dry_run=dry_run,
                )

            # Plot monitor data
            if rank_data.monitor is not None and "monitor" not in ignore:
                if verbose > 1:
                    print("\t\tPlotting monitor data...")

                plot_monitor(
                    rank_data.monitor,
                    generation,
                    color=color,
                    marker=marker,
                    xlabel="generation",
                    dry_run=dry_run,
                )

            # Also plot with some different x values
            if rank_data.monitor is not None and rank_data.config is not None:
                if "monitor" not in ignore and "config" not in ignore:
                    if verbose > 1:
                        print("\t\tPlotting monitor and config data...")

                    # Build the glob pattern for the config attributes
                    pattern = build_pattern(
                        "env_config.animal_configs.*.eye_configs.*.aperture_open",
                        "env_config.animal_configs.*.eye_configs.*.aperture_radius",
                        "env_config.animal_configs.*.eye_configs.*.fov",
                        "env_config.animal_configs.*.eye_configs.*.resolution",
                        "env_config.animal_configs.*.num_eyes",
                    )
                    plot_monitor_and_config(
                        rank_data.monitor,
                        rank_data.config,
                        pattern,
                        color=color,
                        marker=marker,
                        dry_run=dry_run,
                    )

            # Plot evaluations and config data
            if rank_data.evaluations is not None and rank_data.config is not None:
                if "evaluations" not in ignore and "config" not in ignore:
                    if verbose > 1:
                        print("\t\tPlotting evaluations and config data...")

                    # Build the glob pattern for the config attributes
                    pattern = build_pattern(
                        "env_config.animal_configs.*.eye_configs.*.aperture_open",
                        "env_config.animal_configs.*.eye_configs.*.aperture_radius",
                        "env_config.animal_configs.*.eye_configs.*.fov",
                        "env_config.animal_configs.*.eye_configs.*.resolution",
                        "env_config.animal_configs.*.num_eyes",
                    )
                    plot_evaluations_and_config(
                        rank_data.evaluations,
                        rank_data.config,
                        pattern,
                        color=color,
                        marker=marker,
                        dry_run=dry_run,
                    )

    # Now save the plots
    if not dry_run:
        for fig in tqdm.tqdm(plt.get_fignums(), desc="Saving plots..."):
            fig = plt.figure(fig)
            filename = f"{fig._suptitle.get_text().lower().replace(' ', '_')}.png"

            # Plot the average line
            plot_average_line(fig.gca())

            # Add a legend entry for the blue circles
            (selected_agent,) = fig.gca().plot(
                [], [], ".", color=colors, label="Selected Agent"
            )
            fig.gca().legend(handles=[selected_agent])

            fig.tight_layout()
            plt.gca().set_box_aspect(1)
            plt.savefig(
                output_folder / filename,
                dpi=500,
                bbox_inches="tight",
                transparent=False,
            )

            if verbose > 1:
                print(f"Saved plot to {output_folder / filename}.")

            plt.close(fig)


# =======================================================


def phylogenetic_tree(
    data: Data,
    output_folder: Path,
    *,
    rank_to_use: Optional[int] = None,
    generation_to_use: Optional[int] = None,
    dry_run: bool = False,
    verbose: int = 0,
    **kwargs,
):
    import ete3 as ete

    def add_node(
        nodes: Dict[str, ete.Tree],
        *,
        rank_data: Rank,
        generation_data: Generation,
    ):
        rank = rank_data.num
        generation = generation_data.num

        if generation_to_use is not None and generation != generation_to_use:
            return
        if rank_to_use is not None and rank != rank_to_use:
            return

        # Define a unique identifier for each rank
        rank_id = f"G{generation}_R{rank}"
        if rank_id in nodes:
            return

        # Get the parent identifier
        if not (config := rank_data.config):
            print(f"Skipping rank {rank_id} because it has no config.")
            return
        if not (evaluations := rank_data.evaluations):
            print(f"Skipping rank {rank_id} because it has no evaluations.")
            return

        # Fix the config
        rank_data.config.fix()

        # If this is the first generation, the parent is set to the root
        if generation == 0:
            parent_id = "root"
        else:
            if not (parent_config := config.evo_config.parent_generation_config):
                print(f"Skipping rank {rank_id} because it has no parent config.")
                return
            parent_id = f"G{parent_config.generation}_R{parent_config.rank}"

            if parent_id not in nodes:
                parent_generation_data = data.generations[parent_config.generation]
                add_node(
                    nodes,
                    rank_data=parent_generation_data.ranks[parent_config.rank],
                    generation_data=parent_generation_data,
                )

        # Create the rank node under the parent node
        parent = nodes[parent_id]
        node = parent.add_child(name=rank_id)
        nodes[rank_id] = node

        # Add features for the ranks and generations
        node.add_feature("rank", rank)
        node.add_feature("generation", generation)
        if generation != 0:
            node.add_feature("parent_rank", parent_config.rank)
            node.add_feature("parent_generation", parent_config.generation)
        else:
            node.add_feature("parent_rank", -1)
            node.add_feature("parent_generation", -1)

        # Add evaluations to the node
        key = "mean_results" if "mean_results" in evaluations else "results"
        node.add_feature("fitness", np.max(evaluations[key]))

        # Add a text label and feature for the pattern
        globbed_data = config.glob(pattern, flatten=True)
        for key, value in globbed_data.items():
            if isinstance(value, list):
                value = np.average(value).astype(type(value[0]))

            node.add_feature(key, value)

    def build_tree(pattern: str) -> ete.Tree:
        tree = ete.Tree()
        tree.name = "root"

        # Dictionary to keep track of created nodes
        nodes = {"root": tree}

        # Iterate through the generations and ranks
        for generation_data in data.generations.values():
            for rank_data in generation_data.ranks.values():
                add_node(nodes, rank_data=rank_data, generation_data=generation_data)

        return tree

    def style_node(
        node: ete.Tree,
        *,
        style_feature_key: str,
        value_threshold: Optional[float] = None,
        value_range: Optional[Tuple[float, float]] = None,
        color: Optional[str] = None,
        color_range: Optional[Tuple[str, str]] = None,
    ):
        # We'll color the node and the recursive ancestors to highlight good values
        # The horizontal lines which directly lead to the best value are bolded and
        # highlighted red. Horizontal lines which are children of the optimal path but
        # not the best value are highlighted red. The vertical lines are just
        # highlighted red.
        feature_value = getattr(node, style_feature_key)
        assert value_threshold is None or value_range is None

        # If the feature value is less than the threshold, don't style the node
        if value_threshold and feature_value < value_threshold:
            return

        assert color is None or color_range is None
        if color_range:
            assert value_range is not None
            assert value_range[0] <= feature_value <= value_range[1]

            # Interpolate the color between the color range
            # The color range is defined as hex strings
            def hex_to_rgb(hex_str: str):
                return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))

            def rgb_to_hex(rgb: Tuple[int, int, int]):
                return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

            color_range = [hex_to_rgb(color) for color in color_range]
            t = np.interp(feature_value, value_range, [0, 1])
            color = tuple(int(c0 + (c1 - c0) * t) for c0, c1 in zip(*color_range))
            color = rgb_to_hex(color)

        # Create the node style
        path_style = ete.NodeStyle()
        path_style["fgcolor"] = color
        path_style["hz_line_color"] = color
        path_style["hz_line_width"] = 4

        # Set the node style
        current_node = node
        while current_node.up:
            current_node.set_style(path_style)
            current_node = current_node.up

    def layout(
        node: ete.Tree,
        feature_key: str,
        *,
        style_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if node.name == "root":
            return

        # Add a text annotation to each node
        feature_value = getattr(node, feature_key)
        text_face = ete.TextFace(feature_value, bold=True)
        node.add_face(text_face, column=0, position="branch-right")

        # Set the node distance. Each node distance is set such that each generation
        # is at the same distance from the root
        node.dist = getattr(node, "generation") - getattr(node, "parent_generation")
        node.dist -= text_face.get_bounding_rect().width() / 25
        node.dist = max(node.dist, 0)

        # Update the node style, if desired
        if style_kwargs is not None:
            style_node(node, **style_kwargs)

    # Build the tree
    # We'll add features/data to be used in the visualization later
    patterns = {
        "num_eyes": "env_config.animal_configs.*.num_eyes",
        "generation": "evo_config.generation_config.generation",
    }
    pattern = build_pattern(*list(patterns.values()))
    const_tree = build_tree(pattern)

    for feature_key in patterns.keys():
        print(f"Creating phylogenetic tree for feature {feature_key}...")
        tree = const_tree.copy("deepcopy")

        # Sort the descendants of the tree by the feature key
        tree.sort_descendants(feature_key)

        # Get the max value for the feature key
        style_feature_key = "fitness"
        values = [getattr(node, style_feature_key) for node in tree.iter_leaves()]
        threshold = np.percentile(values, 95)
        style_kwargs = dict(
            style_feature_key=style_feature_key,
            value_threshold=threshold,
            color="#ff0000",
        )
        layout_fn = partial(
            layout,
            feature_key=feature_key,
            style_kwargs=style_kwargs,
        )

        # Create the visualization
        tree_style = ete.treeview.TreeStyle()
        tree_style.show_leaf_name = False
        tree_style.layout_fn = layout_fn
        tree_style.mode = 'c'
        tree_style.arc_start = -180
        tree_style.arc_span = 180
        tree_style.title.add_face(
            ete.TextFace(f"Phylogenetic Tree for {feature_key}", fsize=20), column=0
        )
        tree.render(
            str(output_folder / f"phylogenetic_tree_{feature_key}.png"),
            dpi=500,
            tree_style=tree_style,
        )


# =======================================================


def muller_plot(
    data: Data,
    output_folder: Path,
    *,
    rank_to_use: Optional[int] = None,
    generation_to_use: Optional[int] = None,
    dry_run: bool = False,
    verbose: int = 0,
    **kwargs,
):
    """Creates a muller plot."""

    patterns = [
        "env_config.animal_configs.*.num_eyes",
    ]
    pattern = build_pattern(*patterns)

    # Outer keys are the patterns which are used to get the data from the config
    # Inner Keys: {"Generation", "Identity", "Population"}
    populations: Dict[str, Dict[str, Any]] = {}
    # Inner Keys: {"Parent", "Identity"}
    adjacency: Dict[str, Dict[str, Any]] = {}

    for generation, generation_data in data.generations.items():
        if generation_to_use is not None and generation != generation_to_use:
            continue

        for rank, rank_data in generation_data.ranks.items():
            # Only plot the rank we want, if specified
            if rank_to_use is not None and rank != rank_to_use:
                continue

            if not (config := rank_data.config):
                print(f"Skipping rank {rank} because it has no config.")
                continue
            if not (parent_config := config.evo_config.parent_generation_config):
                print(f"Skipping rank {rank} because it has no parent config.")
                continue

            # Glob the data
            globbed_data = config.glob(pattern, flatten=True)

            for key, value in globbed_data.items():
                if isinstance(value, list):
                    value = np.average(value).astype(type(value[0]))

                populations.setdefault(key, {})
                adjacency.setdefault(key, {})

                current_id = f"G{generation}_R{rank}"
                parent_id = f"G{parent_config.generation}_R{parent_config.rank}"

                # Add the population data
                populations[key].setdefault("Generation", []).append(generation)
                populations[key].setdefault("Identity", []).append(rank)
                populations[key].setdefault("Population", []).append(value)

                # Add the adjacency data
                adjacency[key].setdefault("Parent", []).append(current_id)
                adjacency[key].setdefault("Identity", []).append(parent_id)


# =======================================================


def run_trace_eval(logdir: Path, filename: Path, config: MjCambrianConfig):
    env = DummyVecEnv([make_single_env(config, config.training_config.seed)])
    cambrian_env: MjCambrianEnv = env.envs[0].env

    model = MjCambrianModel.load(logdir / "best_model")

    # We'll interpolate from start_rgba to final_rgba over n_runs
    start_rgba, final_rgba = [1, 1, 1, 0.01], [1, 1, 1, 1]
    cambrian_env.env_config.position_tracking_overlay_color = start_rgba

    # Number of runs is calculated based on the training config
    # The current run is then used to read the evaluation pkls
    n_runs = config.training_config.total_timesteps // (
        config.training_config.eval_freq * config.training_config.n_envs
    )

    def done_callback(run: int):
        """Callback called on each reset to update the position of the animal and
        the overlay color."""
        # Update the model rollout to read from
        rollout_pkl = logdir / "evaluations" / f"vis_{run}.pkl"
        model.load_rollout(rollout_pkl)

        # Update the position of the animal to match the current run
        with open(rollout_pkl, "rb") as f:
            data = pickle.load(f)
            positions = np.array(data["positions"])
            assert len(positions[0]) == len(cambrian_env.animals)
            for position, animal in zip(positions[0], cambrian_env.animals.values()):
                animal.pos = np.squeeze(position)
                mj.mj_forward(cambrian_env.model, cambrian_env.data)
                animal.init_pos = animal.pos

        # Interpolate start_rgba to final_rgba over n_runs
        cambrian_env.env_config.position_tracking_overlay_color = [
            start_rgba[i] + (final_rgba[i] - start_rgba[i]) * run / n_runs
            for i in range(4)
        ]

        return True

    # Call the done_callback once to update the position of the animal and the overlay
    # color
    done_callback(0)

    # Update the config with the eval overrides and run evaluation
    temp_attrs = []
    if (eval_overrides := config.env_config.eval_overrides) is not None:
        temp_attrs.append((cambrian_env.env_config, eval_overrides))
    with setattrs_temporary(*temp_attrs):
        record_kwargs = dict(path=filename, save_pkl=False, save_types=["webp", "png"])
        evaluate_policy(
            env,
            model,
            n_runs,
            record_kwargs=record_kwargs,
            done_callback=done_callback,
        )


def eval(
    data: Data,
    output_folder: Path,
    *,
    rank_to_use: Optional[int] = None,
    generation_to_use: Optional[int] = None,
    verbose: int = 0,
    dry_run: bool = False,
    flags: List[str] = [],
    **kwargs,
):
    # Have to update the sys.modules to include the features_extractors module
    # features_extractors is saved in the model checkpoint
    import sys
    from cambrian.ml import features_extractors

    sys.modules["features_extractors"] = features_extractors

    if verbose > 0:
        print("Evaluating model...")

    # Update the flags to default to trace if not passed
    if not flags:
        flags = ["trace"]

    output_folder.mkdir(parents=True, exist_ok=True)
    for generation, generation_data in data.generations.items():
        if generation_to_use is not None and generation != generation_to_use:
            continue

        if verbose > 1:
            print(f"Evaluating generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            if rank_to_use is not None and rank != rank_to_use:
                continue

            if verbose > 1:
                print(f"\tEvaluating rank {rank}...")

            if not dry_run:
                if "trace" in flags:
                    if verbose > 1:
                        print("\t\tEvaluating trace...")

                    run_trace_eval(
                        rank_data.path,
                        output_folder / f"generation_{generation}_rank_{rank}",
                        rank_data.config,
                    )

            if verbose > 1:
                print("\tDone.")


# =======================================================
# Random helpers


def moving_average(values, window, mode="valid"):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, mode=mode)


def build_pattern(*patterns: str) -> str:
    """Build a glob pattern from the passed patterns.

    The underlying method for globbing (`MjCambrianConfig.glob`) uses a regex pattern
    which is parses the dot-separated keys independently.

    Example:
        >>> build_pattern(
        ...     "training_config.seed",
        ...     "env_config.animal_configs.*.eye_configs.*.resolution",
        ...     "env_config.animal_configs.*.eye_configs.*.fov",
        ... )
        '(training_config|env_config).(seed|animal_configs).*.eye_configs.*.(resolution|fov)'
    """
    depth_based_keys: List[List[str]] = []  # list of keys at depths in the patterns
    for pattern in patterns:
        # For each key in the pattern, add at the same depth as the other patterns
        for i, key in enumerate(pattern.split(".")):
            if i < len(depth_based_keys):
                if key not in depth_based_keys[i]:
                    depth_based_keys[i].extend([key])
            else:
                depth_based_keys.append([key])

    # Now build the pattern
    pattern = ""
    for keys in depth_based_keys:
        pattern += "(" + "|".join(keys) + ")."
    pattern = pattern[:-1]  # remove the last dot
    return pattern


# =======================================================


def main(args):
    folder = Path(args.folder)
    plots_folder = (
        folder / "parse_evos" / "plots" if args.output is None else Path(args.output)
    )
    plots_folder.mkdir(parents=True, exist_ok=True)
    evals_folder = (
        folder / "parse_evos" / "evals" if args.output is None else Path(args.output)
    )
    evals_folder.mkdir(parents=True, exist_ok=True)

    if args.force or (data := try_load_pickle_data(folder)) is None:
        data = load_data(**vars(args))

        if not args.no_save:
            save_data(data, folder)

    kwargs = dict(
        data=data,
        rank_to_use=args.rank,
        generation_to_use=args.generation,
        **vars(args),
    )
    if args.plot:
        plot(
            output_folder=plots_folder,
            **kwargs,
        )
    if args.phylogeny:
        phylogenetic_tree(output_folder=plots_folder, **kwargs)
    if args.muller:
        muller_plot(output_folder=plots_folder, **kwargs)
    if args.eval:
        eval(output_folder=evals_folder, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the evolution folder.")

    parser.add_argument("-v", "--verbose", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Dry run.")
    parser.add_argument(
        "-o",
        "--override",
        "--overrides",
        dest="overrides",
        action="extend",
        nargs="+",
        type=str,
        help="Override config values. Do <config>.<key>=<value>",
        default=[],
    )

    parser.add_argument("folder", type=str, help="The folder to parse.")
    parser.add_argument(
        "-O",
        "--output",
        type=str,
        help="The output folder. Defaults to <folder>/parse_evos/",
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force loading of the data. If not passed, this script will try to find a "
        "parse_evos.pkl file and load that instead.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save the parsed data."
    )
    parser.add_argument(
        "--rank",
        type=int,
        help="The rank to plot. If not passed, all ranks are plotted.",
        default=None,
    )
    parser.add_argument(
        "--generation",
        type=int,
        help="The generation to plot. If not passed, all generations are plotted.",
        default=None,
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate the data.")
    parser.add_argument(
        "--phylogeny", action="store_true", help="Construct phylogenetic tree."
    )
    parser.add_argument("--plot", action="store_true", help="Plot the data.")
    parser.add_argument("--muller", action="store_true", help="Plot the muller plot.")
    parser.add_argument(
        "--flags", nargs="+", help="Flags to pass to the script.", default=[]
    )
    parser.add_argument("--ignore", nargs="+", help="Ignore certain data.", default=[])
    parser.add_argument(
        "--no-check-finished",
        action="store_true",
        help="Don't check if a file called `finished` has been written.",
    )

    args = parser.parse_args()

    main(args)
