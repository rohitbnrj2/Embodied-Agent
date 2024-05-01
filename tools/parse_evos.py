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

from typing import Dict, Union, Optional, Any, List, Tuple
from pathlib import Path
import os
from dataclasses import field

import cloudpickle as pickle
from omegaconf import OmegaConf
import tqdm.rich as tqdm
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

from cambrian.utils.config import (
    MjCambrianBaseConfig,
    MjCambrianConfig,
    run_hydra,
    config_wrapper,
    build_pattern,
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
    monitor: Optional[Dict[str, Any]] = None


@config_wrapper
class Generation:
    """A generation is a collection of ranks. Throughout evolution (outer loop), you
    have many generations where each generation consists of many ranks which
    train (inner loop) in parallel."""

    path: Path

    num: int
    ranks: Dict[int, Rank] = field(default_factory=dict)


@config_wrapper
class Data:
    """This is the primary data storage class. It contains all the generations and ranks
    and is used to store the parsed data. It also can accumulated arbitrary data
    which is used for plotting."""

    path: Path

    generations: Dict[int, Generation]

    accumulated_data: Dict[str, Any] = field(default_factory=dict)


@config_wrapper
class PlotData:
    pattern: str

    xlabel: str
    ylabel: str
    title: str
    color: Tuple[float, float, float, float]
    output: str


@config_wrapper
class ParseEvosConfig(MjCambrianBaseConfig):
    """Config for the parse_evos script.

    Attributes:
        folder (Path): The folder to parse.
        output (Path): The output folder.
        plots_folder (Path): The folder to save the plots.
        evals_folder (Path): The folder to save the evaluations.

        force (bool): Force loading of the data. If not passed, this script will try to
            find a parse_evos.pkl file and load that instead.
        no_save (bool): Do not save the parsed data.
        no_check_finished (bool): Don't check if a file called `finished` has been
            written.

        ranks (Optional[List[int]]): The rank to use. If not passed, all ranks are
            used.
        generations (Optional[List[int]]): The generation to use. If not passed, all
            are used.

        plot (bool): Plot the data.
        eval (bool): Evaluate the data.

        verbose (int): The verbosity level.
        dry_run (bool): Dry run.

        patterns (List[str]): The patterns to use for plotting.
        overrides (List[str]): Overrides for the config.
    """

    folder: Path
    output: Path
    plots_folder: Path
    evals_folder: Path

    force: bool
    no_save: bool
    no_check_finished: bool

    ranks: Optional[List[int]] = None
    generations: Optional[List[int]] = None

    plot: bool
    eval: bool

    verbose: int
    dry_run: bool

    patterns: List[str]
    overrides: List[str]


# =======================================================
# Data loaders


def save_data(config: ParseEvosConfig, generations: Dict[int, Generation]):
    """Save the parsed data to a pickle file."""
    pickle_file = config.output / "data.pkl"
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_file, "wb") as f:
        pickle.dump(generations, f)
    print(f"Saved parsed data to {pickle_file}.")


def try_load_pickle_data(folder: Path) -> Union[None, Dict]:
    """Try to load the data from the pickle file."""
    pickle_file = folder / "parse_evos" / "data.pkl"
    if pickle_file.exists():
        print(f"Loading parsed data from {pickle_file}...")
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


def load_data(config: ParseEvosConfig) -> Data:
    """Load the data from the generation/rank folders. This function will walk through
    the folder heirarchy and load the config, evaluations and monitor data for each
    rank."""
    print(f"Loading data from {config.folder}...")

    # Get the file paths to all the generation/rank folders
    data = get_generation_file_paths(config.folder)

    # Convert the overrides to a list
    overrides = OmegaConf.to_container(config.overrides)

    # Walk through each generation/rank and parse the config, evaluations and monitor
    # data
    for generation, generation_data in data.generations.items():
        print(f"Loading generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            print(f"\tLoading rank {rank}...")

            # Check if the `finished` file exists.
            # If not, don't load the data. Sorry for the double negative.
            if not (rank_data.path / "finished").exists():
                if not config.no_check_finished:
                    print(f"\t\tSkipping rank {rank} because it is not finished.")
                    continue

            # Get the config file
            # Try first to load it as a pickle file, then as a yaml file if the pickle
            # file doesn't exist.
            if (config_file := rank_data.path / "config.pkl").exists():
                print(f"\tLoading config from {config_file}...")
                rank_data.config = MjCambrianConfig.load_pickle(
                    config_file, overrides=overrides
                )
            elif (config_file := rank_data.path / "config.yaml").exists():
                print(f"\tLoading config from {config_file}...")
                rank_data.config = MjCambrianConfig.load(config_file, instantiate=False)
                rank_data.config.merge_with_dotlist(overrides)
                rank_data.config.resolve()

            # Get the evaluations file
            evaluations_file = rank_data.path / "evaluations.npz"
            if evaluations_file.exists():
                with np.load(evaluations_file) as eval_data:
                    rank_data.evaluations = {k: eval_data[k] for k in eval_data.files}

            # Get the monitor file
            monitor_file = rank_data.path / "monitor.csv"
            if monitor_file.exists():
                rank_data.monitor = load_results(rank_data.path)

            # Get the gpu_usage file
            # Will save it to config.custom
            # usage_file = rank_data.path / "gpu_usage.csv"
            # if usage_file.exists() and rank_data.config:
            #    with open(usage_file, "r") as f:
            #        reader = csv.reader(f)
            #        headers = next(reader)
            #        gpu_usage = {header: [] for header in headers}
            #        for row in reader:
            #            for i, header in enumerate(headers):
            #                gpu_usage[header].append(float(row[i]) / 1e9)  # GB
            #    rank_data.config.custom["gpu_usage"] = gpu_usage

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
    assert data, f"No data found for pattern {pattern}."

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
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.2, facecolor="C0")


def run_plot(config: ParseEvosConfig, data: Data):
    if config.verbose > 0:
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

    for generation, generation_data in data.generations.items():
        # Only plot the generation we want, if specified
        if config.generations is not None and generation not in config.generations:
            continue

        if config.verbose > 0:
            print(f"Plotting generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            # Only plot the rank we want, if specified
            if config.ranks is not None and rank not in config.ranks:
                continue

            if config.verbose > 0:
                print(f"\tPlotting rank {rank}...")

            # The color + marker of each rank is unique
            color = colors[rank % len(colors)]
            marker = markers[rank % len(markers)]

            # Plot config data
            if rank_data.config is not None:
                if config.verbose > 1:
                    print("\t\tPlotting config data...")

                # Build the glob pattern for the config attributes
                # * indicates anything (like names which we don't know beforehand) and (|) indicates
                # an OR operation (i.e. (resolution|fov) matches either resolution or fov)
                pattern = build_pattern(config.patterns)
                plot_config(
                    rank_data.config,
                    generation,
                    pattern,
                    color=color,
                    marker=marker,
                    xlabel="generation",
                    dry_run=config.dry_run,
                )

            # Plot evaluations data
            if rank_data.evaluations is not None:
                if config.verbose > 1:
                    print("\t\tPlotting evaluations data...")

                plot_evaluations(
                    rank_data.evaluations,
                    generation,
                    color=color,
                    marker=marker,
                    xlabel="generation",
                    label=f"Rank {rank}",
                    dry_run=config.dry_run,
                )

            # Plot monitor data
            if rank_data.monitor is not None:
                if config.verbose > 1:
                    print("\t\tPlotting monitor data...")

                plot_monitor(
                    rank_data.monitor,
                    generation,
                    color=color,
                    marker=marker,
                    xlabel="generation",
                    dry_run=config.dry_run,
                )

            # Also plot with some different x values
            if rank_data.monitor is not None and rank_data.config is not None:
                if config.verbose > 1:
                    print("\t\tPlotting monitor and config data...")

                # Build the glob pattern for the config attributes
                pattern = build_pattern(config.patterns)
                plot_monitor_and_config(
                    rank_data.monitor,
                    rank_data.config,
                    pattern,
                    color=color,
                    marker=marker,
                    dry_run=config.dry_run,
                )

            # Plot evaluations and config data
            if rank_data.evaluations is not None and rank_data.config is not None:
                if config.verbose > 1:
                    print("\t\tPlotting evaluations and config data...")

                # Build the glob pattern for the config attributes
                pattern = build_pattern(config.patterns)
                plot_evaluations_and_config(
                    rank_data.evaluations,
                    rank_data.config,
                    pattern,
                    color=color,
                    marker=marker,
                    dry_run=config.dry_run,
                )

    # Now save the plots
    if not config.dry_run:
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
                config.plots_folder / filename,
                dpi=500,
                bbox_inches="tight",
                transparent=False,
            )

            if config.verbose > 1:
                print(f"Saved plot to {config.plots_folder / filename}.")

            plt.close(fig)


# =======================================================


def run_eval(config: ParseEvosConfig, data: Data):
    from cambrian.ml.trainer import MjCambrianTrainer

    # # Have to update the sys.modules to include the features_extractors module
    # # features_extractors is saved in the model checkpoint
    # import sys
    # from cambrian.ml import features_extractors

    # sys.modules["features_extractors"] = features_extractors

    if config.verbose > 0:
        print("Evaluating model...")

    for generation, generation_data in data.generations.items():
        if config.generations is not None and generation not in config.generations:
            continue

        if config.verbose > 1:
            print(f"Evaluating generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            if config.ranks is not None and rank not in config.ranks:
                continue
            elif rank_data.config is None:
                continue

            if config.verbose > 1:
                print(f"\tEvaluating rank {rank}...")

            if not config.dry_run:
                trainer = MjCambrianTrainer(rank_data.config)
                trainer.eval()

            if config.verbose > 1:
                print("\tDone.")


# =======================================================
# Random helpers


def moving_average(values, window, mode="valid"):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, mode=mode)


# =======================================================


def main(config: ParseEvosConfig):
    config.plots_folder.mkdir(parents=True, exist_ok=True)
    config.evals_folder.mkdir(parents=True, exist_ok=True)

    if config.force or (data := try_load_pickle_data(config.folder)) is None:
        data = load_data(config)

        if not config.no_save:
            save_data(config, data)

    if config.plot:
        run_plot(config, data)
    if config.eval:
        run_eval(config, data)


if __name__ == "__main__":
    run_hydra(main, config_name="tools/parse_evos", instantiate=False)
