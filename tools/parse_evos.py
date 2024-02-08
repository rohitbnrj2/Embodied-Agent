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
from typing import Dict, Union, Optional, Any, List
from pathlib import Path
import pickle
import os
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy

from cambrian.utils import evaluate_policy
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
    check_finished: bool = True,
    *,
    overrides: Dict[str, Any] = {},
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
            # If not, don't load the data.
            if check_finished and not (rank_data.path / "finished").exists():
                print(f"\t\tSkipping rank {rank} because it is not finished.")
                continue

            # Get the config file
            if (config_file := rank_data.path / "config.yaml").exists():
                if "config" not in ignore:
                    print(f"\tLoading config from {config_file}...")
                    rank_data.config = MjCambrianConfig.load(
                        config_file, overrides=overrides
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

    def plot_config_helper(attr: str, values: Any):
        # if values is a list, we'll average it
        if isinstance(values, list):
            values = np.average(values)

        # Plot the values
        title = f"{attr}_vs_{kwargs.get('xlabel')}"  # xlabel required by plot_helper
        ylabel = attr.split(".")[-1].replace("_", " ").title()
        plot_helper(
            xvalues,
            values,
            title=title,
            ylabel=ylabel,
            dry_run=dry_run,
            **kwargs,
        )

    # Get the config attributes to plot
    data = config.glob(pattern, flatten=True)

    # Now loop through each key and plot the values
    for attr, values in data.items():
        plot_config_helper(attr, values)


def plot_evaluations(
    evaluations: Dict[str, Any],
    xvalues: Any,
    *,
    dry_run: bool = False,
    **kwargs,
):
    """Plot the evaluations data."""

    plot_helper(
        xvalues,
        np.average(evaluations["results"]),
        title="average_eval_rewards",
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
    t = ts2xy(monitor, "walltime_hrs")[0] * 60 # convert to minutes
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
    colors = plt.cm.jet(np.linspace(0, 1, num_ranks))
    markers = [".", ",", "o", "v", "^", "<", ">", "s", "p", "*", "h", "+", "x"]

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
                # Build the glob pattern for the config attributes
                # * indicates anything (like names which we don't know beforehand) and (|) indicates
                # an OR operation (i.e. (resolution|fov) matches either resolution or fov)
                pattern = build_pattern(
                    "training_config.seed",
                    "env_config.animal_configs.*.eye_configs.*.resolution",
                    "env_config.animal_configs.*.eye_configs.*.fov",
                    "env_config.animal_configs.*.num_eyes",
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
                    # Build the glob pattern for the config attributes
                    pattern = build_pattern(
                        "env_config.animal_configs.*.eye_configs.apperture_open",
                        "env_config.animal_configs.*.eye_configs.apperture_radius",
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

    # Now save the plots
    if not dry_run:
        for fig in plt.get_fignums():
            fig = plt.figure(fig)
            plt.gca().set_box_aspect(1)

            filename = f"{fig._suptitle.get_text().lower().replace(' ', '_')}.png"
            plt.savefig(output_folder / filename, dpi=500)

            if verbose > 1:
                print(f"Saved plot to {output_folder / filename}.")


def eval(
    data: Data,
    output_folder: Path,
    *,
    rank_to_use: Optional[int] = None,
    generation_to_use: Optional[int] = None,
    verbose: int = 0,
    dry_run: bool = False,
):
    if verbose > 0:
        print("Evaluating model...")

    def _run_eval(logdir: Path, filename: Path, config: MjCambrianConfig):
        env = DummyVecEnv([make_single_env(config, config.training_config.seed)])

        import sys
        from cambrian.ml import feature_extractors

        sys.modules["feature_extractors"] = feature_extractors
        model = MjCambrianModel.load(logdir / config.training_config.checkpoint_path)
        model.load_rollout(filename.with_suffix(".pkl"))

        evaluate_policy(env, model, 1, record_path=filename)

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

            if verbose > 1:
                print(rank_data.config)

            if not dry_run:
                _run_eval(
                    rank_data.path,
                    output_folder / f"generation_{generation}_rank_{rank}",
                    rank_data.config,
                )

            if verbose > 1:
                print(f"\tDone.")


# =======================================================
# Random helpers


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

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
    depth_based_keys: List[List[str]] = [] # list of keys at depths in the patterns
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
        data = load_data(
            folder,
            check_finished=not args.no_check_finished,
            overrides=args.overrides,
            **vars(args),
        )

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
    if args.eval:
        eval(output_folder=evals_folder, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the evolution folder.")

    parser.add_argument("-v", "--verbose", action="count", default=1)
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
    parser.add_argument("--plot", action="store_true", help="Plot the data.")
    parser.add_argument("--ignore", nargs="+", help="Ignore certain data.", default=[])
    parser.add_argument(
        "--no-check-finished",
        action="store_true",
        help="Don't check if a file called `finished` has been written.",
    )

    args = parser.parse_args()

    main(args)
