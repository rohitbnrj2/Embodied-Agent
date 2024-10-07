from typing import Dict, Any, Optional, Callable, Concatenate, List
from pathlib import Path
from enum import Enum
import re

import numpy as np
import matplotlib.pyplot as plt
import tqdm.rich as tqdm

from cambrian.utils import save_data, try_load_pickle, get_logger, set_matplotlib_style
from cambrian.utils.config import (
    MjCambrianConfig,
    MjCambrianBaseConfig,
    run_hydra,
    config_wrapper,
)

# ==================

set_matplotlib_style()

# ==================


@config_wrapper
class ParseSweepsConfig(MjCambrianBaseConfig):
    """
    folder (Path): The folder containing the sweep data.
    output (Path): The folder to save the parsed data.
    pkl (Path): The name of the pickle file to save the data to.

    force (bool): Force loading of the data. If not passed, this script will try to
        find a parse_sweep.pkl file and load that instead.
    plot (bool): Plot the data.
    save (bool): Save the data to the output folder.
    quiet (bool): Quiet mode. Set's the logger to warning.
    debug (bool): Debug mode. Set's the logger to debug and disables tqdm.

    show_plots (bool): Show the plots.
    save_plots (bool): Save the plots. Will save to the output / plots folder.

    ignore (List[str]): List of folders to ignore.
    overrides (Dict[str, Any]): Overrides for the sweep data.

    dry_run (bool): Do not actually do any of the processing, just run the code without
        that part.
    """

    folder: Path
    output: Path
    pkl: Path

    force: bool
    plot: bool
    quiet: bool
    debug: bool

    show_plots: bool
    save_plots: bool

    load_config_fn: Callable[[Path], Optional[MjCambrianConfig]]
    load_eval_fitness_fn: Callable[[Path], float]
    load_train_fitness_fn: Callable[[Path], float]
    load_monitor_fn: Callable[[Path], np.ndarray]
    load_evaluations_fn: Callable[[Path], np.ndarray]

    combine_fn: Optional[Callable[[Concatenate[Path, ...]], "Result"]] = None

    ignore: List[str]
    overrides: Dict[str, Any]

    dry_run: bool


@config_wrapper
class Result:
    """The result of parsing a single training run."""

    config: Optional[MjCambrianConfig] = None
    train_fitness: Optional[float] = None
    eval_fitness: Optional[float] = None
    monitor: Optional[np.ndarray] = None
    evaluations: Optional[np.ndarray] = None


@config_wrapper
class Data:
    """The data collected from parsing the sweep."""

    results: Dict[str, Result]


# ==================


def load_config(
    folder: Path, overrides: Dict[str, Any], *, filename: str = "config.yaml", **kwargs
) -> MjCambrianConfig | None:
    if not (folder / filename).exists():
        return None

    kwargs.setdefault("instantiate", False)
    config = MjCambrianConfig.load(folder / filename, **kwargs)
    config.merge_with_dotlist([o for o in overrides])
    config.resolve()
    return config


# ==================


def load_fitness_from_monitor(folder: Path, *, filename: str) -> float:
    assert (monitor := folder / filename).exists(), f"Monitor {monitor} does not exist."

    from cambrian.ml.fitness_fns import fitness_from_monitor

    return fitness_from_monitor(None, monitor)


def load_fitness_from_txt(folder: Path, *, filename: str) -> float:
    assert (txt := folder / filename).exists(), f"Txt {txt} does not exist."
    return np.genfromtxt(txt)


# ==================


def load_monitor(folder: Path, *, filename: str) -> np.ndarray:
    assert (monitor := folder / filename).exists(), f"Monitor {monitor} does not exist."

    from cambrian.ml.fitness_fns import fitness_from_monitor

    _, (_, rewards) = fitness_from_monitor(None, monitor, return_data=True)
    return rewards


# ==================


class FitnessCombineMethod(Enum):
    """The method to use to combine the fitness results of a sweep."""

    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"


def combine(results: List[Result], *, fitness_method: str) -> Result:
    get_logger().info(f"Combining {len(results)} results...")

    result = {}
    fitness_method = FitnessCombineMethod[fitness_method]

    # config
    # TODO
    result["config"] = None
    for r in results:
        if r.config is not None:
            result["config"] = r.config
            break

    # fitness
    fitnesses = [r.eval_fitness for r in results if r.eval_fitness is not None]
    if fitness_method == FitnessCombineMethod.MEAN:
        fitness = np.mean(fitnesses)
    elif fitness_method == FitnessCombineMethod.MEDIAN:
        fitness = np.median(fitnesses)
    elif fitness_method == FitnessCombineMethod.MAX:
        fitness = np.max(fitnesses)
    elif fitness_method == FitnessCombineMethod.MIN:
        fitness = np.min(fitnesses)
    else:
        raise ValueError(f"Unknown fitness method {fitness_method}.")

    result["eval_fitness"] = fitness

    # train fitness
    train_fitnesses = [r.train_fitness for r in results if r.train_fitness is not None]
    if fitness_method == FitnessCombineMethod.MEAN:
        train_fitness = np.mean(train_fitnesses)
    elif fitness_method == FitnessCombineMethod.MEDIAN:
        train_fitness = np.median(train_fitnesses)
    elif fitness_method == FitnessCombineMethod.MAX:
        train_fitness = np.max(train_fitnesses)
    elif fitness_method == FitnessCombineMethod.MIN:
        train_fitness = np.min(train_fitnesses)
    else:
        raise ValueError(f"Unknown fitness method {fitness_method}.")

    result["train_fitness"] = train_fitness

    # monitor; stack the monitors
    monitor = None
    monitors = [r.monitor for r in results if r.monitor is not None]
    if len(monitors) != 0:
        min_length = min(len(m) for m in monitors)
        monitor = np.stack([m[:min_length] for m in monitors], axis=0)
        monitor = np.median(monitor, axis=0)

    result["monitor"] = monitor

    # evaluations; stack the evaluations
    evaluation = None
    evalutions = [r.evaluations for r in results if r.evaluations is not None]
    if len(evalutions) != 0:
        min_length = min(len(e) for e in evalutions)
        evaluation = np.stack([e[:min_length] for e in evalutions], axis=0)
        evaluation = np.median(evaluation, axis=0)

    result["evaluations"] = evaluation

    return Result(**result)


# ==================


def parse_folder(config: ParseSweepsConfig, folder: Path) -> Result | None:
    if folder.name in config.ignore or folder.parent.name in config.ignore:
        return None
    if not (folder / "finished").exists():
        return None

    def run(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if config.debug:
                raise Exception(f"Error running {fn}: {e}") from e
            get_logger().error(f"Error running {fn}: {e}")
            return None

    result = {}
    result["config"] = run(config.load_config_fn, folder, config.overrides)
    result["train_fitness"] = run(config.load_train_fitness_fn, folder)
    result["eval_fitness"] = run(config.load_eval_fitness_fn, folder)
    result["monitor"] = run(config.load_monitor_fn, folder)
    result["evaluations"] = run(config.load_evaluations_fn, folder)

    return Result(**result)


def load_folder(config: ParseSweepsConfig, folder: Path) -> Result | None:
    if not folder.is_dir():
        return None

    # Check if the folder is a singly-nested or doubly-nested folder
    result = None
    if (folder / "evaluations").exists():
        get_logger().info(f"Processing {folder}...")

        result = parse_folder(config, folder)
    else:
        # Recursively load the subfolders
        subresults = []
        for subfolder in sorted(folder.iterdir()):
            if (subresult := load_folder(config, subfolder)) is not None:
                subresults.append(subresult)

        # And combine them
        if subresults:
            assert config.combine_fn is not None, "No combine function provided."
            result = config.combine_fn(subresults)

    return result


def load_data(config: ParseSweepsConfig) -> Data:
    """It is assumed `config.folder` contains subfolders, each it's own training run.
    We'll parse each training run and return the data we can collect from it.

    The folder can be one of two types:
    1. A singly-nested folder where there is one folder with many training runs within
        it. This is parsed one by one and all plots/outputs are grouped together.
    2. A doubly-nested folder where there are many folders, each with many subfolders.
        Each folder within the subfolder is treated as a trial of a specific
        configuration. These results can then be grouped/averaged/etc. in some way.
    """
    results = {}

    for folder in sorted(config.folder.iterdir()):
        get_logger().debug(f"Loading {folder}...")
        if result := load_folder(config, folder):
            results[folder.name] = result

    return Data(results=results)


# ==================


def plot_data(config: ParseSweepsConfig, data: Data):
    # Clear all figures
    plt.close("all")

    # Sort the data by the suffix in the name
    def sort_key(x):
        # Find all numeric parts in the string
        num_part = re.findall(r"\d+", x[0])
        return (x[0].split("_")[-1], int(num_part[0]) if num_part else 0, x[0])

    data.results = dict(sorted(data.results.items(), key=sort_key))

    get_logger().info("Plotting data...")
    for name, result in tqdm.tqdm(
        data.results.items(), desc="Plotting...", disable=config.debug
    ):
        # Eval Fitness
        if result.eval_fitness is not None:
            plt.figure("Eval Fitness Bar Chart")
            plt.suptitle("Eval Fitness Bar Chart")
            plt.bar(name, result.eval_fitness, alpha=0.5)
            plt.xticks(rotation=90)

            plt.figure("Eval Fitness")
            plt.suptitle("Eval Fitness")
            plt.scatter(name, result.eval_fitness, marker="o")
            plt.annotate(
                name,
                (0, result.eval_fitness),
                textcoords="offset points",
                xytext=(10, 0),
                ha="left",
                rotation=90,
            )

        # Train Fitness
        if result.train_fitness is not None:
            plt.figure("Train Fitness")
            plt.suptitle("Train Fitness")
            plt.scatter(0, result.train_fitness, marker="o")
            plt.annotate(
                name,
                (0, result.train_fitness),
                textcoords="offset points",
                xytext=(10, 0),
                ha="left",
            )

        # Monitor
        if (monitor := result.monitor) is not None:
            plt.figure("Monitor")
            plt.suptitle("Monitor")

            if len(monitor) > 100:
                # convolve; window size 100
                monitor = np.convolve(monitor, np.ones(100) / 100, mode="valid")
            x, y = np.arange(len(monitor)), monitor
            plt.plot(x, y)
            plt.annotate(
                name,
                (x[-1], y[-1]),
                xytext=(10, 0),
                textcoords="offset points",
                verticalalignment="center",
                horizontalalignment="left",
            )

        # Evaluations
        if (evaluations := result.evaluations) is not None:
            plt.figure("Evaluations")
            plt.suptitle("Evaluations")

            evaluations = result.evaluations
            if len(evaluations) > 3:
                # convolve; window size 3
                evaluations = np.convolve(evaluations, np.ones(3) / 3, mode="valid")
            x, y = np.arange(len(evaluations)), evaluations
            plt.plot(x, y)
            plt.annotate(
                name,
                (x[-1], y[-1]),
                xytext=(10, 0),
                textcoords="offset points",
                verticalalignment="center",
                horizontalalignment="left",
            )

    if config.save_plots:
        plots_folder = config.output / "plots"
        plots_folder.mkdir(parents=True, exist_ok=True)
        for fig in plt.get_fignums():
            fig = plt.figure(fig)
            title = fig.get_suptitle()
            filename = title.lower().replace(" ", "_").replace("/", "_")

            get_logger().info(f"Saving {filename}...")
            plt.savefig(
                config.output / "plots" / f"{filename}.png",
                dpi=500,
                bbox_inches="tight",
                transparent=False,
            )

    if config.show_plots:
        plt.show()


# ==================


def main(config: ParseSweepsConfig):
    assert not (config.debug and config.quiet), "Cannot be both debug and quiet."
    if config.debug:
        get_logger().setLevel("DEBUG")
    elif config.quiet:
        get_logger().setLevel("WARNING")

    assert config.folder.exists(), f"Folder {config.folder} does not exist."

    if config.force or (data := try_load_pickle(config.output, config.pkl)) is None:
        data = load_data(config)
        save_data(data, config.output, config.pkl)

    if config.plot:
        plot_data(config, data)


if __name__ == "__main__":
    run_hydra(
        main,
        config_name="tools/parse_sweeps/parse_sweeps",
    )
