from typing import Dict, Any, Optional, Callable, Concatenate, List
from pathlib import Path
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

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

    force (bool): Force loading of the data. If not passed, this script will try to
        find a parse_sweep.pkl file and load that instead.
    plot (bool): Plot the data.
    save (bool): Save the data to the output folder.
    quiet (bool): Quiet mode. Set's the logger to warning.
    debug (bool): Debug mode. Set's the logger to debug and disables tqdm.

    overrides (Dict[str, Any]): Overrides for the sweep data.

    dry_run (bool): Do not actually do any of the processing, just run the code without
        that part.
    """

    folder: Path
    output: Path

    force: bool
    plot: bool
    quiet: bool
    debug: bool

    load_config_fn: Callable[[Path], Optional[MjCambrianConfig]]
    load_fitness_fn: Callable[[Path], float]
    load_monitor_fn: Callable[[Path], np.ndarray]
    load_evaluations_fn: Callable[[Path], np.ndarray]

    combine_fn: Optional[Callable[[Concatenate[Path, ...]], "Result"]] = None

    overrides: Dict[str, Any]

    dry_run: bool


@config_wrapper
class Result:
    """The result of parsing a single training run."""

    config: Optional[MjCambrianConfig] = None
    fitness: Optional[float] = None
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


def load_fitness_from_monitor(
    folder: Path, *, filename: str = "eval_monitor.csv"
) -> float:
    assert (monitor := folder / filename).exists(), f"Monitor {monitor} does not exist."

    from cambrian.ml.fitness_fns import fitness_from_monitor

    return fitness_from_monitor(None, monitor)


# ==================


def load_monitor(folder: Path, *, filename: str = "monitor.csv") -> np.ndarray:
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

    # config
    # TODO
    if all(r.config is not None for r in results):
        result["config"] = results[0].config

    # fitness
    fitnesses = [r.fitness for r in results]
    fitness_method = FitnessCombineMethod[fitness_method]
    if all(f is not None for f in fitnesses):
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

        result["fitness"] = fitness

    # monitor; stack the monitors
    if all(r.monitor is not None for r in results):
        min_length = min(len(r.monitor) for r in results)
        monitor = np.stack([r.monitor[:min_length] for r in results], axis=0)

        result["monitor"] = np.median(monitor, axis=0)

    # evaluations; stack the evaluations
    if all(r.evaluations is not None for r in results):
        min_length = min(len(r.evaluations) for r in results)
        evalutions = np.stack([r.evaluations[:min_length] for r in results], axis=0)

        result["evaluations"] = np.median(evalutions, axis=0)

    return Result(**result)


# ==================


def parse_folder(config: ParseSweepsConfig, folder: Path) -> Result | None:
    exp_config = config.load_config_fn(folder, config.overrides)
    fitness = config.load_fitness_fn(folder)
    monitor = config.load_monitor_fn(folder)
    evaluations = config.load_evaluations_fn(folder)

    return Result(
        config=exp_config, fitness=fitness, monitor=monitor, evaluations=evaluations
    )


def load_folder(config: ParseSweepsConfig, folder: Path) -> Result | None:
    if not folder.is_dir():
        return None

    # Check if the folder is a singly-nested or doubly-nested folder
    result = None
    if (folder / "best_model.zip").exists():
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
        if result := load_folder(config, folder):
            results[folder.name] = result

    return Data(results=results)


# ==================


def plot_data(config: ParseSweepsConfig, data: Data):
    for name, result in data.results.items():
        # Fitness
        plt.figure("Fitness")
        plt.scatter(0, result.fitness, marker="o")
        plt.annotate(
            name,
            (0, result.fitness),
            textcoords="offset points",
            xytext=(10, 0),
            ha="left",
        )

        # Monitor
        plt.figure("Monitor")

        monitor = result.monitor
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
        plt.figure("Evaluations")

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

    plt.show()


# ==================


def main(config: ParseSweepsConfig):
    assert not (config.debug and config.quiet), "Cannot be both debug and quiet."
    if config.debug:
        get_logger().setLevel("DEBUG")
    elif config.quiet:
        get_logger().setLevel("WARNING")

    assert config.folder.exists(), f"Folder {config.folder} does not exist."

    if config.force or (data := try_load_pickle(config.output, "data.pkl")) is None:
        data = load_data(config)
        save_data(data, config.output, "data.pkl")

    if config.plot:
        plot_data(config, data)


if __name__ == "__main__":
    run_hydra(
        main,
        config_name="tools/parse_sweeps/parse_sweeps",
    )
