"""This script will a saved evolution folder."""

import argparse
from typing import Dict, Union
from pathlib import Path
import pickle
import os
import yaml
from prodict import Prodict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from collections import defaultdict

from stable_baselines3.common.results_plotter import load_results, ts2xy

from plot import moving_average

mpl.rcParams['image.cmap'] = 'jet'


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


def get_generation_file_paths(folder: Path) -> Dict:
    """Create the initial storage dict for parsing and get the paths to all the generation/rank folders."""
    generations = dict()
    for root, dirs, files in os.walk(folder):
        root = Path(root)
        if not root.stem.startswith("generation_"):
            continue

        epoch = int(root.stem.split("generation_", 1)[1])

        ranks = dict()
        for dir in dirs:
            dir = Path(dir)
            if not dir.stem.startswith("rank_"):
                continue

            rank = int(dir.stem.split("rank_", 1)[1])
            ranks[rank] = dict(path=root / dir, epoch=epoch, rank=rank)

        ranks = dict(sorted(ranks.items()))
        generations[epoch] = dict(path=root, ranks=ranks)

    generations = dict(sorted(generations.items()))

    return generations


def load_data(folder: Path) -> Dict:
    print(f"Loading data from {folder}...")
    generations = get_generation_file_paths(folder)

    for epoch, data in generations.items():
        print(f"Loading epoch {epoch}...")

        ranks = data["ranks"]
        for rank, rank_data in ranks.items():
            print(f"\tLoading rank {rank}...")

            path = rank_data["path"]

            # Get the config file
            config_file = path / "config.yml"
            assert config_file.exists()
            with open(config_file, "r") as ymlfile:
                config = yaml.load(ymlfile, Loader=yaml.Loader)
                config = Prodict.from_dict(config)
                rank_data["config"] = config

            # Get the evaluations file
            evaluations_file = path / "evaluations.npz"
            assert evaluations_file.exists()
            with np.load(evaluations_file) as evaluations_data:
                evaluations = {k: evaluations_data[k] for k in evaluations_data.files}
            rank_data["evaluations"] = evaluations

            # Get the monitor file
            monitor_folder = path / "ppo"
            assert (monitor_folder / "monitor.csv").exists()
            monitor = load_results(monitor_folder)
            rank_data["monitor"] = monitor

    return generations


def plot(generations: Dict, output_folder: Path) -> Dict:
    print("Parsing data...")

    RANK_FORMAT_MAP = {
        0: ".C0",
        1: ".C1",
        2: "sC2",
        3: "^C3",
        4: "xC4",
        5: "vC5",
        6: "8C6",
    }

    def _plot(
        *args, title, xlabel, ylabel, label=None, locator=tkr.MultipleLocator(), **kwargs
    ):
        fig = plt.figure(title)

        plt.plot(*args, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.suptitle(title)

        if label is not None:
            if not hasattr(fig, "labels"):
                fig.labels = set()
            fig.labels.add(label)
            plt.legend(sorted(fig.labels))

        ax = fig.gca()
        ax.xaxis.set_major_locator(locator)

    output_folder.mkdir(parents=True, exist_ok=True)
    for epoch, data in generations.items():
        print(f"Parsing epoch {epoch}...")
        data.setdefault("config", defaultdict(list))
        data.setdefault("evaluations", dict())
        data.setdefault("monitor", dict())

        ranks = data["ranks"]
        for rank, rank_data in ranks.items():
            print(f"\tParsing rank {rank}...")

            # ======
            # CONFIG
            # ======
            config = rank_data["config"]
            animal_config = config.animal_config
            init_config = animal_config.init_configuration

            # Plot the rank-over-rank config data
            def _config_plot(attr, *args, **kwargs):
                y = getattr(init_config, attr)
                y = np.average(np.abs(y)) if isinstance(y, list) else y
                _plot(
                    epoch,
                    y,
                    f"{RANK_FORMAT_MAP[rank]}",
                    *args,
                    label=f"Rank {rank}",
                    title=attr,
                    xlabel="epoch",
                    ylabel=attr,
                    **kwargs,
                )

            _config_plot("num_pixels")
            _config_plot("angle")
            _config_plot("fov")
            _config_plot("sensor_size")
            _config_plot("closed_pinhole_percentage")

            # Accumulate the data from all ranks
            data["config"]["num_pixels"].append(init_config.num_pixels)
            data["config"]["angle"].append(np.average(np.abs(init_config.angle)))
            data["config"]["fov"].append(np.average(np.abs(init_config.fov)))
            data["config"]["sensor_size"].append(np.average(init_config.sensor_size))
            data["config"]["closed_pinhole_percentage"].append(
                np.average(init_config.closed_pinhole_percentage)
            )

            # ======
            # EVALS
            # ======
            # TODO: we should try to see how performance improves over the course of training
            evaluations = rank_data["evaluations"]
            _plot(
                epoch,
                np.average(evaluations["results"]),
                f"{RANK_FORMAT_MAP[rank]}",
                label=f"Rank {rank}",
                title="average_eval_rewards",
                xlabel="epoch",
                ylabel="rewards",
            )

            # Get the data and delete it from the dict. We'll write the parsed data back
            # under the same key.
            monitor = rank_data["monitor"]
            x, y = ts2xy(monitor, "timesteps")
            y = moving_average(y.astype(float), window=1000)
            x = x[len(x) - len(y) :]

            # TODO: this looks terrible
            _plot(
                x,
                y,
                f"-{RANK_FORMAT_MAP[rank][-2:]}",
                label=f"Rank {rank}",
                title="monitor",
                xlabel="timesteps",
                ylabel="rewards",
                locator=tkr.AutoLocator(),
            )

    for fig in plt.get_fignums():
        fig = plt.figure(fig)
        plt.savefig(output_folder / f"{fig._suptitle.get_text()}.png", dpi=300)


def main(args):
    folder = Path(args.folder)
    output_folder = (
        folder / "parse_evos" / "plots" if args.output is None else Path(args.output)
    )

    if args.force or (generations := try_load_pickle_data(folder)) is None:
        generations = load_data(folder)

        if not args.no_save:
            save_data(generations, folder)

    plot(generations, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the evolution folder.")

    parser.add_argument("folder", type=str, help="The folder to parse.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output folder. Defaults to <folder>/parse_evos/plots/",
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force loading of the data. If not passed, this script will try to find a parse_evos.pkl file and load that instead.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save the parsed data."
    )

    args = parser.parse_args()

    main(args)
