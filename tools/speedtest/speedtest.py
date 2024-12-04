import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import psutil
from scipy.stats import sem, t

from cambrian.ml.trainer import MjCambrianTrainer
from cambrian.utils.config import MjCambrianConfig, run_hydra
from cambrian.utils.logger import get_logger

num_eyes_sweep = np.arange(1, 100, 10).tolist()
resolution_sweep = np.arange(1, 100, 10).tolist()
num_samples = 5  # Number of runs per configuration


def main(config: MjCambrianConfig, *, folder_path: Optional[Path] = None):
    if folder_path is not None:
        timing_data = np.load(folder_path / "timing_data.npy", allow_pickle=True)
        ram_usage_data = np.load(folder_path / "ram_usage_data.npy", allow_pickle=True)
        plot_data(config, timing_data, ram_usage_data)
        return

    # Data storage
    timing_data = []
    ram_usage_data = []

    def run(num_eyes: int, resolution: int, config: MjCambrianConfig):
        with (
            config.set_readonly_temporarily(False),
            config.set_struct_temporarily(False),
        ):
            config.merge_with_dotlist(
                [
                    f"env.agents.agent.eyes.eye.num_eyes[0]={num_eyes}",
                    f"env.agents.agent.eyes.eye.resolution=[{resolution},{resolution}]",
                    f"eval_env.agents.agent.eyes.eye.num_eyes[0]={num_eyes}",
                    f"eval_env.agents.agent.eyes.eye.resolution=[{resolution},{resolution}]",  # noqa
                ]
            )
        trainer = MjCambrianTrainer(config)
        start_time = time.time()
        process = psutil.Process()
        initial_ram = process.memory_info().rss
        trainer.eval(record=False)
        elapsed_time = time.time() - start_time
        ram_used = process.memory_info().rss - initial_ram
        return elapsed_time, ram_used / (1024**2)  # Convert to MB

    # Collect timing and RAM data with multiple samples
    for num_eyes in num_eyes_sweep:
        for resolution in resolution_sweep:
            samples = [run(num_eyes, resolution, config) for _ in range(num_samples)]
            avg_time = np.mean([s[0] for s in samples])
            avg_ram = np.mean([s[1] for s in samples])
            confidence_interval = (
                t.ppf(0.975, df=num_samples - 1) * sem([s[0] for s in samples])
                if num_samples > 1
                else 0
            )
            timing_data.append((num_eyes, resolution, avg_time, confidence_interval))
            ram_usage_data.append((num_eyes, resolution, avg_ram))

            get_logger().info(
                f"num_eyes={num_eyes}, resolution={resolution}, "
                f"time={avg_time:.3f} ± {confidence_interval:.3f}, RAM={avg_ram:.2f} MB"
            )

    # Convert data to structured arrays
    timing_data = np.array(
        timing_data,
        dtype=[
            ("num_eyes", int),
            ("resolution", int),
            ("time", float),
            ("confidence_interval", float),
        ],
    )
    ram_usage_data = np.array(
        ram_usage_data,
        dtype=[
            ("num_eyes", int),
            ("resolution", int),
            ("avg_ram", float),
        ],
    )

    # Save data to files
    np.save(config.expdir / "timing_data.npy", timing_data)
    np.save(config.expdir / "ram_usage_data.npy", ram_usage_data)

    # Plotting
    plot_data(config, timing_data, ram_usage_data)


def plot_data(config: MjCambrianConfig, timing_data, ram_usage_data):
    # Delete the data from the first resolution/num_eyes since it's skewed by caching
    timing_data = timing_data[timing_data["num_eyes"] != timing_data["num_eyes"][0]]
    timing_data = timing_data[timing_data["resolution"] != timing_data["resolution"][0]]
    ram_usage_data = ram_usage_data[
        ram_usage_data["resolution"] != ram_usage_data["resolution"][0]
    ]
    ram_usage_data = ram_usage_data[
        ram_usage_data["num_eyes"] != ram_usage_data["num_eyes"][0]
    ]

    # Extract unique values for num_eyes and resolution
    num_eyes_values = np.unique(timing_data["num_eyes"])
    resolution_values = np.unique(timing_data["resolution"])

    # Plot 1: resolution vs time for different num_eyes
    plt.figure()
    for num_eyes in num_eyes_values:
        subset = timing_data[timing_data["num_eyes"] == num_eyes]
        plt.plot(subset["resolution"], subset["time"], label=f"num_eyes={num_eyes}")
        plt.fill_between(
            subset["resolution"],
            subset["time"] - subset["confidence_interval"],
            subset["time"] + subset["confidence_interval"],
            alpha=0.2,
        )
    plt.xlabel("Resolution")
    plt.ylabel("Time (s)")
    plt.title("Resolution vs Time (varying num_eyes)")
    plt.legend()
    plt.grid()
    plt.savefig(config.expdir / "resolution_vs_time_with_ci.png")

    # Plot 2: num_eyes vs time for different resolutions
    plt.figure()
    for resolution in resolution_values:
        subset = timing_data[timing_data["resolution"] == resolution]
        plt.plot(subset["num_eyes"], subset["time"], label=f"resolution={resolution}")
        plt.fill_between(
            subset["num_eyes"],
            subset["time"] - subset["confidence_interval"],
            subset["time"] + subset["confidence_interval"],
            alpha=0.2,
        )
    plt.xlabel("Num Eyes")
    plt.ylabel("Time (s)")
    plt.title("Num Eyes vs Time (varying resolution)")
    plt.legend()
    plt.grid()
    plt.savefig(config.expdir / "num_eyes_vs_time_with_ci.png")

    # Plot 3: RAM vs CI
    plt.figure()
    for resolution in resolution_values:
        subset = ram_usage_data[ram_usage_data["resolution"] == resolution]
        ci_subset = timing_data[timing_data["resolution"] == resolution]
        plt.plot(
            subset["avg_ram"],
            ci_subset["confidence_interval"],
            label=f"resolution={resolution}",
        )
    plt.xlabel("RAM Usage (MB)")
    plt.ylabel("Confidence Interval (s)")
    plt.title("RAM Usage vs Confidence Interval")
    plt.legend()
    plt.grid()
    plt.savefig(config.expdir / "ram_vs_ci.png")

    print("Plots saved: resolution_vs_time_with_ci.png, ram_vs_ci.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder-path",
        type=Path,
        help="Path to the folder where the data will be saved",
    )

    run_hydra(main, parser=parser)
