import time
from typing import Dict

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from tqdm import tqdm

from cambrian.agents import MjCambrianAgent
from cambrian.eyes.multi_eye import MjCambrianMultiEyeConfig
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.config import MjCambrianConfig, run_hydra
from cambrian.utils.logger import get_logger


def sweep_eye_parameters(config: MjCambrianConfig):
    num_eyes_values = np.linspace(1, 10, 10, dtype=int)
    resolution_values = np.linspace(1, 100, 10, dtype=int)

    time_per_step_num_eyes = []
    time_per_step_resolution = []

    pbar = tqdm(total=len(num_eyes_values) * len(resolution_values))
    for num_eyes in num_eyes_values:
        for resolution in resolution_values:
            pbar.set_description(f"num_eyes={num_eyes}, resolution={resolution}")
            pbar.update(1)

            # Modify the eye configuration for the sweep
            with config.set_readonly_temporarily(False):
                eye_config: MjCambrianMultiEyeConfig = config.env.agents["agent"].eyes[
                    "eye"
                ]
                eye_config.num_eyes = [1, int(num_eyes)]
                eye_config.resolution = [int(resolution), int(resolution)]

            agents: Dict[str, MjCambrianAgent] = {}
            for name, agent_config in config.env.agents.items():
                agents[name] = agent_config.instance(agent_config, name)

            xml = MjCambrianXML.from_string(config.env.xml)
            for agent in agents.values():
                xml += agent.generate_xml()

            model = mj.MjModel.from_xml_string(xml.to_string())
            data = mj.MjData(model)
            mj.mj_step(model, data)

            for name, agent in agents.items():
                agents[name].reset(model, data)

            start_time = time.time()

            for _ in range(config.env.max_episode_steps):
                for agent in agents.values():
                    agent.step()

            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / config.env.max_episode_steps

            time_per_step_num_eyes.append((num_eyes, time_per_step))
            time_per_step_resolution.append((resolution, time_per_step))

    np.save(config.expdir / "time_per_step_num_eyes.npy", time_per_step_num_eyes)
    np.save(config.expdir / "time_per_step_resolution.npy", time_per_step_resolution)

    return time_per_step_num_eyes, time_per_step_resolution


def plot_results(
    config, time_per_step_num_eyes, time_per_step_resolution, *, log: bool = False
):
    num_eyes_values, times_num_eyes = zip(*time_per_step_num_eyes)
    resolution_values, times_resolution = zip(*time_per_step_resolution)
    unique_resolutions = sorted(set(resolution_values))
    unique_num_eyes = sorted(set(num_eyes_values))

    # Prepare FPS data
    fps_num_eyes = [1 / time for time in times_num_eyes]
    fps_resolution = [1 / time for time in times_resolution]

    # Reshape data for plotting lines between like-trials
    num_eyes_vs_fps = {res: [] for res in unique_resolutions}
    resolution_vs_fps = {eyes: [] for eyes in unique_num_eyes}

    for num_eyes, resolution, fps in zip(
        num_eyes_values, resolution_values, fps_num_eyes
    ):
        num_eyes_vs_fps[resolution].append((num_eyes, fps))
    for num_eyes, resolution, fps in zip(
        num_eyes_values, resolution_values, fps_resolution
    ):
        resolution_vs_fps[num_eyes].append((resolution, fps))

    # Sort data by num_eyes and resolution for consistent line plotting
    for resolution in num_eyes_vs_fps:
        num_eyes_vs_fps[resolution].sort()
    for num_eyes in resolution_vs_fps:
        resolution_vs_fps[num_eyes].sort()

    # Plot num_eyes vs fps with resolution as markers
    plt.figure(figsize=(10, 6))
    for resolution, data in num_eyes_vs_fps.items():
        num_eyes, fps = zip(*data)
        plt.plot(
            num_eyes,
            fps,
            marker="o",
            label=f"Resolution {int(resolution)}x{int(resolution)}",
        )
    plt.xlabel("Number of Eyes")
    plt.ylabel("FPS")
    plt.title("Number of Eyes vs FPS (with Resolution)")
    if log:
        plt.yscale("log")
    plt.legend()
    plt.grid(True)

    plt.savefig(config.expdir / "num_eyes_vs_fps.png")

    # Plot resolution vs fps with num_eyes as markers
    plt.figure(figsize=(10, 6))
    for num_eyes, data in resolution_vs_fps.items():
        resolutions, fps = zip(*data)
        plt.plot(resolutions, fps, marker="o", label=f"Number of Eyes {int(num_eyes)}")
    plt.xlabel("Resolution (WxH)")
    plt.ylabel("FPS")
    plt.title("Resolution vs FPS (with Number of Eyes)")
    if log:
        plt.yscale("log")
    plt.legend()
    plt.grid(True)

    plt.savefig(config.expdir / "resolution_vs_fps.png")


def main(config: MjCambrianConfig, *, load: bool, log: bool):
    if load:
        assert (config.expdir / "time_per_step_num_eyes.npy").exists() and (
            config.expdir / "time_per_step_resolution.npy"
        ).exists(), "Need to run the sweep first to generate data."
        time_per_step_num_eyes = np.load(config.expdir / "time_per_step_num_eyes.npy")
        time_per_step_resolution = np.load(
            config.expdir / "time_per_step_resolution.npy"
        )
    else:
        if any(
            f
            for f in config.expdir.iterdir()
            if f.name not in ["evaluations", "hydra", "logs", "full.yaml"]
        ):
            get_logger().warning(
                f"Experiment directory {config.expdir} already exists."
                " Is that okay? [Y]es/[N]o: ",
            )
            response = input().strip()
            if response.lower() not in ["", "y", "yes"]:
                get_logger().info("Exiting.")
                return
        time_per_step_num_eyes, time_per_step_resolution = sweep_eye_parameters(config)

    plot_results(config, time_per_step_num_eyes, time_per_step_resolution, log=log)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--log", action="store_true")

    run_hydra(main, parser=parser)
