from typing import Dict, Any, Iterable
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil
import csv
import subprocess

import torch
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
    ProgressBarCallback,
)
from stable_baselines3.common.results_plotter import load_results, ts2xy
from hydra.experimental.callbacks import Callback as HydraCallback
from omegaconf import DictConfig

from cambrian.envs import MjCambrianEnv
from cambrian.ml.model import MjCambrianModel
from cambrian.utils import setattrs_temporary
from cambrian.utils.logger import get_logger


class MjCambrianPlotMonitorCallback(BaseCallback):
    """Should be used with an EvalCallback to plot the evaluation results.

    This callback will take the monitor.csv file produced by the VecMonitor and
    plot the results and save it as an image. Should be passed as the
    `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory where the evaluation results are stored. The
            evaluations.npz file is expected to be at `<logdir>/monitor.csv`. The
            resulting plot is going to be stored at
            `<logdir>/evaluations/monitor.png`.
    """

    parent: EvalCallback

    def __init__(self, logdir: Path | str):
        self.logdir = Path(logdir)
        self.evaldir = self.logdir / "evaluations"
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.n_calls = 0

    def _on_step(self) -> bool:
        if not (self.logdir / "monitor.csv").exists():
            get_logger().warning("No monitor.csv file found.")
            return

        x, y = ts2xy(load_results(self.logdir), "timesteps")
        if len(x) <= 100 or len(y) <= 100:
            get_logger().warning("Not enough monitor data to plot.")
            return True

        get_logger().info(f"Plotting monitor results at {self.evaldir}")

        def moving_average(values, window) -> np.ndarray:
            weights = np.repeat(1.0, window) / window
            return np.convolve(values, weights, "valid")

        y = moving_average(y.astype(float), window=min(len(y) // 10, 1000))
        x = x[len(x) - len(y) :]  # truncate x

        plt.plot(x, y)
        plt.fill_between(x, y - y.std() * 1.96, y + y.std() * 1.96, alpha=0.2)

        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.savefig(self.evaldir / "monitor.png")
        plt.cla()

        return True


class MjCambrianPlotEvaluationsCallback(BaseCallback):
    """Should be used with an EvalCallback to plot the evaluation results.

    This callback will take the evaluations.npz file produced by the EvalCallback and
    plot the results and save it as an image. Should be passed as the
    `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory where the evaluation results are stored. The
            evaluations.npz file is expected to be at `<logdir>/evaluations.npz`. The
            resulting plot is going to be stored at
            `<logdir>/evaluations/evaluations.png`.
    """

    parent: EvalCallback

    def __init__(self, logdir: Path | str):
        self.logdir = Path(logdir)
        self.evaldir = self.logdir / "evaluations"
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.n_calls = 0

    def _on_step(self) -> bool:
        if not (self.logdir / "evaluations.npz").exists():
            get_logger().warning("No evaluations.npz file found.")
            return

        # Load the evaluation results
        with np.load(self.logdir / "evaluations.npz") as data:
            x = data["timesteps"].flatten()
            y: np.ndarray = np.mean(data["results"], axis=1).flatten()
        if len(x) <= 3 or len(y) <= 3:
            get_logger().warning("Not enough evaluation data to plot.")
            return True

        get_logger().info(f"Plotting evaluation results at {self.evaldir}")

        # Plot the results
        plt.plot(x, y)
        plt.fill_between(x, y - y.std() * 1.96, y + y.std() * 1.96, alpha=0.2)

        plt.xlabel("Number of Timesteps")
        plt.ylabel("Evaluation Results")
        plt.savefig(self.evaldir / "evaluations.png")
        plt.cla()

        return True


class MjCambrianEvalCallback(EvalCallback):
    """Overwrites the default EvalCallback to support saving visualizations at the same
    time as the evaluation.

    NOTE: Only the first environment is visualized
    """

    def _init_callback(self):
        self.log_path = Path(self.log_path)
        self.n_evals = 0

        # Delete all the existing renders
        for f in glob.glob(str(self.log_path / "vis_*")):
            get_logger().info(f"Deleting {f}")
            Path(f).unlink()

        super()._init_callback()

    def _on_step(self) -> bool:
        # Early exit
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        env: MjCambrianEnv = self.eval_env.envs[0].unwrapped

        # Add some overlays
        # env.overlays["Exp"] = env.config.expname # TODO
        env.overlays["Best Mean Reward"] = f"{self.best_mean_reward:.2f}"
        env.overlays["Total Timesteps"] = f"{self.num_timesteps}"

        # Set temporary attributes for the evaluation
        temp_attrs = []
        temp_attrs.append((env, dict(record=True, maze=None)))

        # Run the evaluation
        with setattrs_temporary(*temp_attrs):
            get_logger().info(f"Starting {self.n_eval_episodes} evaluation run(s)...")
            continue_training = super()._on_step()

            # Save the visualization
            filename = Path(f"vis_{self.n_evals}")
            env.save(self.log_path / filename)

        # Copy the most recent gif to latest.gif so that we can just watch this file
        for f in self.log_path.glob(str(filename.with_suffix(".*"))):
            shutil.copy(f, f.with_stem("latest"))

        self.n_evals += 1
        return continue_training

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        env: MjCambrianEnv = self.eval_env.envs[0].unwrapped

        # If done, do some logging
        if locals_["done"]:
            run = locals_["episode_counts"][locals_["i"]]
            cumulative_reward = env.stashed_cumulative_reward
            get_logger().info(f"Run {run} done. Cumulative reward: {cumulative_reward}")

        super()._log_success_callback(locals_, globals_)


class MjCambrianGPUUsageCallback(BaseCallback):
    """This callback will log the GPU usage at the end of each evaluation.
    We'll log to a csv."""

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        logfile: Path | str = "gpu_usage.csv",
        *,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.logfile = self.logdir / logfile
        with open(self.logfile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timesteps",
                    "memory_reserved",
                    "max_memory_reserved",
                    "memory_available",
                ]
            )

    def _on_step(self) -> bool:
        if torch.cuda.is_available():
            # Get the GPU usage, log it and save it to the file
            device = torch.cuda.current_device()
            memory_reserved = torch.cuda.memory_reserved(device)
            max_memory_reserved = torch.cuda.max_memory_reserved(device)
            memory_available = torch.cuda.get_device_properties(device).total_memory

            # Log to the output file
            with open(self.logfile, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.num_timesteps,
                        memory_reserved,
                        max_memory_reserved,
                        memory_available,
                    ]
                )

            # Log to stdout
            if self.verbose > 0:
                get_logger().debug(subprocess.getoutput("nvidia-smi"))
                get_logger().debug(torch.cuda.memory_summary())

        return True


class MjCambrianSavePolicyCallback(BaseCallback):
    """Should be used with an EvalCallback to save the policy.

    This callback will save the policy at the end of each evaluation. Should be passed
    as the `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory to store the generated visualizations. The
            resulting visualizations are going to be stored at
            `<logdir>/evaluations/visualization.gif`.
    """

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        *,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.model: MjCambrianModel = None

    def _on_step(self) -> bool:
        self.model.save_policy(self.logdir)

        return True


class MjCambrianProgressBarCallback(ProgressBarCallback):
    """Overwrite the default progress bar callback to flush the pbar on deconstruct."""

    def __del__(self):
        """This string will restore the terminal back to its original state."""
        if hasattr(self, "pbar"):
            print("\x1b[?25h")


class MjCambrianCallbackListWithSharedParent(CallbackList):
    def __init__(self, callbacks: Iterable[BaseCallback] | Dict[str, BaseCallback]):
        if isinstance(callbacks, dict):
            callbacks = callbacks.values()

        self.callbacks = []
        super().__init__(list(callbacks))

    @property
    def parent(self):
        return getattr(self.callbacks[0], "parent", None)

    @parent.setter
    def parent(self, parent):
        for cb in self.callbacks:
            cb.parent = parent


# ==================


class MjCambrianSaveConfigCallback(HydraCallback):
    """This callback will save the resolved hydra config to the logdir."""

    def on_run_start(self, config: DictConfig, **kwargs):
        self._save_config(config)

    def on_multirun_start(self, config: DictConfig, **kwargs):
        self._save_config(config)

    def _save_config(self, config: DictConfig):
        from omegaconf import OmegaConf

        config.logdir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, config.logdir / "full.yaml")
