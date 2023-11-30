from typing import Optional, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import glob

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
    ProgressBarCallback,
)
from stable_baselines3.common.results_plotter import load_results, ts2xy

from env import MjCambrianEnv
from renderer import MjCambrianRenderer
from cambrian.evolution_envs.three_d.mujoco.animal_pool import MjCambrianAnimalPool
from cambrian.evolution_envs.three_d.mujoco.utils import evaluate_policy


class PlotEvaluationCallback(BaseCallback):
    """Should be used with an EvalCallback to plot the evaluation results.

    This callback will take the evaluations.npz file produced by the EvalCallback and
    plot the results and save it as an image. Should be passed as the
    `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory where the evaluation results are stored. The
            evaluations.npz file is expected to be at `<logdir>/evaluations.npz`. The
            resulting plot is going to be stored at
            `<logdir>/evaluations/monitor.png`.

    Keyword Args:
        verbose (int): The verbosity level. Defaults to 0.
    """

    parent: EvalCallback

    def __init__(self, logdir: Path | str, *, verbose: int = 0):
        self.logdir = Path(logdir)
        self.evaldir = self.logdir / "evaluations"
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.verbose = verbose
        self.n_calls = 0

    def _on_step(self) -> bool:
        if self.verbose > 0:
            print(f"Plotting evaluation results at {self.evaldir}")

        # Also save the monitor
        try:
            x, y = ts2xy(load_results(self.logdir), "timesteps")

            def moving_average(values, window):
                weights = np.repeat(1.0, window) / window
                return np.convolve(values, weights, "valid")

            y = moving_average(y.astype(float), window=min(len(y) // 10, 1000))
            x = x[len(x) - len(y) :]  # truncate x

            plt.plot(x, y)
            plt.xlabel("Number of Timesteps")
            plt.ylabel("Rewards")
            plt.savefig(self.evaldir / "monitor.png")
            plt.cla()
        except Exception as e:
            print(f"Couldn't save monitor: {e}.")

            # Going to try to corret. Possible error states that have been observed:
            #   1. the second line (the header) is missing a new line after `r,l,t`
            with open(self.logdir / "monitor.csv", "r+") as f:
                lines = f.readlines()
                if lines[1][len("r,l,t") :] != "\n":
                    line1, line2 = lines[1][: len("r,l,t")], lines[1][len("r,l,t") :]
                    del lines[1]
                    lines.insert(1, line1 + "\n")
                    lines.insert(2, line2)
                    f.seek(0)
                    f.writelines(lines)

            print("Corrected monitor.csv.")

        return True


class SaveVideoCallback(BaseCallback):
    """Should be used with an EvalCallback to visualize the environment.

    This callback will save a visualization of the environment at the end of each
    evaluation. Should be passed as the `callback_after_eval` for the EvalCallback.

    NOTE: Only the first environment is visualized

    Args:
        logdir (Path | str): The directory to store the generated visualizations. The
            resulting visualizations are going to be stored at
            `<logdir>/evaluations/visualization.gif`.
    """

    parent: EvalCallback

    def __init__(
        self,
        env: DummyVecEnv,
        logdir: Path | str,
        max_episode_steps: int,
        *,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.env = env.envs[0]
        self.cambrian_env: MjCambrianEnv = self.env.unwrapped
        self.renderer: MjCambrianRenderer = self.cambrian_env.renderer

        self.logdir = Path(logdir)
        self.evaldir = self.logdir / "evaluations"
        self.evaldir.mkdir(parents=True, exist_ok=True)

        # Delete all the existing gifs
        for f in glob.glob(str(self.evaldir / "vis_*")):
            if self.verbose > 0:
                print(f"Deleting {f}")
            Path(f).unlink()

        self.max_episode_steps = max_episode_steps

    def _on_step(self) -> bool:
        best_mean_reward = self.parent.best_mean_reward
        self.cambrian_env.overlays["Best Mean Reward"] = f"{best_mean_reward:.2f}"
        self.cambrian_env.overlays["Total Timesteps"] = f"{self.num_timesteps}"

        filename = f"vis_{self.n_calls}"
        evaluate_policy(
            self.env, self.parent.model, 1, record_path=self.evaldir / filename
        )

        return True


class MjCambrianAnimalPoolCallback(BaseCallback):
    parent: EvalCallback

    def __init__(
        self,
        env: DummyVecEnv,
        animal_pool: MjCambrianAnimalPool,
        *,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.env = env.envs[0]
        self.cambrian_env: MjCambrianEnv = self.env.unwrapped
        self.animal_pool = animal_pool

    def _on_step(self) -> bool:
        if self.parent is not None:
            self.animal_pool.write_to_pool(
                self.parent.best_mean_reward, self.cambrian_env.config.copy()
            )


class MjCambrianProgressBarCallback(ProgressBarCallback):
    """Overwrite the default progress bar callback to flush the pbar on deconstruct."""

    def __init__(self):
        super().__init__()

    def __del__(self):
        """This string will restore the terminal back to its original state."""
        print("\x1b[?25h")
        self._on_training_end()


class CallbackListWithSharedParent(CallbackList):
    def __init__(self, *args, **kwargs):
        self.callbacks = []
        super().__init__(*args, **kwargs)

    @property
    def parent(self):
        return getattr(self.callbacks[0], "parent", None)

    @parent.setter
    def parent(self, parent):
        for cb in self.callbacks:
            cb.parent = parent
