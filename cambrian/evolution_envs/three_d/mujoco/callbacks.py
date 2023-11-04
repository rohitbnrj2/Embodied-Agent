from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

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
                return np.convolve(values, weights, 'valid')
            y = moving_average(y.astype(float), window=min(len(y)//10, 1000))
            x = x[len(x) - len(y):] # truncate x

            plt.plot(x, y)
            plt.xlabel('Number of Timesteps')
            plt.ylabel('Rewards')
            plt.savefig(self.evaldir / "monitor.png")
            plt.cla()
        except Exception as e: 
            print(f"Couldn't save monitor: {e}.")