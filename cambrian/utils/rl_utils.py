import os

import gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
# from plot import plot_results 

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        monitor_csv_path = Path(self.log_dir).joinpath('monitor.csv')
        # sometimes it doesn't create a monitor.csv! 
        if not monitor_csv_path.exists():
          open(monitor_csv_path, 'a').close()
            
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.prev_norm = 0. 

    def _sanity_check_norm(self, net):
      for name, param in net.named_parameters():
      #     print(name)
          if name == "linear_relu_stack.perception_head/linear1.weight":
              return torch.norm(param)
      return None

    def _on_training_start(self) -> None:
        self.envs = self.training_env

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                    try: 
                      sim = self.envs.get_attr('sim') # they should all be the same! 
                      if isinstance(sim, list):
                        sim = sim[0] # take the first env's sim 

                      if sim.mode == 'test':
                        print("Visualizing rollout... at {}".format(self.n_calls))
                        st = time.time()
                        self.sim.visualize_rollout(overwrite_path="best_model_{}".format(self.n_calls))
                        tt = time.time()-st
                        print("Visualization Time {}".format(tt))

                    except Exception as e: 
                      print("Couldn't visualize rollout: {}".format(e))

        return True
class PlotRewardCallback(BaseCallback):
    """
    Callback for Plotting a graph (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.plot_path = "{}/plot.png".format(self.log_dir)
        print("Saving Reward Plot at {}".format(self.plot_path))


    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          print("------------- Callback for Plotting a graph at {} -------------".format(self.n_calls))
          print("Experiment Folder: {}".format(self.log_dir))
          # Retrieve training reward
          # import pdb; pdb.set_trace()
          try: 
            pass 
          except Exception as e : 
            print("Couldn't save perception model becasue of {} ".format(e))
            raise e

        return True