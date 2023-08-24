import sys
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from matplotlib import pyplot as plt

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, name, window=1000):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y.astype(float), window=window)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title('/'.join(log_folder.split('/')[1:3]))
    plt.savefig(name)


if __name__ == "__main__":
    log_dir = sys.argv[1]
    name = sys.argv[2]
    window = sys.argv[3] 
    plot_results(log_dir, name, int(window))
