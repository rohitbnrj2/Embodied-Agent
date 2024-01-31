import argparse
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


def plot_results(log_folders, name):
    """
    plot the results

    :param log_folders: (list) the list of log folders to plot
    :param name: (str) the path to save the plot
    :param window: (int) the moving average window size
    """
    fig = plt.figure()
    for log_folder in log_folders:
        print(f"Plotting {log_folder}")
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        y = moving_average(y.astype(float), window=min(len(y) // 10, 1000))
        x = x[len(x) - len(y) :]  # truncate x
        label = log_folder.split('/')[-1]
        label = '...' + label[15:] if len(label) > 20 else label
        plt.plot(x, y, label=label)
        plt.fill_between(x, y - y.std() * 1.96, y + y.std() * 1.96, alpha=0.2)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(f'{name.split(".")[0]}')
    plt.legend(loc='best')
    plt.savefig(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot results from multiple log folders')
    parser.add_argument('--log_folders', '-l', nargs='+', help='List of log folders to plot')
    parser.add_argument('--name', help='Path to save the plot')
    args = parser.parse_args()

    plot_results(args.log_folders, args.name)

    """
    python /home/tools/plot.py \
        -l logs-mujoco/60s_40ap_3eyes_2x2_aperture_1 \
           logs-mujoco/60s_40ap_3eyes_2x2_aperture_dot2 \
           logs-mujoco/60s_40ap_3eyes_2x2_aperture_dot5 \
           logs-mujoco/optic_false_3eyes_2x2 \
        --name 3eyes_60scene_40apres_2x2.png
    """
    
