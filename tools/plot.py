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


def plot_results(args):
    """
    plot the results

    :param log_folders: (list) the list of log folders to plot
    :param name: (str) the path to save the plot
    :param window: (int) the moving average window size
    """
    fig = plt.figure()
    for log_folder in args.log_folders:
        print(f"Plotting {log_folder}")
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        y = moving_average(y.astype(float), window=min(len(y) // 10, 1000))
        x = x[len(x) - len(y) :]  # truncate x
        label = log_folder.split('/')[-1]
        if args.shorten_label:
            label = '...' + label[:19] if len(label) > 20 else label
            # label = '...' + label[19:] if len(label) > 20 else label
        plt.plot(x, y, label=label)
        if args.fill_between:
            plt.fill_between(x, y - y.std() * 1.96, y + y.std() * 1.96, alpha=0.2)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(f'{args.name.split(".")[0]}')
    plt.legend(loc='best')
    plt.savefig(args.name)
    print(f"Plot saved to {args.name}")


if __name__ == "__main__":
    import os 
    parser = argparse.ArgumentParser(description='Plot results from multiple log folders')
    parser.add_argument('--parent_folder', '-p', type=str, help='Parent folder', 
                         default=None)
    parser.add_argument('--log_folders', '-l', nargs='+', help='List of log folders to plot', 
                        default=[])
    parser.add_argument('--name', help='Path to save the plot')
    parser.add_argument('--fill_between', action='store_true', help='Fill between the curves')
    parser.add_argument('--shorten_label', action='store_true', help='Shorten Label to the first 20 characters')
    args = parser.parse_args()

    # parent and log folders are mutually exclusive
    assert not (args.parent_folder and args.log_folders), "Only one of parent_folder and log_folders can be set"

    if args.parent_folder:
        # get all directories in parent_folder
        args.log_folders = [f"{args.parent_folder}/{d}" for d in os.listdir(args.parent_folder) if os.path.isdir(f"{args.parent_folder}/{d}")]

    plot_results(args)

    """
    python /home/tools/plot.py \
        -l logs-mujoco/60s_40ap_3eyes_2x2_aperture_1 \
           logs-mujoco/60s_40ap_3eyes_2x2_aperture_dot2 \
           logs-mujoco/60s_40ap_3eyes_2x2_aperture_dot5 \
           logs-mujoco/optic_false_3eyes_2x2 \
        --name 3eyes_60scene_40apres_2x2.png
    """
    
