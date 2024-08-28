from typing import Tuple, TYPE_CHECKING, Dict
from pathlib import Path
import csv

import numpy as np

if TYPE_CHECKING:
    from cambrian.utils.config import MjCambrianConfig


def parse_evaluations_npz(evaluations_npz: Path) -> Dict[str, np.ndarray]:
    """Parse the evaluations npz file and return the rewards."""
    assert (
        evaluations_npz.exists()
    ), f"Evaluations file {evaluations_npz} does not exist."
    data = np.load(evaluations_npz, allow_pickle=True)
    return {k: data[k] for k in data}


def parse_monitor_csv(monitor_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse the monitor csv file and return the timesteps and rewards."""
    assert monitor_csv.exists(), f"Monitor file {monitor_csv} does not exist."
    timesteps, rewards = [], []
    with open(monitor_csv, "r") as f:
        # Skip the comment line
        f.readline()

        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            timesteps.append(float(row["t"]))
            rewards.append(float(row["r"]))

    return np.array(timesteps), np.array(rewards)


def fitness_from_evaluations(
    config: "MjCambrianConfig", evaluations_npz: Path, *, return_data: bool = False
) -> float | Tuple[float, np.ndarray]:
    """Calculate the fitness of the agent. This is done by taking the 3rd quartile of
    the evaluation rewards."""
    # Return negative infinity if the evaluations file doesn't exist
    if not evaluations_npz.exists():
        return -float("inf")

    def top_25_excluding_outliers(data: np.ndarray) -> float:
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        filtered_data = data[(data > q1 - 1.5 * iqr) & (data < q3 + 1.5 * iqr)]
        return float(np.mean(np.sort(filtered_data)[-len(filtered_data) // 4 :]))

    # The rewards array will be stored in a 2D array where each row represents each
    # evaluation run and each column represents the rewards for each evaluation step.
    # We may run multiple steps of the same or slightly different environment to reduce
    # variance. We will average the rewards across each row to get the final rewards.
    evaluations = parse_evaluations_npz(evaluations_npz)
    rewards = evaluations["results"]
    rewards = np.mean(rewards, axis=1)

    if return_data:
        return float(top_25_excluding_outliers(rewards)), evaluations
    return float(top_25_excluding_outliers(rewards))


def fitness_from_monitor(
    config: "MjCambrianConfig", monitor_csv: Path, *, return_data: bool = False
) -> float | Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    """Calculate the fitness of the agent. Uses the 3rd quartile of the cumulative
    monitor rewards."""
    timesteps, rewards = parse_monitor_csv(monitor_csv)

    if len(rewards) == 0:
        return -float("inf")

    if return_data:
        return float(np.percentile(rewards, 75)), (timesteps, rewards)
    return float(np.percentile(rewards, 75))


def fitness_from_txt(config: "MjCambrianConfig", txt_file: Path) -> float:
    """Calculate the fitness of the agent. Uses the 3rd quartile of the cumulative
    monitor rewards."""
    with open(txt_file, "r") as f:
        fitness = float(f.read().strip())

    return fitness


# ========================
# Fake eval fns which are used to test optimizers
# They should be realistic in the sense that they return a fitness value which is
# similar to the fitness value returned by the real eval fns. This means they should
# be fairly noisy and a similar correlation with the genotype as the real eval fns.


def fitness_num_eyes(
    config: "MjCambrianConfig",
    *,
    pattern: str,
    mean: float = 0,
    std: float = 5,
    assume_one: bool = True,
) -> float:
    """This fitness function will return higher rewards generally for agents with more
    eyes.

    Args:
        pattern (str): The path to the number of eyes in the config.
    """
    num_eyes = config.glob(pattern, flatten=True, assume_one=assume_one)

    # Set the seed such that when loading later, the same random values are generated
    # Get the seed based on the agent's id
    seed = config.seed
    if config.evo is not None:
        rank = config.evo.rank
        generation = config.evo.generation
        population_size = config.evo.population_size
        seed = generation * population_size + rank
    return np.random.default_rng(seed).normal(mean + np.prod(num_eyes), std)

def fitness_num_eyes_and_fov(
    config: "MjCambrianConfig",
    *,
    num_eyes_pattern: str,
    fov_pattern: str,
    mean: float = 0,
    std: float = 5,
    assume_one: bool = True,
    optimal_fov: float = 45
) -> float:
    """This fitness function will return higher rewards generally for agents with more
    eyes and a fov closer to the `optimal_fov`.
    """
    num_eyes = config.glob(num_eyes_pattern, flatten=True, assume_one=assume_one)
    fov = config.glob(fov_pattern, flatten=True, assume_one=assume_one)

    # Set the seed such that when loading later, the same random values are generated
    # Get the seed based on the agent's id
    seed = config.seed
    if config.evo is not None:
        rank = config.evo.rank
        generation = config.evo.generation
        population_size = config.evo.population_size
        seed = generation * population_size + rank

    rng = np.random.default_rng(seed)
    num_eyes = rng.normal(mean + np.prod(num_eyes), std)
    fov = rng.normal(mean + optimal_fov / max(abs(optimal_fov - fov), 1), std)
    return num_eyes + fov
