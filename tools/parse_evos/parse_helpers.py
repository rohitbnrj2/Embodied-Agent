import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from parse_types import (
    AxisData,
    AxisDataType,
    Color,
    ColorData,
    ColorType,
    Data,
    Generation,
    ParsedAxisData,
    ParsedColorData,
    ParsedPlotData,
    ParseEvosConfig,
    PlotData,
    Rank,
    SizeData,
    SizeType,
)
from stable_baselines3.common.results_plotter import ts2xy

from cambrian.ml.fitness_fns import (
    fitness_from_evaluations,
    fitness_from_monitor,
    fitness_from_txt,
)
from cambrian.utils import moving_average
from cambrian.utils.config import MjCambrianConfig
from cambrian.utils.logger import get_logger


def get_generation_file_paths(folder: Path) -> Data:
    """Create the initial storage dict for parsing and get the paths to all the
    generation/rank folders.

    The generation data is stored as follows:
    - <exp_name>
        - generation_0
            - rank_0
            - rank_1
            - ...
        - generation_1
            - rank_0
            - rank_1
            - ...
        - ...

    This function will parse the folder heirarchy as a preprocessing step and returns
    the file paths to all the generation/rank folders.
    """

    data = Data(path=folder, generations=dict())
    for root, dirs, files in os.walk(folder):
        root = Path(root)
        # Only parse the generation folders
        if not root.stem.startswith("generation_"):
            continue

        generation = Generation(
            data=data,
            path=root,
            num=int(root.stem.split("generation_", 1)[1]),
            ranks=dict(),
        )
        data.generations[generation.num] = generation
        for dir in dirs:
            dir = Path(dir)
            # Only parse the rank folders
            if not dir.stem.startswith("rank_"):
                continue

            # Grab the rank number
            rank = int(dir.stem.split("rank_", 1)[1])
            generation.ranks[rank] = Rank(
                path=root / dir, num=rank, generation=generation
            )

        # Sort the ranks by rank number
        generation.ranks = dict(sorted(generation.ranks.items()))

    # Sort the generations by generation number
    data.generations = dict(sorted(data.generations.items()))

    return data


def load_data(config: ParseEvosConfig) -> Data:
    """Load the data from the generation/rank folders. This function will walk through
    the folder heirarchy and load the config, evaluations and monitor data for each
    rank."""
    get_logger().info(f"Loading data from {config.folder}...")

    # Get the file paths to all the generation/rank folders
    data = get_generation_file_paths(config.folder)

    # Convert the overrides to a list
    overrides = config.overrides.to_container()

    # If we're loading nevergrad data, we do that here to store the parent rank and
    # generation
    if (nevergrad_log := config.folder / "nevergrad.log").exists() and False:
        import nevergrad as ng

        parameters_logger = ng.callbacks.ParametersLogger(nevergrad_log)
        loaded_parameters = parameters_logger.load()

        # By default, the nevergrad loader loads the data in a way that best works for
        # hiplot, restructure the data so it's easier for us to use.
        uid_to_rank_and_generation: Dict[str, Tuple[int, int]] = {}
        rank_and_generation_to_uid: Dict[Tuple[int, int], str] = {}
        uid_to_parent_uid: Dict[str, str] = {}
        for param in loaded_parameters:
            uid = param["#uid"]
            if "CMA" in param["#optimizer"]:
                generation = param["#generation"] - 1  # 1-indexed, convert to 0 indexed
            else:
                generation = param["#generation"]
            rank = param["#num-tell"] % int(param["#num-ask"] // (generation + 1))

            uid_to_rank_and_generation[uid] = (rank, generation)
            rank_and_generation_to_uid[(rank, generation)] = uid
            if len(param["#parents_uids"]) == 1:
                parent_uid = param["#parents_uids"][0]
                uid_to_parent_uid[uid] = parent_uid

        def get_parent_rank_and_generation(
            rank: int, generation: int
        ) -> Tuple[int, int]:
            if (uid := rank_and_generation_to_uid.get((rank, generation))) is None:
                return (-1, -1)
            if (parent_uid := uid_to_parent_uid.get(uid)) is None:
                return (-1, -1)
            return uid_to_rank_and_generation.get(parent_uid, (-1, -1))

    # Walk through each generation/rank and parse the config, evaluations and monitor
    # data
    for generation, generation_data in data.generations.items():
        # Only load the generation we want, if specified
        if config.generations is not None and generation not in config.generations:
            generation_data.ignored = True
            continue

        get_logger().info(f"Loading generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            # Only load the rank we want, if specified
            if config.ranks is not None and rank not in config.ranks:
                continue

            get_logger().info(f"\tLoading rank {rank}...")

            # Check if the `finished` file exists.
            # If not, don't load the data. Sorry for the double negative.
            if not (rank_data.path / "finished").exists() and config.check_finished:
                get_logger().warning(
                    f"\t\tSkipping rank {rank} because it is not finished."
                )
                continue
            if (rank_data.path / "ignore").exists():
                get_logger().warning(f"\t\tSkipping rank {rank} because it is ignored.")
                rank_data.ignored = True
                continue

            # Get the config file
            if (config_file := rank_data.path / config.config_filename).exists():
                get_logger().debug(f"\tLoading config from {config_file}...")
                rank_data.config = MjCambrianConfig.load(config_file, instantiate=False)
                rank_data.config.merge_with_dotlist(overrides)
                rank_data.config.resolve()

            # Get the evaluations file
            evaluations_file = rank_data.path / "evaluations.npz"
            if evaluations_file.exists():
                (
                    rank_data.eval_fitness,
                    rank_data.evaluations,
                ) = fitness_from_evaluations(
                    rank_data.config, evaluations_file, return_data=True
                )
                if np.isnan(rank_data.eval_fitness):
                    get_logger().warning(
                        f"\t\tRank {rank} has NaN evaluation fitness. Ignoring..."
                    )
                    rank_data.ignored = True
            else:
                txt_file = rank_data.path / "test_fitness.txt"
                if txt_file.exists():
                    rank_data.eval_fitness = fitness_from_txt(
                        rank_data.config, txt_file
                    )

            # Get the monitor file
            monitor_file = rank_data.path / "monitor.csv"
            if monitor_file.exists():
                (
                    rank_data.train_fitness,
                    rank_data.monitor,
                ) = fitness_from_monitor(None, monitor_file, return_data=True)

            # If we're loading nevergrad data, we do that here to store the parent rank
            # and generation
            if "get_parent_rank_and_generation" in locals():
                parent_rank, parent_generation = get_parent_rank_and_generation(
                    rank, generation
                )
                if parent_rank != -1 and parent_generation != -1:
                    rank_data.parent = data.generations[parent_generation].ranks[
                        parent_rank
                    ]
                    rank_data.parent.children.append(rank_data)

            # Set ignored to False
            rank_data.ignored = False

        # Set ignored to False
        generation_data.ignored = False

    return data


def get_axis_label(axis_data: AxisData, label: str = "") -> str:
    if axis_data.type is AxisDataType.GENERATION:
        label = "Generation"
    elif axis_data.type is AxisDataType.CONFIG:
        label = axis_data.label if axis_data.label is not None else axis_data.pattern
    elif axis_data.type is AxisDataType.MONITOR:
        label = "Training Reward"
    elif axis_data.type is AxisDataType.WALLTIME:
        label = "Walltime (minutes)"
    elif axis_data.type is AxisDataType.EVALUATION:
        label = "Fitness"
    elif axis_data.type is AxisDataType.CONSTANT:
        label = ""
    return axis_data.label if axis_data.label is not None else label


def parse_axis_data(
    axis_data: AxisData, data: Data, generation_data: Generation, rank_data: Rank
) -> ParsedAxisData:
    data: Any
    if axis_data.type is AxisDataType.GENERATION:
        data = generation_data.num + 1
    elif axis_data.type is AxisDataType.CONFIG:
        assert rank_data.config is not None, "Config is required for CONFIG."
        data = rank_data.config.glob(axis_data.pattern, flatten=True)
        pattern = axis_data.pattern.split(".")[-1]
        assert pattern in data, f"Pattern {axis_data.pattern} not found."
        data = data[pattern]
        if isinstance(data, list):
            data = np.average(data)
    elif axis_data.type is AxisDataType.MONITOR:
        assert rank_data.train_fitness is not None, "Monitor is required."
        data = rank_data.train_fitness
    elif axis_data.type is AxisDataType.WALLTIME:
        assert rank_data.monitor is not None, "Monitor is required."
        t = ts2xy(rank_data.monitor, "walltime_hrs")[0] * 60  # convert to minutes
        window = axis_data.window or 100
        if len(t) < window:
            t = moving_average(t, axis_data.window)
        data = t[-1]
    elif axis_data.type is AxisDataType.EVALUATION:
        assert rank_data.eval_fitness is not None, "Evaluations is required."
        data = rank_data.eval_fitness
    elif axis_data.type is AxisDataType.CUSTOM:
        assert axis_data.custom_fn is not None, "Custom function is required."
        return axis_data.custom_fn(axis_data, data, generation_data, rank_data)
    elif axis_data.type is AxisDataType.CONSTANT:
        assert axis_data.value is not None, "Value is required for CONSTANT."
        data = axis_data.value
    else:
        raise ValueError(f"Unknown data type {axis_data.type}.")

    assert data is not None
    return data, get_axis_label(axis_data)


def get_color_label(color_data: ColorData | None) -> str:
    label: str
    if color_data is None:
        return ""
    elif color_data.type is ColorType.SOLID:
        label = "Color"
    elif color_data.type is ColorType.CONFIG:
        label = color_data.pattern
    elif color_data.type is ColorType.GENERATION:
        label = "Generation"
    elif color_data.type is ColorType.RANK:
        label = "Rank"
    elif color_data.type is ColorType.MONITOR:
        label = "Training Reward"
    elif color_data.type is ColorType.EVALUATION:
        label = "Fitness"
    else:
        raise ValueError(f"Unknown color type {color_data.type}.")
    return color_data.label if color_data.label is not None else label


def parse_color_data(
    color_data: ColorData, data: Data, generation_data: Generation, rank_data: Rank
) -> ParsedColorData:
    color: Color
    if color_data.type is ColorType.SOLID:
        assert color_data.color is not None, "Color is required for solid color."
        color = color_data.color
    elif color_data.type is ColorType.CONFIG:
        assert rank_data.config is not None, "Config is required for CONFIG."
        assert color_data.pattern is not None, "Pattern is required for CONFIG."
        color = rank_data.config.glob(color_data.pattern, flatten=True)
        pattern = color_data.pattern.split(".")[-1]
        assert pattern in color, f"Pattern {color_data.pattern} not found."
        color = color[pattern]
        if isinstance(color, list):
            color = np.average(color)
    elif color_data.type is ColorType.GENERATION:
        assert color_data.cmap is not None or (
            color_data.start_color is not None and color_data.end_color is not None
        ), "Cmap or start_color and end_color is required."

        # NOTE: The cmap is generated later and is normalized to the number of
        # generations.
        color = generation_data.num + 1
    elif color_data.type is ColorType.RANK:
        assert color_data.cmap is not None or (
            color_data.start_color is not None and color_data.end_color is not None
        ), "Cmap or start_color and end_color is required."

        # NOTE: The cmap is generated later and is normalized to the number of ranks.
        color = rank_data.num
    elif color_data.type is ColorType.MONITOR:
        color = rank_data.train_fitness
    elif color_data.type is ColorType.EVALUATION:
        color = rank_data.eval_fitness
        assert color is not None, "Evaluations is required."
    else:
        raise ValueError(f"Unknown color type {color_data.type}.")

    assert not np.any(np.isnan(color)), "Color cannot be NaN."
    return [color], get_color_label(color_data)


def get_size_label(size_data: SizeData | None) -> str:
    label: str
    if size_data is None:
        return ""
    elif size_data.type is SizeType.NONE:
        label = None
    elif size_data.type is SizeType.NUM:
        label = "Number"
    elif size_data.type is SizeType.GENERATION:
        label = "Generation"
    elif size_data.type is SizeType.MONITOR:
        label = "Training Reward"
    elif size_data.type is SizeType.EVALUATION:
        label = "Fitness"
    elif size_data.type is SizeType.CUSTOM:
        label = "Custom"
    elif size_data.type is SizeType.CONFIG:
        label = size_data.pattern
    return size_data.label if size_data.label is not None else label


def parse_size_data(
    size_data: SizeData, data: Data, generation_data: Generation, rank_data: Rank
) -> ParsedAxisData:
    size: float
    if size_data.type is SizeType.NONE:
        size = 1
    elif size_data.type is SizeType.NUM:
        # Will be updated later to reflect the number of overlapping points at the same
        # location
        size = 1
    elif size_data.type is SizeType.GENERATION:
        size = generation_data.num + 1
    elif size_data.type is SizeType.MONITOR:
        assert rank_data.train_fitness is not None, "Monitor is required."
        size = rank_data.train_fitness
    elif size_data.type is SizeType.EVALUATION:
        assert rank_data.eval_fitness is not None, "Evaluations is required."
        size = rank_data.eval_fitness
    elif size_data.type is SizeType.CUSTOM:
        assert size_data.custom_fn is not None, "Custom function is required."
        return size_data.custom_fn(size_data, rank_data)
    elif size_data.type is SizeType.CONFIG:
        assert rank_data.config is not None, "Config is required for CONFIG."
        size = rank_data.config.glob(size_data.pattern, flatten=True)
        pattern = size_data.pattern.split(".")[-1]
        assert pattern in size, f"Pattern {size_data.pattern} not found."
        size = size[pattern]
        if isinstance(size, list):
            size = np.average(size)
    else:
        raise ValueError(f"Unknown size type {size_data.type}.")

    return size, get_size_label(size_data)


def parse_plot_data(
    plot_data: PlotData, data: Data, generation_data: Generation, rank_data: Rank
) -> ParsedPlotData:
    """Parses the plot data object and grabs the relevant x and y data."""

    x_data = parse_axis_data(plot_data.x_data, data, generation_data, rank_data)
    y_data = parse_axis_data(plot_data.y_data, data, generation_data, rank_data)
    z_data = (
        parse_axis_data(plot_data.z_data, data, generation_data, rank_data)
        if plot_data.z_data
        else None
    )
    color = parse_color_data(plot_data.color_data, data, generation_data, rank_data)
    size = parse_size_data(plot_data.size_data, data, generation_data, rank_data)
    return x_data, y_data, z_data, color, size
