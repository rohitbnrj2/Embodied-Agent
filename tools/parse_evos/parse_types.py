from typing import List, Tuple, TypeAlias, Dict, Any, Optional, Self, Callable
from pathlib import Path
from dataclasses import field
from enum import Enum

import numpy as np
from omegaconf import SI

from cambrian.utils.config import MjCambrianConfig, MjCambrianBaseConfig
from cambrian.utils.config.utils import config_wrapper

Color: TypeAlias = List[Tuple[float, float, float, float]] | List[float]
Size: TypeAlias = List[float]

ParsedAxisData: TypeAlias = Tuple[np.ndarray | float | int, str]
ParsedColorData: TypeAlias = Tuple[Color, str]
ParsedSizeData: TypeAlias = Tuple[Size, str]
ParsedPlotData: TypeAlias = Tuple[
    ParsedAxisData,
    ParsedAxisData,
    ParsedAxisData | None,
    ParsedColorData,
    ParsedSizeData,
]
ExtractedData: TypeAlias = Tuple[np.ndarray | None]

# =======================================================
# Dataclasses


@config_wrapper
class Rank:
    """A rank is a single run of an inner training loop. Like in a generation, you have
    many ranks which in themselves have many parallel environments to speed up training.
    The terminology comes from MPI.
    """

    path: Path

    num: int
    generation: "Generation"

    config: Optional[MjCambrianConfig] = None
    evaluations: Optional[Dict[str, Any]] = None
    eval_fitness: Optional[float] = None
    monitor: Optional[Dict[str, Any]] = None
    train_fitness: Optional[float] = None

    parent: Optional[Self] = None
    children: List[Self] = field(default_factory=list)

    # Defauls to True
    ignored: bool = True


@config_wrapper
class Generation:
    """A generation is a collection of ranks. Throughout evolution (outer loop), you
    have many generations where each generation consists of many ranks which
    train (inner loop) in parallel.
    """

    path: Path

    num: int
    ranks: Dict[int, Rank] = field(default_factory=dict)

    data: "Data"

    # Defaults to True
    ignored: bool = True


@config_wrapper
class Data:
    """This is the primary data storage class. It contains all the generations and ranks
    and is used to store the parsed data. It also can accumulated arbitrary data
    which is used for plotting."""

    path: Path

    generations: Dict[int, Generation]

    accumulated_data: Dict[str, Any] = field(default_factory=dict)


class AxisDataType(Enum):
    GENERATION = "generation"
    CONFIG = "config"
    MONITOR = "monitor"
    WALLTIME = "walltime"
    EVALUATION = "evaluation"
    CONSTANT = "constant"
    CUSTOM = "custom"


@config_wrapper
class AxisData:
    type: AxisDataType

    label: Optional[str] = None
    lim: Optional[Tuple[float, float]] = None
    ticks: Optional[List[float]] = None
    tick_labels: Optional[List[str]] = None

    # CONFIG
    pattern: Optional[str] = None

    # MONITOR and WALLTIME
    window: Optional[int] = None

    # CUSTOM
    custom_fn: Optional[Callable[[Self, Data, Generation, Rank], ParsedAxisData]] = None

    # CONSTANT
    value: Optional[float] = None


class ColorType(Enum):
    SOLID = "solid"
    GENERATION = "generation"
    RANK = "rank"
    MONITOR = "monitor"
    EVALUATION = "evaluation"


@config_wrapper
class ColorData:
    type: ColorType = ColorType.SOLID

    label: Optional[str] = None
    clim: Optional[Tuple[float, float]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # SOLID
    color: Tuple[float, float, float, float] = (0.65490, 0.78039, 0.90588, 0.75)

    # GENERATION or RANK
    cmap: Optional[str] = None
    start_color: Optional[Tuple[float, float, float, float]] = None
    end_color: Optional[Tuple[float, float, float, float]] = None


class SizeType(Enum):
    NONE = "none"
    NUM = "num"
    GENERATION = "generation"
    MONITOR = "monitor"
    EVALUATION = "evaluation"
    CUSTOM = "custom"


@config_wrapper
class SizeData:
    type: SizeType = SizeType.NUM

    label: Optional[str] = None

    factor: float = 36  # default size for matplotlib scatter plots

    size_min: float = SI("${eval:'0.25*${.factor}'}")
    size_max: float = SI("${eval:'5*${.factor}'}")
    normalize: bool = True

    # CUSTOM
    custom_fn: Optional[Callable[[Self, Data, Generation, Rank], ParsedAxisData]] = None


class CustomPlotFnType(Enum):
    """Defines when the custom plot function is called.

    LOCAL: Called for each rank when plotting. Function signature should be
        `fn(plt.Axes, Rank, ...)`.
    GLOBAL: Called after the plot is created. Function signature should be
        `fn(plt.Axes, ...)`.
    """

    LOCAL = "local"
    GLOBAL = "global"


@config_wrapper
class CustomPlotFn:
    type: CustomPlotFnType

    fn: Optional[Callable] = None


@config_wrapper
class PlotData:
    x_data: AxisData
    y_data: AxisData
    z_data: Optional[AxisData] = None
    color_data: ColorData = ColorData()
    size_data: SizeData = SizeData()
    custom_fns: List[CustomPlotFn] = field(default_factory=list)

    projection: Optional[str] = None

    add_legend: bool = True

    title: Optional[str] = None
    add_title: bool = True

    name: str = SI("${parent:}")


@config_wrapper
class ParseEvosConfig(MjCambrianBaseConfig):
    """Config for the parse_evos script.

    Attributes:
        folder (Path): The folder to parse.
        output (Path): The output folder.
        plots_folder (Path): The folder to save the plots.
        evals_folder (Path): The folder to save the evaluations.

        force (bool): Force loading of the data. If not passed, this script will try to
            find a parse_evos.pkl file and load that instead.
        no_save (bool): Do not save the parsed data.
        no_check_finished (bool): Don't check if a file called `finished` has been
            written.
        quiet (bool): Quiet mode. Set's the logger to warning.
        debug (bool): Debug mode. Set's the logger to debug and disables tqdm.

        ranks (Optional[List[int]]): The rank to use. If not passed, all ranks are
            used.
        generations (Optional[List[int]]): The generation to use. If not passed, all
            are used.
        plots_mask (Optional[List[str]]): The plots to create. If not passed, all are
            created. This is the name of the plot to mask (i.e. use).
        plots_to_ignore (List[str]): Plots to ignore. This is the name of the plot to
            ignore.

        plot (bool): Plot the data.
        plot_nevergrad (bool): During evolution, if nevergrad is used, a `nevergrad.log`
            file might be saved which contains evolution information. We can load it
            and plot it if this is true.
        plot_phylogenetic_tree (bool): Plot the phylogenetic tree.
        render (bool): Run renderings for each processed rank. This will create a bunch
            of renders depending on the render dictionary.
        eval (bool): Evaluate the data.

        plots (Dict[str, PlotData]): The plots to create.
        renders (Dict[str, List[str]]): The render configurations to use. The
            images are saved to `{output}/renders/G{generation}_R{rank}_{key}`. The
            values correspond to overrides applied to the specific rank's config for
            rendering. NOTE: This may take a while, so it's recommended to use a subset
            of the full evo results.
        overrides (List[str]): Overrides for the config.

        dry_run (bool): Dry run.
    """

    folder: Path
    output: Path
    plots_folder: Path
    evals_folder: Path

    force: bool
    force_plot: bool
    no_save: bool
    no_check_finished: bool
    quiet: bool
    debug: bool

    ranks: Optional[List[int]] = None
    generations: Optional[List[int]] = None
    plots_mask: Optional[List[str]] = None
    plots_to_ignore: List[str]

    plot: bool
    plot_nevergrad: bool
    plot_phylogenetic_tree: bool
    render: bool
    eval: bool

    plots: Dict[str, PlotData] = field(default_factory=dict)
    renders: Dict[str, List[str]] = field(default_factory=dict)
    overrides: List[str] = field(default_factory=list)

    dry_run: bool
