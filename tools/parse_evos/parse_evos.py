from typing import Dict, Optional, Any, Tuple
from functools import partial
import sys

import tqdm.rich as tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from cambrian.utils import (
    is_integer,
    save_data,
    try_load_pickle,
    set_matplotlib_style,
    get_logger,
)
from cambrian.utils.config.utils import clean_key
from cambrian.utils.config import MjCambrianConfig, run_hydra
from parse_types import (
    Rank,
    Generation,
    ParseEvosConfig,
    Data,
    CustomPlotFnType,
    AxisDataType,
    PlotData,
)
from parse_helpers import parse_plot_data, load_data
from plot_helpers import (
    plot_helper,
    adjust_points,
    adjust_axes,
    run_custom_fns,
    add_legend,
    add_colorbar,
)
from utils import extract_data


# =======================================================

# Parsing the yaml files sometimes causes a RecursionError
# Default is 1000, but we'll set it to 10000 to be safe
sys.setrecursionlimit(10000)

set_matplotlib_style()

# =======================================================


def run_plot(config: ParseEvosConfig, data: Data):
    get_logger().info("Plotting data...")

    # Update the rcparams.figure.max_open_warning to suppress the warning
    get_logger().debug(f"Setting max open warning to {len(config.plots)}.")
    plt.rcParams["figure.max_open_warning"] = len(config.plots)

    for generation, generation_data in data.generations.items():
        # Only plot the generation we want, if specified
        if generation_data.ignored:
            continue

        get_logger().info(f"Plotting generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            # Only plot the rank we want, if specified
            if rank_data.ignored:
                continue

            get_logger().info(f"\tPlotting rank {rank}...")

            for plot in config.plots.values():
                if config.plots_mask is not None and plot.name not in config.plots_mask:
                    get_logger().debug(f"Skipping plot {plot.name}.")
                    continue
                elif plot.name in config.plots_to_ignore:
                    get_logger().debug(f"Skipping plot {plot.name}.")
                    continue

                rank_id = f"G{generation}_R{rank}"
                try:
                    x_data, y_data, z_data, color_data, size_data = parse_plot_data(
                        plot, data, generation_data, rank_data
                    )
                except AssertionError as e:
                    if rank_data.ignored:
                        title = f" {plot.title}" if plot.title else ""
                        get_logger().debug(f"{rank_id} Ignoring plot{title}: {e}")
                        continue
                    elif config.debug:
                        raise ValueError(
                            f"{rank_id} Error parsing plot {plot.name}: {e}"
                        )
                    else:
                        get_logger().warning(
                            f"{rank_id} Couldn't parse plot {plot.name}: {e}"
                        )
                        if config.quiet:
                            get_logger().warning(
                                f"{rank_id} Ignoring this plot in the future."
                            )
                            with config.set_readonly_temporarily(False):
                                config.plots_to_ignore = [
                                    *config.plots_to_ignore,
                                    plot.name,
                                ]
                        continue
                except Exception as e:
                    if config.debug:
                        get_logger().error(f"{rank_id} Error parsing plot {plot.name}")
                        raise e
                    else:
                        get_logger().error(
                            f"{rank_id} Error parsing plot {plot.name}: {e}"
                        )
                    continue

                x_data, xlabel = x_data
                y_data, ylabel = y_data
                z_data, zlabel = z_data or (None, None)
                color, _ = color_data
                size, size_label = size_data

                default_title = f"{ylabel} vs {xlabel}"
                if zlabel:
                    default_title = f"{zlabel} vs {default_title}"
                title = plot.title or default_title

                projection = plot.projection or ("3d" if z_data else "rectilinear")

                # Plot the data
                get_logger().debug(f"\t\tPlotting {plot.name}...")

                # Run custom functions
                if plot.custom_fns:
                    fig = plt.figure(plot.name)
                    ax = fig.gca()
                    for custom_fn in plot.custom_fns:
                        if custom_fn.type is CustomPlotFnType.LOCAL:
                            custom_fn.fn(ax, plot, rank_data)

                plot_helper(
                    x_data,
                    y_data,
                    z_data,
                    name=plot.name,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    zlabel=zlabel,
                    projection=projection,
                    marker=".",
                    c=color,
                    s=size,
                    label=size_label,
                    dry_run=config.dry_run,
                )

    # If there is no data, set all the plots to be ignored
    if len(data.generations) == 0:
        if config.debug:
            raise ValueError("No data found. Ignoring all plots.")
        else:
            get_logger().warning("No data found. Ignoring all plots.")
        with config.set_readonly_temporarily(False):
            config.plots_to_ignore = [
                *config.plots_to_ignore,
                *[p.name for p in config.plots.values()],
            ]


def update_plots(
    config: ParseEvosConfig,
    *,
    save: bool = True,
    show: bool = False,
):
    # Filter the plots
    plots: Dict[str, PlotData] = {}
    for plot in config.plots.values():
        if config.plots_mask is not None and plot.name not in config.plots_mask:
            continue
        elif plot.name in config.plots_to_ignore:
            continue
        plots[plot.name] = plot

    if len(plots) == 0:
        get_logger().warning("No plots to save.")
        return

    # Now save the plots
    progress_bar = tqdm.tqdm(total=len(plots), desc="Saving...", disable=config.debug)
    for plot_data in plots.values():
        progress_bar.update(1)

        fig = plt.figure(plot_data.name)
        ax = fig.gca()

        try:
            # Extract the data from the plot
            x_data, y_data, z_data, colors, sizes = extract_data(
                ax, return_data=True, return_color=True, return_size=True
            )
            is_3d = z_data is not None
            if is_3d:
                points = np.column_stack((x_data, y_data, z_data))
            else:
                points = np.column_stack((x_data, y_data))

            # We'll ignore any plots which don't have unique data along any axis
            # These plots aren't really useful as there is no independent variable.
            if plot_data.x_data.type is not AxisDataType.CONSTANT and np.all(
                x_data == x_data[0]
            ):
                get_logger().debug(f"Skipping plot {plot_data.name}: no unique x_data.")
                continue
            elif plot_data.y_data.type is not AxisDataType.CONSTANT and np.all(
                y_data == y_data[0]
            ):
                get_logger().debug(f"Skipping plot {plot_data.name}: no unique y_data.")
                continue
            elif (
                is_3d
                and plot_data.z_data.type is not AxisDataType.CONSTANT
                and np.all(z_data == z_data[0])
            ):
                get_logger().debug(f"Skipping plot {plot_data.name}: no unique z_data.")
                continue

            # Adjust the points, if necessary
            adjust_points(plot_data.size_data, ax, sizes, points, colors)

            # Adjust the axes
            adjust_axes(ax, plot_data)

            # Run any custom functions
            run_custom_fns(ax, plot_data)

            # Add a legend entry for the blue circles
            if plot_data.add_legend:
                add_legend(ax, plot_data, points, sizes, colors, loc="upper right")

            # Normalize the colorbar
            add_colorbar(plot_data.color_data, ax, colors)

            # If all the values of an axis are integers, set the axis to integer
            if is_integer(x_data):
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if is_integer(y_data):
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if is_3d and is_integer(z_data):
                ax.zaxis.set_major_locator(MaxNLocator(integer=True))
        except (ValueError, TypeError) as e:
            # Get the line number of the error
            if config.debug:
                raise e
            _fn_name = e.__traceback__.tb_frame.f_code.co_name
            _line_number = e.__traceback__.tb_lineno
            get_logger().error(
                f"{_fn_name}:{_line_number}: "
                f"Error extracting data from plot {plot_data.name}: {e}"
            )
            continue

        # Remove the title if desired
        if not plot_data.add_title:
            fig.suptitle("")

        # Set the aspect ratio
        if is_3d:
            ax.set_box_aspect([1, 1, 1])
        elif ax.name == "rectilinear":
            ax.set_box_aspect(1)
        elif ax.name == "polar":
            ax.set_theta_zero_location("N")
        fig.tight_layout()

        # Save the plot
        if save:
            filename = f"{plot_data.name}.png"
            plt.savefig(
                config.plots_folder / filename,
                dpi=500,
                bbox_inches="tight",
                transparent=False,
            )

            get_logger().debug(f"Saved plot to {config.plots_folder / filename}.")
        if show:
            plt.show()

        plt.close(fig)


# =======================================================


def plot_nevergrad(config: ParseEvosConfig, data: Data):
    try:
        import hiplot
        import nevergrad as ng
    except ImportError:
        get_logger().error(
            "Could not import hiplot or nevergrad. Please run "
            "`pip install hiplot nevergrad`."
        )
        return

    get_logger().info("Plotting nevergrad...")

    # Load the nevergrad log file. We'll load it and convert it to a
    # hiplot experiment. HiPlot is a package from Meta which helps
    # visualize hyperparameter optimization experiments.
    if not (nevergrad_log := config.folder / "nevergrad.log").exists():
        get_logger().error(f"Nevergrad log file {nevergrad_log} not found.")
        return

    # Generate the nevergrad html file for offline viewing.
    parameters_logger = ng.callbacks.ParametersLogger(nevergrad_log)
    exp: hiplot.Experiment = parameters_logger.to_hiplot_experiment()
    try:
        exp.to_html(config.plots_folder / "nevergrad.html")
    except hiplot.ExperimentValidationMissingParent as e:
        get_logger().warning(f"Error generating nevergrad html: {e}")


# =======================================================


def plot_phylogenetic_tree(config: ParseEvosConfig, data: Data):
    try:
        import ete3 as ete
    except ImportError:
        get_logger().error(
            "Could not import ete3. Please run `pip install ete3 pyqt5`."
        )
        return
    import os

    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    get_logger().info("Plotting phylogenetic tree...")

    def add_node(
        nodes: Dict[str, ete.Tree],
        *,
        rank_data: Rank,
        generation_data: Generation,
        pattern: str,
    ):
        rank = rank_data.num
        generation = generation_data.num

        if generation_data.ignored:
            return

        # Define a unique identifier for each rank
        rank_id = f"G{generation}_R{rank}"
        if rank_id in nodes:
            return

        # Get the parent identifier
        if not (config := rank_data.config):
            get_logger().debug(f"Skipping rank {rank_id} because it has no config.")
            return
        if not (evaluations := rank_data.evaluations):
            get_logger().debug(
                f"Skipping rank {rank_id} because it has no evaluations."
            )
            return

        # If this is the first generation, the parent is set to the root
        if generation == 0 or rank_data.parent.num == -1:
            parent_id = "root"
        else:
            parent_id = f"G{rank_data.parent.generation.num}_R{rank_data.parent.num}"

            if parent_id not in nodes:
                parent_generation_data = data.generations[
                    rank_data.parent.generation.num
                ]
                parent_rank_data = parent_generation_data.ranks[rank_data.parent.num]
                add_node(
                    nodes,
                    rank_data=parent_rank_data,
                    generation_data=parent_generation_data,
                    pattern=pattern,
                )

        # Create the rank node under the parent node
        parent = nodes[parent_id]
        node = parent.add_child(name=rank_id)
        nodes[rank_id] = node

        # Add features for the ranks and generations
        node.add_feature("rank", rank)
        node.add_feature("generation", generation)
        if generation != 0:
            node.add_feature("parent_rank", rank_data.parent.num)
            node.add_feature("parent_generation", rank_data.parent.generation.num)
        else:
            node.add_feature("parent_rank", -1)
            node.add_feature("parent_generation", -1)

        # Add evaluations to the node
        key = "mean_results" if "mean_results" in evaluations else "results"
        node.add_feature("fitness", np.max(evaluations[key]))

        # Add a text label and feature for the pattern
        globbed_data = config.glob(pattern, flatten=True)
        for key, value in globbed_data.items():
            if isinstance(value, list):
                value = np.average(value).astype(type(value[0]))

            node.add_feature(key, value)

    def build_tree(pattern: Optional[str] = None) -> ete.Tree:
        tree = ete.Tree()
        tree.name = "root"

        # Dictionary to keep track of created nodes
        nodes = {"root": tree}

        # Iterate through the generations and ranks
        for generation_data in data.generations.values():
            if generation_data.num < 2:
                # Generations < 2 don't have any parents in nevergrad
                continue
            elif generation_data.ignored:
                continue

            for rank_data in generation_data.ranks.values():
                if rank_data.ignored:
                    continue

                add_node(
                    nodes,
                    rank_data=rank_data,
                    generation_data=generation_data,
                    pattern=pattern,
                )

        return tree

    def style_node(
        node: ete.Tree,
        *,
        style_feature_key: str,
        value_threshold: Optional[float] = None,
        value_range: Optional[Tuple[float, float]] = None,
        color: Optional[str] = None,
        color_range: Optional[Tuple[str, str]] = None,
    ):
        # We'll color the node and the recursive ancestors to highlight good values
        # The horizontal lines which directly lead to the best value are bolded and
        # highlighted red. Horizontal lines which are children of the optimal path but
        # not the best value are highlighted red. The vertical lines are just
        # highlighted red.
        feature_value = getattr(node, style_feature_key)
        assert value_threshold is None or value_range is None

        # If the feature value is less than the threshold, don't style the node
        if value_threshold and feature_value < value_threshold:
            return

        assert color is None or color_range is None
        if color_range:
            assert value_range is not None
            assert value_range[0] <= feature_value <= value_range[1]

            # Interpolate the color between the color range
            # The color range is defined as hex strings
            def hex_to_rgb(hex_str: str):
                return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))

            def rgb_to_hex(rgb: Tuple[int, int, int]):
                return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

            color_range = [hex_to_rgb(color) for color in color_range]
            t = np.interp(feature_value, value_range, [0, 1])
            color = tuple(int(c0 + (c1 - c0) * t) for c0, c1 in zip(*color_range))
            color = rgb_to_hex(color)

        # Create the node style
        path_style = ete.NodeStyle()
        path_style["fgcolor"] = color
        path_style["hz_line_color"] = color
        path_style["hz_line_width"] = 4

        # Set the node style
        current_node = node
        while current_node.up:
            current_node.set_style(path_style)
            current_node = current_node.up

    def layout(
        node: ete.Tree,
        feature_key: str,
        *,
        style_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if node.name == "root":
            return

        # Add a text annotation to each node
        feature_value = getattr(node, feature_key)
        text_face = ete.TextFace(feature_value, bold=True)
        node.add_face(text_face, column=0, position="branch-right")

        # Set the node distance. Each node distance is set such that each generation
        # is at the same distance from the root
        node.dist = getattr(node, "generation") - getattr(node, "parent_generation")
        node.dist -= text_face.get_bounding_rect().width() / 25
        node.dist = max(node.dist, 0)

        # Update the node style, if desired
        if style_kwargs is not None:
            style_node(node, **style_kwargs)

    # Get the parameterization properties from the first parameter
    raise NotImplementedError("Bug below, this is not working.")
    loaded_parameters = None
    patterns: Dict[str, str] = {"generation": "evo.generation.num"}
    for maybe_param in next(iter(loaded_parameters)).keys():
        if "#" not in maybe_param:
            cleaned_key = clean_key(maybe_param)
            patterns[cleaned_key] = maybe_param

    # Loop through the patterns and create a tree for each
    for feature_key, pattern in patterns.items():
        # Build the tree
        # We'll add features/data to be used in the visualization later
        get_logger().info(f"Creating phylogenetic tree for feature {feature_key}...")
        tree = build_tree(pattern).copy("deepcopy")

        # Sort the descendants of the tree by the feature key
        tree.sort_descendants(feature_key)
        tree.ladderize()

        # Get the max value for the feature key
        style_feature_key = "fitness"
        values = [getattr(node, style_feature_key) for node in tree.iter_leaves()]
        threshold = np.max(values)
        style_kwargs = dict(
            style_feature_key=style_feature_key,
            value_threshold=threshold,
            color="#ff0000",
        )
        layout_fn = partial(
            layout,
            feature_key=feature_key,
            style_kwargs=style_kwargs,
        )

        # Create the visualization
        tree_style = ete.treeview.TreeStyle()
        tree_style.show_leaf_name = False
        tree_style.layout_fn = layout_fn
        # tree_style.mode = "c"
        tree_style.arc_start = -180
        tree_style.arc_span = 180
        tree_style.title.add_face(
            ete.TextFace(f"Phylogenetic Tree for {feature_key}", fsize=20), column=0
        )
        tree.render(
            str(config.folder / f"phylogenetic_tree_{feature_key}.png"),
            dpi=500,
            tree_style=tree_style,
        )


# =======================================================


def run_render(config: ParseEvosConfig, data: Data):
    import mujoco as mj
    from cambrian.renderer import MjCambrianRendererSaveMode

    get_logger().info("Rendering data...")

    for generation, generation_data in data.generations.items():
        if generation_data.ignored:
            continue

        get_logger().debug(f"\tRendering generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            if rank_data.ignored:
                continue

            get_logger().debug(f"\tRendering rank {rank}...")
            config.renders_folder.mkdir(parents=True, exist_ok=True)

            for fname, exp_overrides in config.renders.items():
                exp_config = MjCambrianConfig.compose(
                    rank_data.path.absolute(),
                    config.config_filename,
                    overrides=[
                        *config.overrides,
                        *exp_overrides,
                        "hydra.searchpath=['configs']",
                    ],
                )

                # Run the experiment
                # Involves first creating the environment and then rendering it
                env = exp_config.env.instance(exp_config.env)

                env.record = True
                env.reset(seed=exp_config.seed)
                for _ in range(30):
                    mj.mj_step(env.model, env.data)
                    env.render()
                env.save(
                    config.renders_folder / f"G{generation}_R{rank}_{fname}",
                    save_pkl=False,
                    save_mode=MjCambrianRendererSaveMode.PNG,
                )


# =======================================================


def run_eval(config: ParseEvosConfig, data: Data):
    from cambrian.ml.trainer import MjCambrianTrainer

    # # Have to update the sys.modules to include the features_extractors module
    # # features_extractors is saved in the model checkpoint
    # import sys
    # from cambrian.ml import features_extractors

    # sys.modules["features_extractors"] = features_extractors

    get_logger().info("Evaluating model...")

    for generation, generation_data in data.generations.items():
        if config.generations is not None and generation not in config.generations:
            continue

        get_logger().info(f"Evaluating generation {generation}...")

        for rank, rank_data in generation_data.ranks.items():
            if config.ranks is not None and rank not in config.ranks:
                continue
            elif rank_data.config is None:
                continue

            get_logger().info(f"\tEvaluating rank {rank}...")

            if not config.dry_run:
                trainer = MjCambrianTrainer(rank_data.config)
                trainer.eval()

            get_logger().info("\t\tDone evaluating.")


# =======================================================


def main(config: ParseEvosConfig):
    assert not (config.debug and config.quiet), "Cannot be both debug and quiet."
    if config.debug:
        get_logger().setLevel("DEBUG")
    elif config.quiet:
        get_logger().setLevel("WARNING")

    assert config.folder.exists(), f"Folder {config.folder} does not exist."
    config.plots_folder.mkdir(parents=True, exist_ok=True)

    if config.force or (data := try_load_pickle(config.output, "data.pkl")) is None:
        data = load_data(config)

        if not config.no_save:
            save_data(data, config.output, "data.pkl")

    # Update the generations and ranks list, if desired
    if config.filter_fn is not None:
        with config.set_readonly_temporarily(False):
            data.generations = config.filter_fn(data)

    if config.plot:
        run_plot(config, data)
        update_plots(config)
    if config.plot_nevergrad:
        plot_nevergrad(config, data)
    if config.plot_phylogenetic_tree:
        plot_phylogenetic_tree(config, data)
    if config.render:
        run_render(config, data)
    if config.eval:
        config.evals_folder.mkdir(parents=True, exist_ok=True)
        run_eval(config, data)


if __name__ == "__main__":
    run_hydra(
        main,
        config_name="tools/parse_evos/parse_evos",
    )
