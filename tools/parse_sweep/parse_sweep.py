from typing import Dict, Any
from pathlib import Path

from cambrian.utils.config import MjCambrianConfig, MjCambrianBaseConfig
from cambrian.utils.config.utils import config_wrapper

# ==================


@config_wrapper
class ParseSweepConfig(MjCambrianBaseConfig):
    """
    folder (Path): The folder containing the sweep data.
    output (Path): The folder to save the parsed data.

    force (bool): Force loading of the data. If not passed, this script will try to
        find a parse_sweep.pkl file and load that instead.
    no_save (bool): Do not save the parsed data.
    quiet (bool): Quiet mode. Set's the logger to warning.
    debug (bool): Debug mode. Set's the logger to debug and disables tqdm.

    overrides (Dict[str, Any]): Overrides for the sweep data.

    dry_run (bool): Do not actually do any of the processing, just run the code without
        that part.
    """

    folder: Path
    output: Path

    force: bool
    no_save: bool
    quiet: bool
    debug: bool

    overrides: Dict[str, Any]

    dry_run: bool


# ==================


def save_data(config: Any, data: Any, pickle_file: Path):
    """Save the parsed data to a pickle file."""
    pickle_file = config.output / pickle_file
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_file, "wb") as f:
        pickle.dump(data, f)
    get_logger().info(f"Saved parsed data to {pickle_file}.")


def try_load_pickle(folder: Path, pickle_file: Path) -> Any | None:
    """Try to load the data from the pickle file."""
    pickle_file = folder / pickle_file
    if pickle_file.exists():
        get_logger().info(f"Loading parsed data from {pickle_file}...")
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        get_logger().info(f"Loaded parsed data from {pickle_file}.")
        return data

    get_logger().warning(f"Could not load {pickle_file}.")
    return None
