from typing import List
from pathlib import Path
import warnings

from cambrian.utils.config import MjCambrianConfig
from cambrian.ml.trainer import MjCambrianTrainer

warnings.filterwarnings("ignore")

def run_eval(folder: Path, overrides: List[str]):
    assert (folder / "config.pkl").exists()

    config = MjCambrianConfig.load_pickle(folder / "config.pkl", overrides=overrides)
    trainer = MjCambrianTrainer(config)
    fitness = trainer.eval()

    print(folder)
    print(f"Fitness: {fitness}")
    print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("folder", type=Path, help="The folder containing the config.yaml file")
    parser.add_argument(
        "-o",
        "--override",
        "--overrides",
        dest="overrides",
        action="extend",
        nargs="+",
        type=str,
        help="Override config values. Do <config>.<key>=<value>",
        default=[],
    )

    args = parser.parse_args()

    run_eval(args.folder.absolute(), args.overrides)