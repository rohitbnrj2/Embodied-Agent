from typing import List, Dict
from pathlib import Path

import mujoco as mj

from cambrian.renderer import MjCambrianRendererSaveMode
from cambrian.utils.logger import get_logger
from cambrian.utils.config import (
    MjCambrianBaseConfig,
    MjCambrianConfig,
    run_hydra,
    config_wrapper,
)


@config_wrapper
class AnimalShowcaseConfig(MjCambrianBaseConfig):
    """The configuration for the animal showcase.

    Attributes:
        logdir (Path): The primary directory which simulation data is stored in. This is
            the highest level directory used for saving the showcase outdir.
        outdir (Path): The directory used for saving the showcase. This is the directory
            where the showcase's data is stored. Should evaluate to
            `logdir` / `outsubdir`.
        outsubdir (Path): The subdirectory relative to `logdir` where the showcase's data
            is stored. This is the directory where the showcase's data is
            actually stored.

        exp (str): The experiment to run. This is the path to the hydra exp file
            as if you are you running the experiment from the root of the project
            (i.e. relative to the exp/ directory).

        overrides (Dict[str, List[str]]): The overrides to apply to the loaded
            configuration. This is a number of overrides that are used to generate
            images. The image is saved using the key as the filename and the value as
            the overrides to apply to the configuration.
    """

    logdir: Path
    outdir: Path
    outsubdir: Path

    exp: str

    overrides: Dict[str, List[str]]


def main(config: AnimalShowcaseConfig, *, overrides: List[str]):
    overrides = [*overrides, f"exp={config.exp}", "hydra/sweeper=basic"]
    for fname, exp_overrides in config.overrides.items():
        get_logger().info(f"Composing animal showcase {fname}...")
        print(overrides)
        exp_config = MjCambrianConfig.compose(
            Path.cwd() / "configs",
            "base",
            overrides=[*exp_overrides, *overrides],
        )

        # Run the experiment
        # Involves first creating the environment and then rendering it
        get_logger().info(f"Running {config.exp}...")
        env = exp_config.env.instance(exp_config.env)

        env.record = True
        env.reset(seed=exp_config.seed)
        for _ in range(30):
            mj.mj_step(env.model, env.data)
            env.render()
        config.outdir.mkdir(parents=True, exist_ok=True)
        env.save(
            config.outdir / fname,
            save_pkl=False,
            save_mode=MjCambrianRendererSaveMode.PNG,
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--override",
        "--overrides",
        dest="overrides",
        action="extend",
        nargs="+",
        type=str,
        help="Global override config values. Do <config>.<key>=<value>. Used for all exps.",
        default=[],
    )

    run_hydra(
        main,
        config_path=Path.cwd() / "configs" / "tools" / "paper",
        config_name="animal_showcase",
        parser=parser,
    )
