from typing import List, Dict
from pathlib import Path

import mujoco as mj

from cambrian.renderer import MjCambrianRendererSaveMode
from cambrian.utils.logger import get_logger
from cambrian.utils.config import (
    MjCambrianConfig,
    run_hydra,
)


def main(config: MjCambrianConfig):
    for fname, exp_overrides in config.overrides.items():
        get_logger().info(f"Composing animal showcase {fname}...")
        exp_config = MjCambrianConfig.compose(
            Path.cwd() / "configs",
            "base",
            overrides=[*exp_overrides, *overrides, f"exp={config.exp}"],
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
        env.save(
            config.outdir / fname,
            save_pkl=False,
            save_mode=MjCambrianRendererSaveMode.PNG,
        )

if __name__ == "__main__":
    run_hydra(main)

