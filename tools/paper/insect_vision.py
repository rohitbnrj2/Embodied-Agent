from typing import Dict
import os

import mujoco as mj

from cambrian.renderer import MjCambrianRendererSaveMode
from cambrian.utils.config import MjCambrianConfig


def create_config(overrides: Dict[str, str] = {}) -> MjCambrianConfig:
    overrides.setdefault("tools", "paper/insect_vision")
    overrides.setdefault("exp", "demos/insect_vision")
    overrides.setdefault("env.add_overlays", "False")
    overrides.setdefault("env/animals", "ant")
    overrides.setdefault("~env.animals.animal_0.instance.target_object", "goal")

    # Convert overrides to a list of strings where it's {key}={value}
    overrides = [
        f"{key}={value}" if value is not ... else key
        for key, value in overrides.items()
    ]

    return MjCambrianConfig.compose(
        f"{os.getcwd()}/configs", "base", overrides=overrides
    )


def bev_config() -> MjCambrianConfig:
    return create_config({"env.add_overlays": "True"})


def first_person_config() -> MjCambrianConfig:
    return create_config({"env.render_animal_composite_only": "animal_0"})


def third_person_config() -> MjCambrianConfig:
    return create_config({"env/renderer": "tracking"})


def run(config: MjCambrianConfig, output: str):
    print(f"Running {output}...")
    env = config.env.instance(config.env)

    env.record = True
    env.reset()

    for _ in range(30):
        mj.mj_step(env.model, env.data)
        env.render()
    env.save(
        config.expdir / output, save_pkl=False, save_mode=MjCambrianRendererSaveMode.PNG
    )

    for _ in range(100):
        env.step(env.action_spaces.sample())
        env.render()

    env.save(
        config.expdir / output, save_pkl=False, save_mode=MjCambrianRendererSaveMode.MP4
    )


if __name__ == "__main__":
    # run(create_config({"env.renderer.render_modes": "['human', 'rgb_array']"}), "human")

    run(bev_config(), "bev")
    run(first_person_config(), "first_person")
    run(third_person_config(), "third_person")
