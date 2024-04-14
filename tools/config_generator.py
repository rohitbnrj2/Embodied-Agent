from typing import Optional, List
from pathlib import Path
import os

import yaml
from omegaconf import OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra

from cambrian.utils import generate_sequence_from_range
from cambrian.utils.config import (
    run_hydra,
    MjCambrianConfig,
    MjCambrianBaseConfig,
    config_wrapper,
)


@config_wrapper
class GeneratorConfig(MjCambrianBaseConfig):
    """Config for the generator script.

    Attributes:
        base (Path): The base path for the config to generate.
        output (Path): The output path for the generated files.

        default_animal (Optional[str]): The default animal to generate. If not set, the
            first animal in the config will be used.
        num_animals (int): The number of animals to generate.

        default_eye (Optional[str]): The default eye to generate. If not set, the first
            eye in the config will be used. NOTE: There must be at least 1 eye in the
            config.
        num_eyes_lat (int): The number of latitudinal eyes to generate.
        num_eyes_lon (int): The number of longitudinal eyes to generate.
    """

    base: Path
    output: Path

    default_animal: Optional[str] = None
    num_animals: int

    default_eye: Optional[str] = None
    num_eyes_lat: int
    num_eyes_lon: int


def main(config: GeneratorConfig, *, overrides: List[str] = []):
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(f"{os.getcwd()}/configs/", version_base=None):
        exp = f"{'/'.join(config.base.parts[config.base.parts.index('exp') + 1:-1])}/{config.base.stem}"
        base = MjCambrianConfig.create(
            hydra.compose(config_name="base", overrides=[f"exp={exp}", "expname=''"]),
            instantiate=False,
        )
        OmegaConf.set_struct(base, False)

    with open(config.base, "r") as f:
        # get the first line if it's a comment
        line = f.readline()
        if line.startswith("#"):
            comment = line
        f.seek(0)
        original_base: MjCambrianConfig = MjCambrianConfig.create(
            yaml.safe_load(f), instantiate=False
        )
        original_base.merge_with_dotlist(overrides)

    env = base.env

    assert config.num_animals > 0, "Number of animals must be greater than 0"
    base_animal_name = config.default_animal or next(iter(env.animals))
    assert (
        base_animal_name in env.animals
    ), f"Animal {base_animal_name} not found in config"
    animal = env.animals[base_animal_name].copy()
    original_animal = original_base.env.animals[base_animal_name].copy()

    # Get the default entry for the animal
    for animal_default_index, animal_default in enumerate(
        original_base.defaults.to_container()
    ):
        base_animal_default_key = list(animal_default.keys())[0]
        if base_animal_name in base_animal_default_key:
            break
    else:
        raise ValueError(f"Animal {base_animal_name} not found in defaults")
    # Delete the old entry
    original_base.defaults.pop(animal_default_index)

    assert (
        config.num_eyes_lat > 0 and config.num_eyes_lon > 0
    ), "Number of eyes must be greater than 0"
    assert animal.eyes, f"Animal {base_animal_name} has no eyes"
    base_eye_name = config.default_eye or next(iter(animal.eyes))
    assert (
        base_eye_name in animal.eyes
    ), f"Eye {base_eye_name} not found in animal {base_animal_name}"
    original_eye = original_animal.eyes[base_eye_name].copy()

    # Get the default entry for the eye
    for eye_default_index, eye_default in enumerate(
        original_base.defaults.to_container()
    ):
        base_eye_default_key = list(eye_default.keys())[0]
        if base_eye_name in base_eye_default_key:
            break
    else:
        raise ValueError(f"Eye {base_eye_name} not found in defaults")
    # Delete the old entry
    original_base.defaults.pop(eye_default_index)

    # Clean the config
    config.custom = {}
    env.animals = {}
    animal.eyes = {}

    for animal_idx in range(config.num_animals):
        original_animal = original_animal.copy()
        animal_name = f"animal_{animal_idx}"

        # Add the animal
        original_base.env.animals[animal_name] = original_animal
        animal_default_key = base_animal_default_key.replace(
            base_animal_name, animal_name
        )
        original_base.defaults.insert(
            animal_default_index,
            {animal_default_key: animal_default[base_animal_default_key]},
        )
        animal_default_index += 1

        lat_eye_sequence = generate_sequence_from_range(
            animal.eyes_lat_range, config.num_eyes_lat
        )
        lon_eye_sequence = generate_sequence_from_range(
            animal.eyes_lon_range, config.num_eyes_lon
        )

        total_eyes = config.num_eyes_lat * config.num_eyes_lon
        for eye_idx in range(total_eyes):
            original_eye = original_eye.copy()
            eye_name = f"{animal_name}_eye_{eye_idx}"
            eye_default_key = base_eye_default_key.replace(base_eye_name, eye_name)
            original_base.defaults.insert(
                eye_default_index, {eye_default_key: eye_default[base_eye_default_key]}
            )
            eye_default_index += 1

            lat_idx = eye_idx // config.num_eyes_lon
            lon_idx = eye_idx % config.num_eyes_lon

            lat = lat_eye_sequence[lat_idx]
            lon = lon_eye_sequence[lon_idx]

            original_eye.coord = [lat, lon]

            original_base.env.animals[animal_name].eyes[eye_name] = original_eye

    original_base.save(
        config.output,
        header=f"{comment}\n# This is autogenerated by tools/config_generator.py.\n",
        resolve=False,
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
        help="Override config values. Do <config>.<key>=<value>",
        default=[],
    )

    run_hydra(
        main,
        parser=parser,
        config_path=f"{os.getcwd()}/configs/tools/",
        config_name="config_generator",
        is_readonly=False,
        is_struct=False,
    )
