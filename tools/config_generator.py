from typing import Tuple, Optional
from functools import partial
import random
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from cambrian.utils import MjCambrianArgumentParser, generate_sequence_from_range
from cambrian.utils.config import MjCambrianConfig, MjCambrianEyeConfig, MjCambrianAnimalConfig

list_repr = "tag:yaml.org,2002:seq"
yaml.add_representer(list, lambda d, seq: d.represent_sequence(list_repr, seq, True))
yaml.add_representer(tuple, lambda d, seq: d.represent_sequence(list_repr, seq, True))

def generate_demos(args):
    import subprocess

    output = Path(args.output)
    assert output.is_dir(), f"Output {output} is not a folder"

    demos = [
        ("compound_eye.yaml", "--num-eyes 10,10 --uniform-eyes"),
        ("simple_eye.yaml", "--num-eyes 1,1 --uniform-eyes"),
        (
            "human_eye.yaml",
            "--num-eyes 1,1 --uniform-eyes -eo resolution='[1000, 1000]' fov='[120, 120]'",
        ),
        ("random_eye.yaml", "--num-eyes 100"),
    ]

    for filename, cmd_args in demos:
        cmd = f"python {__file__} {args.config} {output / filename} {cmd_args}"
        if len(args.eye_overrides):
            cmd += f"-eo {' '.join(args.eye_overrides)}"
        if args.resolve:
            cmd += " --resolve"

        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True)


def int_or_tuple(x: str, *, num: Optional[int] = None) -> int | Tuple[int, ...]:
    def convert_to_tuple(x, delimiter):
        return tuple(int(y) for y in x.split(delimiter))

    if "," in x:
        res = convert_to_tuple(x, ",")
        if num is not None and len(res) != num:
            raise ValueError(f"Expected {num} values, got {len(res)}")
        return res
    else:
        return int(x)


if __name__ == "__main__":
    parser = MjCambrianArgumentParser()

    parser.add_argument("output", type=str, help="Output config file")

    parser.add_argument(
        "-eo",
        "--eye-overrides",
        type=str,
        nargs="+",
        action="extend",
        help="Overrides for eye configs. Do <config>.<key>=<value>.",
        default=[],
    )

    parser.add_argument(
        "--num-eyes",
        type=partial(int_or_tuple, num=2),
        help="Number of eyes to use. If `--uniform-eyes` is passed, this should be a tuple of 2 ints and it will represent the number of horizontal and vertical eyes. Pass as `--num-eyes 2,3` to use 2 horizontal eyes and 3 vertical eyes. Otherwise, this should be a single int and it will represent the total number of randomly generated eyes.",
    )
    parser.add_argument(
        "--random-eyes",
        action="store_true",
        help="Use randomly generated eyes. Otherwise, will copy the eyes from the base config.",
    )
    parser.add_argument(
        "--uniform-eyes",
        action="store_true",
        help="Use uniformly generated eyes. Otherwise, will randomly select eye positions.",
    )
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="Resolve the config. This will fill in any missing values with defaults and resolve interpolations. It is recommended to _not_ resolve for readibility. For loading speed, resolving is better.",
    )

    parser.add_argument(
        "--generate-demos",
        action="store_true",
        help="Generate demo configs. Output is interpreted as a folder in this case.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    random.seed(args.seed)

    if args.generate_demos:
        generate_demos(args)
        exit()

    assert args.num_eyes is not None, "Expected `--num-eyes` to be passed"
    num_eyes_total = args.num_eyes
    if args.uniform_eyes:
        assert isinstance(args.num_eyes, Tuple), "Expected tuple for `--num-eyes`"
        num_lat, num_lon = args.num_eyes
        num_eyes_total = num_lon * num_lat

    # duck typed if resolve is false
    config = MjCambrianConfig.load(
        args.config,
        overrides=args.overrides,
        resolve=args.resolve,
        instantiate=args.resolve,
    )
    resolved_config = MjCambrianConfig.load(
        args.config, overrides=args.overrides, instantiate=False
    )

    for animal_idx, animal_config in enumerate(
        config.env_config.animal_configs.values()
    ):
        animal_config.setdefault("name", f"animal_{animal_idx}")

        key = f"env_config.animal_configs.{animal_config.name}"
        resolved_animal_config: MjCambrianAnimalConfig = OmegaConf.select(
            resolved_config, key
        )

        eye_configs = {}
        for eye_idx in range(num_eyes_total):
            eye_config = random.choice(list(animal_config.eye_configs.values())).copy()
            eye_config.merge_with_dotlist(args.eye_overrides)
            eye_config.setdefault("name", f"{animal_config.name}_eye_{eye_idx}")

            key = f"eye_configs.{eye_config.name}"
            resolved_eye_config: MjCambrianEyeConfig = OmegaConf.select(
                resolved_animal_config, key
            )

            if args.random_eyes:

                def edit(attrs, low, high):
                    return [int(random.uniform(low, high)) for _ in range(attrs)]

                eye_config.resolution = edit(resolved_eye_config.resolution, 1, 1000)
                eye_config.fov = edit(resolved_eye_config.fov, 1, 180)

            if args.uniform_eyes:
                longitudes = generate_sequence_from_range(
                    resolved_animal_config.model_config.eyes_lon_range, num_lon
                )
                latitudes = generate_sequence_from_range(
                    resolved_animal_config.model_config.eyes_lat_range, num_lat
                )

                lon = float(longitudes[eye_idx % num_lon])
                lat = float(latitudes[eye_idx // num_lon])
            else:
                lon = random.uniform(
                    *resolved_animal_config.model_config.eyes_lon_range
                )
                lat = random.uniform(
                    *resolved_animal_config.model_config.eyes_lat_range
                )

            eye_config.coord = [lat, lon]

            eye_config.name = f"{animal_config.name}_eye_{eye_idx}"
            eye_configs[eye_config.name] = eye_config

        animal_config.eye_configs = eye_configs

    with open(args.output, "w") as f:
        yaml.dump(OmegaConf.to_container(config), f)
