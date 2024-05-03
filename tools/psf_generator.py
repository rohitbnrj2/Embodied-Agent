from typing import Optional
from pathlib import Path
from enum import Enum

import numpy as np

from cambrian.eyes import MjCambrianEyeConfig, MjCambrianOpticsEyeConfig
from cambrian.utils.config import (
    MjCambrianBaseConfig,
    run_hydra,
    config_wrapper,
)


class RandomType(Enum):
    UNIFORM = "uniform"
    FROM_ZEROS = "from_zeros"


@config_wrapper
class RandomInitConfig:
    type: RandomType


class RadialType(Enum):
    UNIFORM = "uniform"
    FROM_ZEROS = "from_zeros"


@config_wrapper
class RadialInitConfig:
    type: RadialType


class InitType(Enum):
    RANDOM = "random"
    RADIAL = "radial"
    CONSTANT = "constant"


@config_wrapper
class PSFGeneratorConfig(MjCambrianBaseConfig):
    """Config for the PSF generator.

    Attributes:
        init (InitType): The type of initialization for the height map.

        quantize (bool): Whether to quantize the height map. This basically bins the
            height map to a few discrete values.
        random (Optional[RandomInitConfig]): Config for random initialization.
        radial (Optional[RadialInitConfig]): Config for radial initialization.
        constant (Optional[float]): Constant value for the height map.

        eye (MjCambrianOpticsEyeConfig): Config for the eye.
    """

    output: Path

    init: InitType

    quantize: bool
    random: Optional[RandomInitConfig] = None
    radial: Optional[RadialInitConfig] = None
    constant: Optional[float] = None

    eye: MjCambrianEyeConfig | MjCambrianOpticsEyeConfig


def create_random_height_map(config: PSFGeneratorConfig) -> np.ndarray:
    assert config.random is not None
    if config.random.type is RandomType.UNIFORM:
        hmap = np.random.uniform(0, 1, config.eye.pupil_resolution)
    elif config.random.type is RandomType.FROM_ZEROS:
        hmap = np.zeros(config.eye.pupil_resolution)
    return hmap


def create_radial_height_map(config: PSFGeneratorConfig) -> np.ndarray:
    assert config.radial is not None

    # radially symmetric so hmap is a function of r only
    Mx, My = config.eye.pupil_resolution
    maxr = np.sqrt((Mx / 2) ** 2 + (My / 2) ** 2)

    Lx, Ly = config.eye.sensorsize
    x1 = np.linspace(-Lx / 2.0, Lx / 2.0, Mx)
    y1 = np.linspace(-Ly / 2.0, Ly / 2.0, My)
    X1, Y1 = np.meshgrid(x1, y1)

    if config.radial.type is RadialType.UNIFORM:
        h_r = np.random.uniform(0, 1, np.ceil(maxr).astype(int))
    elif config.radial.type is RadialType.FROM_ZEROS:
        h_r = np.zeros(np.ceil(maxr).astype(int)) + 0.5

    # create hmap
    x, y = np.meshgrid(np.arange(Mx), np.arange(My))
    r = np.sqrt((x - Mx / 2) ** 2 + (y - My / 2) ** 2)
    hmap = h_r[r.astype(int)]
    print(hmap.shape)

    return hmap


def create_constant_height_map(config: PSFGeneratorConfig) -> np.ndarray:
    assert config.constant is not None
    return np.full(config.eye.pupil_resolution, config.constant)


def create_height_map(config: PSFGeneratorConfig) -> np.ndarray:
    if config.init is InitType.RANDOM:
        hmap = create_random_height_map(config)
    elif config.init is InitType.RADIAL:
        hmap = create_radial_height_map(config)
    elif config.init is InitType.CONSTANT:
        hmap = create_constant_height_map(config)

    if config.quantize:
        # quantize hmap to {0, 0.25, 0.5, 0.75, 1.0}
        # small perturbation in the hmap can result in large changes in the PSF
        hmap = np.round(hmap * 4) / 4

    # scale hmap & clamp hmap
    wavelengths = np.asarray(config.eye.wavelengths)
    max_height = np.max(1 / (wavelengths * (config.eye.refractive_index - 1.0 + 1e-8)))
    hmap = np.clip(hmap, 0, max_height)
    # hmap = np.interp(hmap, (hmap.min(), hmap.max()), (0, max_height))

    return hmap


def main(config: PSFGeneratorConfig):
    hmap = create_height_map(config)

    # Save the height map as a plane txt
    print(f"Saving height map to {config.output}")
    hmap_str = "[" + ", ".join([str(row) for row in hmap.tolist()]) + "]"
    with open(config.output, "w") as f:
        f.write(hmap_str)


if __name__ == "__main__":
    run_hydra(main, config_name="tools/psf_generator")
