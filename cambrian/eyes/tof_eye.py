from typing import Callable, Self

import numpy as np
import torch
from gymnasium import spaces
from hydra_config import config_wrapper

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.renderer.render_utils import convert_depth_to_rgb
from cambrian.utils.constants import C
from cambrian.utils.types import ObsType, RenderFrame


@config_wrapper
class MjCambrianToFEyeConfig(MjCambrianEyeConfig):
    """Config for MjCambrianToFEye.

    Inherits from MjCambrianEyeConfig and adds attributes for procedural eye placement.

    Attributes:
        instance (Callable[[Self, str], MjCambrianEye]): The class instance to use
            when creating the eye. Takes the config and the name of the eye as
            arguments.
    """

    instance: Callable[[Self, str], "MjCambrianToFEye"]

    num_bins: int
    timing_resolution: float


class MjCambrianToFEye(MjCambrianEye):
    """Defines an eye that outputs a ToF transient as its observation.

    Args:
        config (MjCambrianToFEyeConfig): The config for the eye.
        name (str): The name of the eye.
    """

    def __init__(self, config: MjCambrianToFEyeConfig, name: str):
        super().__init__(config, name)
        self._config: MjCambrianToFEyeConfig

        assert (
            not self._renders_rgb and self._renders_depth
        ), "ToF eye must render depth and not RGB."

        # Set the prev obs shape to the depth resolution
        self._prev_obs_shape = self.config.resolution

        # Some class variables to store so we don't need to keep recomputing
        self._meters_to_bin = 2 / C * 1e9 / self._config.timing_resolution

    def _update_obs(self, obs: ObsType) -> ObsType:
        """Updates the observation with the ToF transient."""
        tof = self._convert_depth_to_tof(obs)
        super()._update_obs(obs)
        return tof

    def _convert_depth_to_tof(self, depth: ObsType) -> ObsType:
        bin_indices = depth.mul_(self._meters_to_bin).long()
        bin_mask = bin_indices < self._config.num_bins  # can assume depth is never neg
        bin_indices.clamp_(0, self._config.num_bins - 1)

        transient = torch.nn.functional.one_hot(bin_indices, self._config.num_bins)
        transient[~bin_mask] = 0

        return transient.permute(2, 0, 1).float()

    def render(self) -> RenderFrame:
        return convert_depth_to_rgb(self._prev_obs)

    @property
    def observation_space(self) -> spaces.Box:
        """Constructs the observation space for the ToF eye as a transient with
        ``num_bins`` and each frame having a resolution of ``resolution``."""

        shape = (self._config.num_bins, *self._config.resolution)
        return spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
