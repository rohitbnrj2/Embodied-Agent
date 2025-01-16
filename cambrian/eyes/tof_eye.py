from enum import Enum, auto
from functools import cached_property
from typing import Callable, Self

import numpy as np
import torch
from gymnasium import spaces
from hydra_config import config_wrapper

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.renderer.render_utils import DepthDistanceConverter
from cambrian.utils import device
from cambrian.utils.constants import C
from cambrian.utils.spec import MjCambrianSpec
from cambrian.utils.types import ObsType, RenderFrame


class MjCambrianToFRenderType(Enum):
    DEPTH_MAP = auto()
    HISTOGRAM = auto()


@config_wrapper
class MjCambrianToFEyeConfig(MjCambrianEyeConfig):
    """Config for MjCambrianToFEye.

    Inherits from MjCambrianEyeConfig and adds attributes for procedural eye placement.

    Attributes:
        instance (Callable[[Self, str], MjCambrianEye]): The class instance to use
            when creating the eye. Takes the config and the name of the eye as
            arguments.

        num_bins (int): The number of bins to use for the ToF transient.
        timing_resolution_ns (float): The timing resolution of the ToF transient in ns.
        subsampling_factors (tuple[int, int]): The sub-sampling factor to use for the
            ToF  transient. Used to reduce the resolution of the ToF transient. Fmt:
            [h, w]. Must be a factor of the resolution.
    """

    instance: Callable[[Self, str], "MjCambrianToFEye"]

    num_bins: int
    timing_resolution_ns: float
    subsampling_factor: tuple[int, int]

    render_type: MjCambrianToFRenderType


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
        if self._config.render_type == MjCambrianToFRenderType.DEPTH_MAP:
            self._prev_obs_shape = self._config.resolution
        elif self._config.render_type == MjCambrianToFRenderType.HISTOGRAM:
            self._prev_obs_shape = (self._config.num_bins, *self.subsampled_resolution)
            self._histogram_image = torch.zeros(
                (self._config.num_bins, self._config.num_bins),
                device=device,
                dtype=torch.float32,
            )
            self._histogram_indices = torch.arange(
                self._config.num_bins, device=device, dtype=torch.int32
            ).unsqueeze(1)
        else:
            raise ValueError(f"Unsupported render type {self._config.render_type}")

        # Some class variables to store so we don't need to keep recomputing
        self._meters_to_bin = 2 / C * 1e9 / self._config.timing_resolution_ns
        self._depth_converter: DepthDistanceConverter = None

    def reset(self, spec: MjCambrianSpec) -> ObsType:
        # The converter to convert depth to distances
        self._depth_converter = DepthDistanceConverter(spec.model)

        return super().reset(spec)

    def _update_obs(self, obs: ObsType) -> ObsType:
        """Updates the observation with the ToF transient."""

        tof = self._convert_depth_to_tof(obs)
        if self._config.render_type == MjCambrianToFRenderType.DEPTH_MAP:
            self._prev_obs.copy_(obs, non_blocking=True)
        elif self._config.render_type == MjCambrianToFRenderType.HISTOGRAM:
            self._prev_obs.copy_(tof, non_blocking=True)
        else:
            raise ValueError(f"Unsupported render type {self._config.render_type}")
        return tof

    def _convert_depth_to_tof(self, depth: ObsType) -> ObsType:
        depth = self._depth_converter.convert(depth)
        bin_indices = depth.mul_(self._meters_to_bin).long()
        bin_mask = bin_indices < self._config.num_bins  # can assume depth is never neg
        bin_indices.clamp_(0, self._config.num_bins - 1)

        transient = torch.nn.functional.one_hot(bin_indices, self._config.num_bins)
        transient[~bin_mask] = 0

        height_sub, width_sub = self._config.subsampling_factor
        transient = (
            torch.nn.functional.avg_pool2d(
                transient.permute(2, 0, 1).float().unsqueeze(0),
                kernel_size=(height_sub, width_sub),
                stride=(height_sub, width_sub),
                count_include_pad=False,
            )
            * height_sub
            * width_sub
        )
        transient = transient.squeeze(0)
        transient = transient / transient.max()

        return transient

    def render(self) -> RenderFrame:
        if self._config.render_type == MjCambrianToFRenderType.DEPTH_MAP:
            return super().render()
        elif self._config.render_type == MjCambrianToFRenderType.HISTOGRAM:
            histogram = self._prev_obs.sum(dim=(1, 2))
            histogram /= histogram.max()
            histogram_scaled = (histogram * self._config.num_bins).to(torch.int32)
            mask = self._histogram_indices < histogram_scaled
            self._histogram_image.fill_(0.0)
            self._histogram_image[mask] = 1.0
            return self._histogram_image.repeat(3, 1, 1).permute(1, 2, 0)
        else:
            raise ValueError(f"Unsupported render type {self._config.render_type}")

    @property
    def observation_space(self) -> spaces.Box:
        """Constructs the observation space for the ToF eye as a transient with
        ``num_bins`` and each frame having a resolution of ``resolution``."""

        shape = (self._config.num_bins, *self.subsampled_resolution)
        return spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)

    @cached_property
    def subsampled_resolution(self) -> tuple[int, int]:
        """Returns the resolution of the ToF transient."""
        height, width = self._config.resolution
        height_sub, width_sub = self._config.subsampling_factor
        assert height % height_sub == 0, "Height must be divisible by sub_sampling."
        assert width % width_sub == 0, "Width must be divisible by sub_sampling."
        return height // height_sub, width // width_sub


class MjCambrianMultiBounceToFEye(MjCambrianToFEye):
    """This eye is similar to the :class:`MjCambrianToFEye`"""
