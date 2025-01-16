from dataclasses import field
from enum import Enum, auto
from functools import cached_property
from typing import Callable, Self

import mujoco as mj
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

    num_eyes: tuple[int, int] = field(default_factory=lambda: (1, 1))
    lat_range: tuple[float, float] = field(default_factory=lambda: (-90, 90))
    lon_range: tuple[float, float] = field(default_factory=lambda: (-180, 180))
    flatten_observations: bool = False
    eye_instance: Callable[[Self, str], "MjCambrianEye"] = None


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


class MjCambrianThreeBounceToFEye(MjCambrianToFEye):
    """This eye is similar to the :class:`MjCambrianToFEye`, but supports three bounce
    transients. These three bounce ToF eyes are approximated as a single bounce ToF
    sensor, where we compute the world point for each pixel the corresponds to the
    center of the IFOV of that pixel. We then do a bunch of ray casts
    from this point. The final transient is the sum of the three transients.
    """

    def reset(self, spec: MjCambrianSpec) -> ObsType:
        return super().reset(spec)

    def _update_obs(self, obs):
        # 1 bounce
        tof = self._convert_depth_to_tof(obs)
        nlos_tof = torch.zeros_like(tof)

        # For each eye, do a ray cast to get the world point to compute the 3rd bounce
        # from
        # print(self._parent_body.xpos)
        camera: mj._structs._MjDataCameraViews = self._spec.data.camera(self._name)
        pos = camera.xpos
        mat = camera.xmat.reshape(3, 3)
        height, width = self._config.resolution
        fov_h, fov_v = np.radians(self._config.fov)
        for i in range(height):
            for j in range(width):
                # Normalize pixel coordinates to [-1, 1]
                x_norm = (j - width / 2) / (width / 2)
                y_norm = (i - height / 2) / (height / 2)

                # Scale by FOV to get angles
                x_angle = x_norm * fov_h / 2
                y_angle = y_norm * fov_v / 2

                # Convert to 3D direction vector
                vec = np.array([np.tan(x_angle), np.tan(y_angle), 1])
                vec /= np.linalg.norm(vec)

                # Rotate to world coordinates
                vec = -np.dot(mat, vec)

                geomid = np.zeros(1, np.int32)
                geomgroup_mask = np.ones(6, np.uint8)
                geomgroup_mask[2] = 0  # ignore agent
                geomgroup_mask[3] = 0  # ignore walls
                distance = mj.mj_ray(
                    self._spec.model,
                    self._spec.data,
                    pos,
                    vec,
                    geomgroup_mask,  # can use to mask out agents to avoid self collision
                    1,  # include static geometries
                    -1,  # include all bodies
                    geomid,
                )
                if distance == -1:
                    continue
                geomid = geomid[0]
                geom = self._spec.model.geom(geomid)
                if geom.type != mj.mjtGeom.mjGEOM_PLANE:
                    # TODO: handle other types
                    continue

                # Get the normal of the geom
                # TODO: assumes normal is up for now
                normal = np.array([0, 0, 1], dtype=np.float32)
                tangent = np.array([1, 0, 0], dtype=np.float32)
                bitangent = np.array([0, 1, 0], dtype=np.float32)

                # Calculate a bunch of new rays to sample. Should be uniformly dist
                max_theta = np.pi / 2 - np.radians(2)
                min_theta = np.pi / 2 - np.radians(0)
                nrays = 20
                rays = []
                for _ in range(nrays):
                    # Generate random numbers
                    u1, u2 = np.random.rand(2)

                    # Spherical coordinates for cosine-weighted hemisphere sampling
                    theta = min_theta + (max_theta - min_theta) * u1
                    phi = 2 * np.pi * u2

                    # Convert to Cartesian coordinates in the local frame
                    local_ray = np.array(
                        [
                            np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(theta),
                        ]
                    )

                    # Transform the local ray to world coordinates
                    world_ray = (
                        local_ray[0] * tangent
                        + local_ray[1] * bitangent
                        + local_ray[2] * normal
                    )
                    if world_ray[2] >= 0:  # Ensure no rays go below the plane
                        rays.append(world_ray / np.linalg.norm(world_ray))
                rays = np.array(rays, dtype=np.float32)

                # Cast the rays
                geomid = np.zeros(nrays, np.int32) - 1
                dist = np.zeros(nrays, np.float64)

                geom_pos = pos + vec * distance
                geom_pos[2] = 0.1
                mj.mj_multiRay(
                    self._spec.model,
                    self._spec.data,
                    geom_pos,
                    rays.flatten(),
                    geomgroup_mask,  # can use to mask out agents to avoid self collision
                    1,
                    -1,
                    geomid,
                    dist,
                    nrays,
                    mj.mjMAXVAL,
                )
                if False:
                    from cambrian.renderer.overlays import MjCambrianSiteViewerOverlay

                    for _i, _id in enumerate(geomid):
                        if _id == -1:
                            continue
                        new_pos = geom_pos + rays[_i] * dist[_i]
                        self._spec.env._overlays[
                            f"other_site_{_i}"
                        ] = MjCambrianSiteViewerOverlay(new_pos, (0, 1, 0, 1), 0.1)

                    self._spec.env._overlays[
                        f"site_{i}_{j}"
                    ] = MjCambrianSiteViewerOverlay(geom_pos, (1, 0, 0, 1), 0.1)

                # Create transient from the rays
                dist += distance
                for _id, _d in zip(geomid, dist):
                    if _id == -1:
                        continue
                    bin_idx = int(_d * self._meters_to_bin)
                    if bin_idx < 0 or bin_idx >= self._config.num_bins:
                        continue
                    nlos_tof[bin_idx, i, j] += 1

        tof += nlos_tof

        if self._config.render_type == MjCambrianToFRenderType.DEPTH_MAP:
            self._prev_obs.copy_(obs, non_blocking=True)
        elif self._config.render_type == MjCambrianToFRenderType.HISTOGRAM:
            self._prev_obs.copy_(tof, non_blocking=True)
        else:
            raise ValueError(f"Unsupported render type {self._config.render_type}")
        return tof
