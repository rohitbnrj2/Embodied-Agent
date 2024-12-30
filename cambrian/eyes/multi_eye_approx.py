from typing import Callable, Dict, Self, Tuple

import mujoco as mj
import numpy as np
import torch

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.eyes.multi_eye import MjCambrianMultiEye, MjCambrianMultiEyeConfig
from cambrian.renderer import MjCambrianRenderer
from cambrian.renderer.render_utils import CubeToEquirectangularConverter
from cambrian.utils import MjCambrianGeometry, device, get_logger, round_half_up
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.config import config_wrapper
from cambrian.utils.spec import MjCambrianSpec


@config_wrapper
class MjCambrianMultiEyeApproxConfig(MjCambrianMultiEyeConfig):
    """Config for MjCambrianMultiEyeApprox.

    Inherits from MjCambrianApproxEyeConfig and adds additional attributes for
    an approximate multi-eye setup.
    """

    instance: Callable[[Self, str], "MjCambrianMultiEyeApprox"]


class MjCambrianApproxEye(MjCambrianEye):
    """Defines a single eye which is an approximation of an actual eye. Basically,
    it will crop/downsample from a rendered image to simulate an eye with that specific
    FOV/resolution."""

    def __init__(self, config: MjCambrianEyeConfig, name: str):
        super().__init__(config, name, disable_render=True)

        self._total_fov: Tuple[float, float] = None
        self._total_resolution: Tuple[int, int] = None
        self._crop_rect: Tuple[int, int, int, int] = None

    def compute_crop_rect(
        self,
        total_fov: Tuple[float, float],
        total_resolution: Tuple[int, int],
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
    ) -> torch.Tensor:
        """Computes the cropping rectangle for the eye."""

        # total_fovx is vertical fov (height), total_fovx is horizontal fov (width)
        total_fovy, total_fovx = total_fov
        # total_res_x is height (rows), total_res_y is width (cols)
        total_res_y, total_res_x = total_resolution
        eye_fovy, eye_fovx = self._config.fov

        # Unpack latitude and longitude ranges
        min_lat, max_lat = lat_range
        min_lon, max_lon = lon_range

        # Unpack the target coordinates (latitude, longitude)
        coord_lat, coord_lon = self._config.coord

        # Map latitude to y_center
        # Invert latitude because image y increases downward
        y_normalized = (max_lat - coord_lat) / (max_lat - min_lat)
        y_center = y_normalized * total_res_y

        # Map longitude to x_center
        # Normalize longitude within the range [0, 1] and scale to image width
        x_normalized = (max_lon - coord_lon) / (max_lon - min_lon)
        x_center = x_normalized * total_res_x

        # Compute eye resolution in pixels
        eye_res_y = (eye_fovy / total_fovy) * total_res_y
        eye_res_x = (eye_fovx / total_fovx) * total_res_x

        # Compute cropping rectangle
        y_start = int(round_half_up(y_center - eye_res_y / 2))
        y_end = int(round_half_up(y_center + eye_res_y / 2))
        x_start = int(round_half_up(x_center - eye_res_x / 2))
        x_end = int(round_half_up(x_center + eye_res_x / 2))

        # Ensure indices are within image bounds
        y_start = max(0, y_start)
        y_end = min(total_res_y, y_end)
        x_start = max(0, x_start)
        x_end = min(total_res_x, x_end)

        assert y_start < y_end, f"y_start={y_start}, y_end={y_end}"
        assert x_start < x_end, f"x_start={x_start}, x_end={x_end}"

        self._total_fov = total_fov
        self._total_resolution = total_resolution
        self._crop_rect = (y_start, y_end, x_start, x_end)

        # NOTE: yy is flipped
        yy = torch.linspace(y_end - 1, y_start, self._config.resolution[0])
        xx = torch.linspace(x_start, x_end - 1, self._config.resolution[1])
        yy = (yy / (total_res_y - 1)) * 2 - 1
        xx = (xx / (total_res_x - 1)) * 2 - 1
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).to(device)

    def reset(self, _: MjCambrianSpec):
        self._prev_obs = torch.zeros(
            (*self._config.resolution, 3),
            dtype=torch.float32,
            device=device,
        )
        return self.step(self._prev_obs)


class MjCambrianMultiEyeApprox(MjCambrianMultiEye):
    """Defines a multi-eye system by rendering images from multiple cameras facing
    different directions."""

    def __init__(
        self,
        config: MjCambrianMultiEyeApproxConfig,
        name: str,
        *,
        allow_disabling: bool = False,
    ):
        # If the number of total eyes is less than 10, we default to only MultiEye
        # methods
        self.disable = False
        if np.prod(config.num_eyes) < 10 and allow_disabling:
            get_logger().warning(
                "Number of eyes is less than 10. Defaulting to MultiEye methods."
            )
            self.disable = True
            with config.set_readonly_temporarily(False):
                config.eye_instance = MjCambrianEye
            super().__init__(config, name)
            return
        else:
            super().__init__(config, name, disable_render=True)

        self._config: MjCambrianMultiEyeApproxConfig
        self._eyes: Dict[str, MjCambrianApproxEye]

        # Create cameras for the 6 faces of the cube
        self._lats = [0, 0, 0, 0, -90, 90]
        self._lons = [135, 45, -45, -135, 45, 45]
        self._resolution = (
            max(int(self._config.resolution[0] * 90 / self._config.fov[0]), 3),
            max(int(self._config.resolution[1] * 90 / self._config.fov[1]), 3),
        )

        # Compute total FOV and resolution
        self._total_fov = (180.0, 360.0)
        self._total_resolution = (
            self._resolution[0],
            self._resolution[1] * len(self._lons),
        )

        # Set min and max lat and lon
        self._min_lon = -180.0
        self._max_lon = 180.0
        self._min_lat = -90.0
        self._max_lat = 90.0

        # Compute cropping rectangles for each eye
        grids = []
        for eye in self._eyes.values():
            grids.append(
                eye.compute_crop_rect(
                    self._total_fov,
                    self._total_resolution,
                    (self._min_lat, self._max_lat),
                    (self._min_lon, self._max_lon),
                )
            )
        self.batched_grids = torch.stack(grids, dim=0).to(device)

        # Initialize the cube to equirectangular converter
        self._cube_to_equirectangular = CubeToEquirectangularConverter(
            self._total_resolution, self._resolution
        )

        # Above or below equator where the upward looking or downward
        # looking cameras should be enabled
        phi_limit = np.degrees(np.arctan(1 / np.sqrt(2)))

        self._renderers: Dict[str, MjCambrianRenderer | torch.Tensor] = {}
        for i in range(len(self._lons)):
            renderer_name = f"{name}_renderer_{i}"

            # Check that the renderer is actually needed. A renderer isn't needed if
            # no eye is going to use any pixels from it.
            renderer_needed = False
            for eye in self._eyes.values():
                _, _, x_start, x_end = eye._crop_rect
                renderer_index_start = x_start // self._resolution[1] - 1
                renderer_index_end = x_end // self._resolution[1] - 1

                # If the eye's cropping rect overlaps this renderer, it is needed
                if renderer_index_start <= i <= renderer_index_end:
                    renderer_needed = True
                    break

                if i == 4:
                    if -eye.config.fov[0] / 2 + eye.config.coord[0] < -phi_limit:
                        get_logger().warning("Adding downward looking camera")
                        renderer_needed = True
                        break
                elif i == 5:
                    if eye.config.fov[0] / 2 + eye.config.coord[0] > phi_limit:
                        get_logger().warning("Adding upward looking camera")
                        renderer_needed = True
                        break

            if not renderer_needed:
                self._renderers[renderer_name] = torch.zeros(
                    (*self._resolution, 3), dtype=torch.float32
                ).to(device)
            else:
                self._renderers[renderer_name] = MjCambrianRenderer(config.renderer)

    def generate_xml(
        self, parent_xml: MjCambrianXML, geom: MjCambrianGeometry, parent_body_name: str
    ) -> MjCambrianXML:
        """Generates the XML for the cameras."""
        if self.disable:
            return super().generate_xml(parent_xml, geom, parent_body_name)

        xml = MjCambrianXML.make_empty()

        # Get the parent body reference
        parent_body = parent_xml.find(".//body", name=parent_body_name)
        assert parent_body is not None, f"Could not find body '{parent_body_name}'."

        # Iterate through the path and add the parent elements to the new xml
        parent = None
        elements, _ = parent_xml.get_path(parent_body)
        for element in elements:
            if (
                temp_parent := xml.find(f".//{element.tag}", **element.attrib)
            ) is not None:
                # If the element already exists, then we'll use that as the parent
                parent = temp_parent
                continue
            parent = xml.add(parent, element.tag, **element.attrib)
        assert parent is not None, f"Could not find parent for '{parent_body_name}'"

        # For each eye, add it to the xml; we won't actually render using any of the
        # cameras it creates
        for eye in self._eyes.values():
            xml += eye.generate_xml(xml, geom, parent_body_name)

        # For each camera, calculate pos and quat, and add to xml
        for lat, lon, name in zip(self._lats, self._lons, self._renderers.keys()):
            if not isinstance(self._renderers[name], MjCambrianRenderer):
                continue

            # For each camera, set up pos and quat
            pos = geom.pos  # assume camera is at the center of the geom
            _, quat = self._calculate_pos_quat(geom, (lat, lon))

            # Sensorsize calculation
            focal = self._config.focal
            sensorsize = [
                2 * focal[0] * np.tan(np.radians(90) / 2),
                2 * focal[1] * np.tan(np.radians(90) / 2),
            ]
            resolution = self._resolution

            xml.add(
                parent,
                "camera",
                name=name,
                mode="fixed",
                pos=" ".join(map(str, pos)),
                quat=" ".join(map(str, quat)),
                focal=" ".join(map(str, focal)),
                sensorsize=" ".join(map(str, sensorsize)),
                resolution=" ".join(map(str, resolution)),
            )

        return xml

    def reset(self, spec: MjCambrianSpec):
        if self.disable:
            return super().reset(spec.model, spec.data)

        # Initialize renderers for each camera
        for name, renderer_or_image in self._renderers.items():
            if not isinstance(renderer_or_image, MjCambrianRenderer):
                continue
            renderer = renderer_or_image

            renderer.reset(spec, self._resolution[1], self._resolution[0])

            fixedcamid = spec.get_camera_id(name)
            assert fixedcamid != -1, f"Camera '{name}' not found."
            renderer.viewer.camera.type = mj.mjtCamera.mjCAMERA_FIXED
            renderer.viewer.camera.fixedcamid = fixedcamid

        # Initialize each eye
        for eye in self._eyes.values():
            eye.reset(spec)

        return self.step()

    def step(self) -> Dict[str, np.ndarray]:
        if self.disable:
            return super().step()

        images = []
        for renderer_or_image in self._renderers.values():
            if not isinstance(renderer_or_image, MjCambrianRenderer):
                image = renderer_or_image
            else:
                image = renderer_or_image.render()
                if self._renders_depth:
                    image = image[0]
            images.append(image)

        full_image = self._cube_to_equirectangular.convert(images)
        full_image = full_image.permute(2, 0, 1).unsqueeze(0)

        # Batch crop
        batched = torch.nn.functional.grid_sample(
            full_image.expand(len(self._eyes), -1, -1, -1),
            self.batched_grids,
            mode="nearest",
            align_corners=False,
        )

        obs = {}
        for i, eye in enumerate(self._eyes.values()):
            obs[eye.name] = eye.step(batched[i].permute(1, 2, 0))

        return obs
