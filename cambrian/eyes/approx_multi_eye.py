from typing import Callable, Dict, Self, Tuple

import cv2
import mujoco as mj
import numpy as np
from mujoco import MjData, MjModel
from scipy.spatial.transform import Rotation as R

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.eyes.multi_eye import MjCambrianMultiEye, MjCambrianMultiEyeConfig
from cambrian.renderer import MjCambrianRenderer
from cambrian.utils import MjCambrianGeometry, get_camera_id
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.config import config_wrapper


@config_wrapper
class MjCambrianApproxMultiEyeConfig(MjCambrianMultiEyeConfig):
    """Config for MjCambrianApproxMultiEye.

    Inherits from MjCambrianApproxEyeConfig and adds additional attributes for
    an approximate multi-eye setup.
    """

    instance: Callable[[Self, str], "MjCambrianApproxMultiEye"]


class MjCambrianApproxEye(MjCambrianEye):
    """Defines a single eye which is an approximation of an actual eye. Basically,
    it will crop/downsample from a rendered image to simulate an eye with that specific
    FOV/resolution."""

    def __init__(self, config: MjCambrianEyeConfig, name: str):
        super().__init__(config, name, disable_render=True)

        self._total_fov: Tuple[float, float] = None
        self._total_resolution: Tuple[int, int] = None
        self._crop_rect: Tuple[int, int, int, int] = None

    def generate_xml(self, *args, **kwargs) -> MjCambrianXML:
        """Generates an empty XML."""
        return MjCambrianXML.make_empty()

    def compute_crop_rect(
        self,
        total_fov: Tuple[float, float],
        total_resolution: Tuple[int, int],
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
    ):
        """Computes the cropping rectangle for the eye."""

        # total_fovx is horizontal fov (width), total_fovy is vertical fov (height)
        total_fovx, total_fovy = total_fov
        # total_res_x is width (columns), total_res_y is height (rows)
        total_res_x, total_res_y = total_resolution
        eye_fovx, eye_fovy = self._config.fov

        # Unpack latitude and longitude ranges
        min_lat, max_lat = lat_range
        min_lon, max_lon = lon_range

        # Unpack the target coordinates (latitude, longitude)
        coord_lat, coord_lon = self._config.coord

        # Map longitude to x_center
        # Normalize longitude within the range [0, 1] and scale to image width
        x_normalized = (max_lon - coord_lon) / (max_lon - min_lon)
        x_center = x_normalized * total_res_x

        # Map latitude to y_center
        # Invert latitude because image y increases downward
        y_normalized = (max_lat - coord_lat) / (max_lat - min_lat)
        y_center = y_normalized * total_res_y

        # Compute eye resolution in pixels
        eye_res_x = (eye_fovx / total_fovx) * total_res_x
        eye_res_y = (eye_fovy / total_fovy) * total_res_y

        # Compute cropping rectangle
        x_start = int(np.round(x_center - eye_res_x / 2))
        x_end = int(np.round(x_center + eye_res_x / 2))
        y_start = int(np.round(y_center - eye_res_y / 2))
        y_end = int(np.round(y_center + eye_res_y / 2))

        # Ensure indices are within image bounds
        x_start = max(0, x_start)
        x_end = min(total_res_x, x_end)
        y_start = max(0, y_start)
        y_end = min(total_res_y, y_end)

        self._total_fov = total_fov
        self._total_resolution = total_resolution
        self._crop_rect = (x_start, x_end, y_start, y_end)

    def reset(self, model: MjModel, data: MjData):
        self._model = model
        self._data = data

        self._prev_obs = np.zeros((*self._config.resolution, 3), dtype=np.float32)
        return self.step(np.zeros((*self._total_resolution, 3)))

    def step(self, full_image: np.ndarray) -> np.ndarray:
        """Renders the image and returns the observation for the eye."""
        # Crop and resize to eye resolution
        x_start, x_end, y_start, y_end = self._crop_rect
        eye_image = full_image[x_start:x_end, y_start:y_end, :]
        resized_eye_image = self._resize_image(eye_image, *self._config.resolution)
        return super().step(resized_eye_image)

    def _resize_image(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resizes the image to the given size using bilinear interpolation."""
        if image.shape[0] == width and image.shape[1] == height:
            return image
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)


class MjCambrianApproxMultiEye(MjCambrianMultiEye):
    """Defines a multi-eye system by rendering images from multiple cameras facing
    different directions."""

    def __init__(self, config: MjCambrianApproxMultiEyeConfig, name: str):
        super().__init__(config, name, disable_render=True)
        self._config: MjCambrianApproxMultiEyeConfig
        self._eyes: Dict[str, MjCambrianApproxEye]

        # Create cameras for the 4 directions
        self._lat = np.add(*self._config.lat_range) / 2.0
        self._lons = [225, 135, 45, -45]
        self._resolution = (
            self._config.resolution[0] * self._config.num_eyes[1],
            self._config.resolution[1] * self._config.num_eyes[0],
        )
        self._renderers: Dict[str, MjCambrianRenderer] = {}
        for i in range(4):
            renderer_name = f"{name}_renderer_{i}"
            self._renderers[renderer_name] = MjCambrianRenderer(config.renderer)

        # Compute total FOV and resolution
        lat_range = max(np.subtract(*self._config.lat_range), self._config.fov[1])
        self._total_fov = (360.0, lat_range)
        self._total_resolution = (self._resolution[0] * 4, self._resolution[1])

        # Set min and max lat and lon
        self._min_lon = -180.0
        self._max_lon = 180.0
        self._min_lat = self._config.lat_range[0] - self._config.fov[1] / 2.0
        self._max_lat = self._config.lat_range[1] + self._config.fov[1] / 2.0

        # Compute cropping rectangles for each eye
        for eye in self._eyes.values():
            eye.compute_crop_rect(
                self._total_fov,
                self._total_resolution,
                (self._min_lat, self._max_lat),
                (self._min_lon, self._max_lon),
            )

    def generate_xml(
        self, parent_xml: MjCambrianXML, geom: MjCambrianGeometry, parent_body_name: str
    ) -> MjCambrianXML:
        """Generates the XML for the cameras."""
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

        # For each camera, calculate pos and quat, and add to xml
        for lon, name in zip(self._lons, self._renderers.keys()):
            # For each camera, set up pos and quat
            pos, quat = self._calculate_camera_pose(geom, lon)

            # Sensorsize calculation
            focal = self._config.focal
            sensorsize = [
                2 * focal[0] * np.tan(np.radians(90) / 2),
                2 * focal[1] * np.tan(np.radians(self._config.fov[1]) / 2),
            ]
            resolution = [self._config.resolution[0], self._config.resolution[1]]

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

    def _calculate_camera_pose(
        self, geom: MjCambrianGeometry, yaw: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the position and quaternion for the camera facing at yaw
        degrees."""
        pos = geom.pos  # Assuming camera is at the center of the geom
        default_rot = R.from_euler("z", np.pi / 2)
        quat = (R.from_euler("y", -np.deg2rad(yaw)) * default_rot).as_quat()
        return pos, quat

    def reset(self, model: MjModel, data: MjData):
        self._model = model
        self._data = data

        # Initialize renderers for each camera
        for name, renderer in self._renderers.items():
            renderer.reset(model, data, *self._resolution)

            fixedcamid = get_camera_id(model, name)
            assert fixedcamid != -1, f"Camera '{name}' not found."
            renderer.viewer.camera.type = mj.mjtCamera.mjCAMERA_FIXED
            renderer.viewer.camera.fixedcamid = fixedcamid

        # Initialize each eye
        for eye in self._eyes.values():
            eye.reset(model, data)

        return self.step()

    def step(self) -> Dict[str, np.ndarray]:
        """Renders the images from all cameras, stitches them, and
        returns observations for each eye."""
        images = []
        for renderer in self._renderers.values():
            image = renderer.render()
            if self._renders_depth:
                image = image[0]  # Assuming RGB image is the first element
            images.append(image)
        # Now stitch the images together
        # full_image = project_images_to_spherical_panorama(
        #     images=images,
        #     yaw_angles=self._lons,
        #     fov_x=90,
        #     fov_y=self._config.fov[1],
        #     total_resolution=self._total_resolution,
        # )
        full_image = np.concatenate(images, axis=0)  # Concatenate along width
        # Save image for debugging
        # if "i" not in globals():
        #     global i
        #     i = 0
        # cv2.imwrite(
        #     f"logs/images/full_image_{i}.png",
        #     (np.swapaxes(full_image, 0, 1) * 255).astype(np.uint8),
        # )
        # i += 1
        obs = {}
        for name, eye in self._eyes.items():
            obs[name] = eye.step(full_image)
        return obs
