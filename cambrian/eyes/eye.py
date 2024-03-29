from typing import Tuple, Optional
import numpy as np

import mujoco as mj
from gymnasium import spaces

from cambrian.renderer import MjCambrianRenderer, MjCambrianRendererConfig
from cambrian.eyes.optics import MjCambrianOptics, MjCambrianOpticsConfig
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.cambrian_xml import MjCambrianXML


@config_wrapper
class MjCambrianEyeConfig(MjCambrianBaseConfig):
    """Defines the config for an eye. Used for type hinting.

    Attributes:
        pos (Tuple[float, float, float]): The initial position of the camera.
            This is computed by the animal from the coord during placement. Fmt: xyz
        quat (Tuple[float, float, float, float]): The initial rotation of the
            camera. This is computed by the animal from the coord during placement.
            Fmt: wxyz.
        fov (Tuple[float, float]): Independent of the `fovy` field in the MJCF
            xml. Used to calculate the sensorsize field. Specified in degrees. Mutually
            exclusive with `fovy`. If `focal` is unset, it is set to 1, 1. Will override
            `sensorsize`, if set. Fmt: fovx fovy.
        focal (Tuple[float, float]): The focal length of the camera.
            Fmt: focal_x focal_y.
        sensorsize (Tuple[float, float]): The size of the sensor. Fmt: width height.
        resolution (Tuple[int, int]): The width and height of the rendered image.
            Fmt: width height.
        coord (Tuple[float, float]): The x and y coordinates of the eye.
            This is used to determine the placement of the eye on the animal.
            Specified in degrees. This attr isn't actually used by eye, but by the
            animal. The eye has no knowledge of the geometry it's trying to be placed
            on. Fmt: lat lon

        use_depth_obs (bool): Whether to use depth observations. If True, the depth
            observation will be included in the observation space. If False, only the
            rgb observation will be included.
        optics (Optional[MjCambrianOpticsConfig]): The optics config to use for the eye.
            Optics is disabled if this is unset.

        renderer (MjCambrianRendererConfig): The renderer config to use for the
            underlying renderer. The width and height of the renderer will be set to the
            padded resolution (resolution + int(psf_filter_size/2)) of the eye.
    """

    pos: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]
    fov: Tuple[float, float]
    focal: Tuple[float, float]
    sensorsize: Tuple[float, float]
    resolution: Tuple[int, int]
    coord: Tuple[float, float]

    use_depth_obs: bool
    optics: Optional[MjCambrianOpticsConfig] = None

    renderer: MjCambrianRendererConfig


class MjCambrianEye:
    """Defines an eye for the cambrian environment. It essentially wraps a mujoco Camera
    object and provides some helper methods for rendering and generating the XML. The
    eye is attached to the parent body such that movement of the parent body will move
    the eye.

    Args:
        config (MjCambrianEyeConfig): The configuration for the eye.
        name (str): The name of the eye.
    """

    def __init__(self, config: MjCambrianEyeConfig, name: str):
        self.config = config
        self.name = name

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._prev_obs: np.ndarray = None

        self._optics: MjCambrianOptics = None
        if self.config.optics is not None:
            self._optics = MjCambrianOptics(self.config.optics)
            assert (
                "depth_array" in self.config.renderer.render_modes
            ), "Must specify 'depth_array' in the render modes for the renderer config."

            # If optics is enabled, update the renderer w/h to the padded resolution
            self.config.renderer.width = self.config.optics.padded_resolution[0]
            self.config.renderer.height = self.config.optics.padded_resolution[1]

        assert (
            "rgb_array" in self.config.renderer.render_modes
        ), "Must specify 'rgb_array' in the render modes for the renderer config."
        self._renderer = MjCambrianRenderer(self.config.renderer)

    def generate_xml(
        self, parent_xml: MjCambrianXML, parent_body_name: str
    ) -> MjCambrianXML:
        """Generate the xml for the eye.

        In order to combine the xml for an eye with the xml for the animal that it's
        attached to, we need to replicate the path with which we want to attach the eye.
        For instance, if the body with which we want to attach the eye to is at
        `mujoco/worldbody/torso`, then we need to replicate that path in the new xml.
        This is kind of difficult with the `xml` library, but we'll utilize the
        `CambrianXML` helpers for this.

        Args:
            parent_xml (MjCambrianXML): The xml of the parent body. Used as a reference to
            extract the path of the parent body.
            parent_body_name (str): The name of the parent body. Will search for the
            body tag with this name, i.e. <body name="<parent_body_name>" ...>.
        """

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

        # Finally add the camera element at the end
        resolution = [self._renderer.config.width, self._renderer.config.height]
        xml.add(
            parent,
            "camera",
            name=self.name,
            mode="fixed",
            pos=" ".join(map(str, self.config.pos)),
            quat=" ".join(map(str, self.config.quat)),
            focal=" ".join(map(str, self.config.focal)),
            sensorsize=" ".join(map(str, self.config.sensorsize)),
            resolution=" ".join(map(str, resolution)),
        )

        return xml

    def reset(self, model: mj.MjModel, data: mj.MjData):
        """Sets up the camera for rendering. This should be called before rendering
        the first time."""
        self._model = model
        self._data = data

        self._renderer.reset(model, data, *self.config.resolution)

        fixedcamid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, self.name)
        assert fixedcamid != -1, f"Camera '{self.name}' not found."
        self._renderer.viewer.camera.type = mj.mjtCamera.mjCAMERA_FIXED
        self._renderer.viewer.camera.fixedcamid = fixedcamid

        self._prev_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return self.step()

    def step(self) -> np.ndarray:
        """Simply calls `render` and sets the last observation.
        See `render()` for more information."""
        obs = self.render()
        np.copyto(self._prev_obs, obs)
        return obs

    def render(self) -> np.ndarray:
        """Render the image from the camera. If `return_depth` is True, returns the
        both depth image and rgb, otherwise only rgb.

        NOTE: The actual rendered resolution is `self.sensor_resolution`. This is so
        that the convolution with the psf filter doesn't cut off any of the image. After
        the psf is applied, the image will be cropped to `self.resolution`.
        """

        if "depth_array" in self.config.renderer.render_modes:
            rgb, depth = self._renderer.render()
        else:
            rgb = self._renderer.render()

        if self._optics is not None:
            rgb = self._optics.step(rgb, depth)

        if self.config.use_depth_obs:
            return np.concatenate([rgb, depth[..., None]], axis=-1)
        return rgb

    @property
    def observation_space(self) -> spaces.Box:
        """Constructs the observation space for the eye. The observation space is a
        `spaces.Box` with the shape of the resolution of the eye. If `use_depth_obs` is
        True, then the observation space will have 4 channels (r, g, b, depth). If
        `use_depth_obs` is False, then the observation space will have 3 channels (r, g,
        b). The values are in the range [0, 1]."""

        shape = (*self.config.resolution, 4 if self.config.use_depth_obs else 3)
        return spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)

    @property
    def num_pixels(self) -> int:
        """The number of pixels in the image."""
        return np.prod(self.config.resolution)

    @property
    def prev_obs(self) -> np.ndarray:
        """The last observation returned by `self.render()`."""
        return self._prev_obs
