from typing import Tuple, Callable, Self
import numpy as np

import mujoco as mj
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from cambrian.renderer import MjCambrianRenderer, MjCambrianRendererConfig
from cambrian.utils import MjCambrianGeometry, get_camera_id
from cambrian.utils.config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.cambrian_xml import MjCambrianXML


@config_wrapper
class MjCambrianEyeConfig(MjCambrianBaseConfig):
    """Defines the config for an eye. Used for type hinting.

    Attributes:
        instance (Callable[[Self, str], MjCambrianEye]): The class instance to use
            when creating the eye. Takes the config and the name of the eye as
            arguments.

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
            This is used to determine the placement of the eye on the agent.
            Specified in degrees. This attr isn't actually used by eye, but by the
            agent. The eye has no knowledge of the geometry it's trying to be placed
            on. Fmt: lat lon

        renderer (MjCambrianRendererConfig): The renderer config to use for the
            underlying renderer. The width and height of the renderer will be set to the
            padded resolution (resolution + int(psf_filter_size/2)) of the eye.
    """

    instance: Callable[[Self, str], "MjCambrianEye"]

    fov: Tuple[float, float]
    focal: Tuple[float, float]
    sensorsize: Tuple[float, float]
    resolution: Tuple[int, int]
    coord: Tuple[float, float]

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
        self._config = config
        self._name = name

        self._renders_rgb = "rgb_array" in self._config.renderer.render_modes
        self._renders_depth = "depth_array" in self._config.renderer.render_modes
        assert self._renders_rgb, f"Eye ({name}): 'rgb_array' must be a render mode."

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._prev_obs: np.ndarray = None
        self._fixedcamid = -1

        self._renderer = MjCambrianRenderer(self._config.renderer)

    def generate_xml(
        self, parent_xml: MjCambrianXML, geom: MjCambrianGeometry, parent_body_name: str
    ) -> MjCambrianXML:
        """Generate the xml for the eye.

        In order to combine the xml for an eye with the xml for the agent that it's
        attached to, we need to replicate the path with which we want to attach the eye.
        For instance, if the body with which we want to attach the eye to is at
        `mujoco/worldbody/torso`, then we need to replicate that path in the new xml.
        This is kind of difficult with the `xml` library, but we'll utilize the
        `CambrianXML` helpers for this.

        Args:
            parent_xml (MjCambrianXML): The xml of the parent body. Used as a reference to
                extract the path of the parent body.
            geom (MjCambrianGeometry): The geometry of the parent body. Used to
                calculate the pos and quat of the eye.
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
        pos, quat = self._calculate_pos_quat(geom)
        resolution = [self._renderer.config.width, self._renderer.config.height]
        xml.add(
            parent,
            "camera",
            name=self._name,
            mode="fixed",
            pos=" ".join(map(str, pos)),
            quat=" ".join(map(str, quat)),
            focal=" ".join(map(str, self._config.focal)),
            sensorsize=" ".join(map(str, self._config.sensorsize)),
            resolution=" ".join(map(str, resolution)),
        )

        return xml

    def _calculate_pos_quat(
        self, geom: MjCambrianGeometry
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the position and quaternion of the eye based on the geometry of
        the parent body. The position is calculated by moving the eye to the edge of the
        geometry in the negative x direction. The quaternion is calculated by rotating
        the eye to face the center of the geometry.

        TODO: rotations are weird. fix this.
        """
        lat, lon = np.deg2rad(self._config.coord)
        lon += np.pi / 2

        default_rot = R.from_euler("z", np.pi / 2)
        pos_rot = default_rot * R.from_euler("yz", [lat, lon])
        rot_rot = R.from_euler("z", lat) * R.from_euler("y", -lon) * default_rot

        pos = pos_rot.apply([-geom.rbound, 0, 0]) + geom.pos
        quat = rot_rot.as_quat()
        return pos, quat

    def reset(self, model: mj.MjModel, data: mj.MjData):
        """Sets up the camera for rendering. This should be called before rendering
        the first time."""
        self._model = model
        self._data = data

        self._renderer.reset(model, data, *self._config.resolution)

        self._fixedcamid = get_camera_id(model, self._name)
        assert self._fixedcamid != -1, f"Camera '{self._name}' not found."
        self._renderer.viewer.camera.type = mj.mjtCamera.mjCAMERA_FIXED
        self._renderer.viewer.camera.fixedcamid = self._fixedcamid

        self._prev_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return self.step()

    def step(self) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Simply calls `render` and sets the last observation. See `render()` for more
        information.
        """
        obs = self.render()
        np.copyto(self._prev_obs, obs)
        return obs

    def render(self) -> np.ndarray:
        """Render the image from the camera. Will always only return the rgb array."""
        obs = self._renderer.render()
        if self._renders_depth:
            return obs[0]
        return obs

    @property
    def config(self) -> MjCambrianEyeConfig:
        """The config for the eye."""
        return self._config

    @property
    def renderer(self) -> MjCambrianRenderer:
        """The renderer for the eye."""
        return self._renderer

    @property
    def name(self) -> str:
        """The name of the eye."""
        return self._name

    @property
    def observation_space(self) -> spaces.Box:
        """Constructs the observation space for the eye. The observation space is a
        `spaces.Box` with the shape of the resolution of the eye."""

        shape = (*self._config.resolution, 3)
        return spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)

    @property
    def num_pixels(self) -> int:
        """The number of pixels in the image."""
        return np.prod(self._config.resolution)

    @property
    def prev_obs(self) -> np.ndarray:
        """The last observation returned by `self.render()`."""
        return self._prev_obs

    @property
    def pos(self) -> np.ndarray:
        return self._data.cam_xpos[self._fixedcamid].copy()

    @property
    def mat(self) -> np.ndarray:
        return self._data.cam_xmat[self._fixedcamid].reshape(3, 3).copy()
