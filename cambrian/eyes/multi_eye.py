from typing import Any, Callable, Dict, Self, Tuple

import mujoco as mj
import numpy as np
from gymnasium import spaces

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.utils import MjCambrianGeometry, generate_sequence_from_range
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.config import config_wrapper


@config_wrapper
class MjCambrianMultiEyeConfig(MjCambrianEyeConfig):
    """Config for MjCambrianMultiEye.

    Inherits from MjCambrianEyeConfig and adds attributes for procedural eye placement.

    Attributes:
        instance (Callable[[Self, str], MjCambrianEye]): The class instance to use
            when creating the eye. Takes the config and the name of the eye as
            arguments.
        eye_instance (Callable[[Self, str], MjCambrianEye]): The class instance to use
            when creating the single eye instances. Takes the config and the name of the
            eye as arguments.

        lat_range (Optional[Tuple[float, float]]): The x range of the eye. This is
            used to determine the placement of the eye on the agent. Specified in
            degrees. This is the latitudinal/vertical range of the evenly placed eye
            about the agent's bounding sphere.
        lon_range (Optional[Tuple[float, float]]): The y range of the eye. This is
            used to determine the placement of the eye on the agent. Specified in
            degrees. This is the longitudinal/horizontal range of the evenly placed eye
            about the agent's bounding sphere.
        num_eyes (Optional[Tuple[int, int]]): The num of eyes to generate.
            If this is specified, then the eyes will be generated on a spherical
            grid. The first element is the number of eyes to generate latitudinally and
            the second element is the number of eyes to generate longitudinally. The
            eyes will be named sequentially starting from `eye_0`. Each eye will default
            to use the first eye config in the `eyes` attribute. `eyes` must have a
            length of 1 if this is specified. Each eye is named `eye_{lat}_{lon}` where
            `lat` is the latitude index and `lon` is the longitude index.
    """

    instance: Callable[[Self, str], "MjCambrianMultiEye"]
    eye_instance: Callable[[Self, str], "MjCambrianEye"]

    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    num_eyes: Tuple[int, int]


class MjCambrianMultiEye(MjCambrianEye):
    """Defines a multi-eye system that procedurally generates multiple eyes and manages
    them.

    Inherits from MjCambrianEye but manages multiple eyes.

    Args:
        config (MjCambrianMultiEyeConfig): Configuration for the multi-eye system.
        name (str): Base name for the eyes.
    """

    def __init__(
        self, config: MjCambrianMultiEyeConfig, name: str, disable_render: bool = True
    ):
        super().__init__(config, name, disable_render=disable_render)
        self._config: MjCambrianMultiEyeConfig

        self._eyes: Dict[str, MjCambrianEye] = {}

        # Generate eyes procedurally
        self._place_eyes()

    def _place_eyes(self):
        """Place the eyes procedurally based on config."""
        nlat, nlon = self._config.num_eyes
        lat_bins = generate_sequence_from_range(self._config.lat_range, nlat)
        lon_bins = generate_sequence_from_range(self._config.lon_range, nlon)
        for lat_idx, lat in enumerate(lat_bins):
            for lon_idx, lon in enumerate(lon_bins):
                eye_name = f"{self._name}_{lat_idx}_{lon_idx}"
                eye_config = self._config.copy()
                # Update the eye's coord to the current lat, lon
                with eye_config.set_temporarily(is_readonly=False, is_struct=False):
                    eye_config.update("coord", [lat, lon])
                # Create the eye instance
                eye = eye_config.eye_instance(eye_config, eye_name)
                self._eyes[eye_name] = eye

    def generate_xml(
        self, parent_xml: MjCambrianXML, geom: MjCambrianGeometry, parent_body_name: str
    ) -> MjCambrianXML:
        """Generate the XML for all eyes."""
        xml = super().generate_xml(parent_xml, geom, parent_body_name)
        for eye in self._eyes.values():
            eye_xml = eye.generate_xml(parent_xml, geom, parent_body_name)
            xml += eye_xml
        return xml

    def reset(self, model: mj.MjModel, data: mj.MjData):
        """Reset all eyes."""
        obs = {}
        for name, eye in self._eyes.items():
            obs[name] = eye.reset(model, data)

        super().reset(model, data)

        return obs

    def step(self) -> Dict[str, Any]:
        """Step all eyes and collect observations."""
        obs = {}
        for name, eye in self._eyes.items():
            obs[name] = eye.step()
        return obs

    def render(self) -> np.ndarray | None:
        """This is a debug method which renders the eye's as a composite image.

        Will appear as a compound eye. For example, if we have a 3x3 grid of eyes:
            TL T TR
            ML M MR
            BL B BR

        Each eye has a red border around it.
        """
        if self._config.num_eyes == 0:
            return

        from cambrian.renderer.render_utils import generate_composite

        # Calculate the max resolution; min of 10
        max_res = (
            max(max([eye.config.resolution[1] for eye in self.eyes.values()]), 10),
            max(max([eye.config.resolution[0] for eye in self.eyes.values()]), 10),
        )

        # Sort the eyes based on their lat/lon
        images: Dict[float, Dict[float, np.ndarray]] = {}
        for eye in self.eyes.values():
            lat, lon = eye.config.coord
            if lat not in images:
                images[lat] = {}
            assert lon not in images[lat], f"Duplicate eye at {lat}, {lon}."

            # Add the image to the dictionary
            images[lat][lon] = eye.prev_obs[:, :, :3]

        return generate_composite(images, max_res)

    @property
    def observation_space(self) -> spaces.Space:
        """Constructs the observation space for the multi-eye."""
        observation_space = {}
        for name, eye in self._eyes.items():
            observation_space[name] = eye.observation_space
        return spaces.Dict(observation_space)

    @property
    def prev_obs(self) -> Dict[str, np.ndarray]:
        """The last observations from all eyes."""
        obs = {}
        for name, eye in self._eyes.items():
            obs[name] = eye.prev_obs
        return obs

    @property
    def eyes(self) -> Dict[str, MjCambrianEye]:
        """Returns the dictionary of eyes."""
        return self._eyes

    @property
    def name(self) -> str:
        """Returns the base name of the multi-eye system."""
        return self._name
