from typing import Dict, Any, Tuple, Optional

import numpy as np
import mujoco as mj
from gymnasium import spaces

from cambrian.envs import MjCambrianEnvConfig, MjCambrianEnv
from cambrian.utils import get_body_id
from cambrian.utils.config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.cambrian_xml import MjCambrianXML


@config_wrapper
class MjCambrianObjectConfig(MjCambrianBaseConfig):
    """Defines a config for an object in the environment.

    Attributes:
        xml (MjCambrianXML): The xml for the object.

        # TODO: remove in favor of keyword arg in reward_fns
        terminate_if_close (bool): Whether to terminate the episode if the animal is
            close to the object. Termination indicates success.
        truncate_if_close (bool): Whether to truncate the episode if the animal is
            close to the object. Truncation indicates failure.
        reward_if_close (float): The reward to give the animal if it is close to the
            object.
        distance_threshold (float): The distance to the object at which the
            animal is assumed to be close to the object.

        use_as_obs (bool): Whether to use the object as an observation or not.
    """

    xml: MjCambrianXML

    terminate_if_close: bool
    truncate_if_close: bool
    reward_if_close: float
    distance_threshold: float

    use_as_obs: bool

    pos: Tuple[float, float, float]


@config_wrapper
class MjCambrianObjectEnvConfig(MjCambrianEnvConfig):
    """Defines a config for the cambrian environment with objects.

    Attributes:
        objects (Dict[str, MjCambrianObjectConfig]): The objects in the environment.
    """

    objects: Dict[str, MjCambrianObjectConfig]


class MjCambrianObjectEnv(MjCambrianEnv):
    """This is a subclass of `MjCambrianEnv` that adds support for goals."""

    def __init__(self, config: MjCambrianObjectEnvConfig, **kwargs):
        self._config = config

        # Have to initialize the objects first since generate_xml is called from the
        # MjCambrianEnv constructor
        self._objects: Dict[str, MjCambrianObject] = {}
        self._create_objects()

        super().__init__(config, **kwargs)

    def _create_objects(self):
        """Creates the objects in the environment."""
        for name, obj_config in self._config.objects.items():
            self._objects[name] = MjCambrianObject(obj_config, name)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = super().generate_xml()

        # TODO: Add targets
        for obj in self._objects.values():
            xml += obj.generate_xml()

        return xml

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[Any, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Reset the environment.

        Will reset all underlying components (the maze, the animals, etc.). The
        simulation will then be stepped once to ensure that the observations are
        up-to-date.

        Returns:
            Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]: The observations for each
                animal and the info dict for each animal.
        """
        # Reset each object
        # Update before super since it calls _update_obs and _update_info
        for obj in self._objects.values():
            obj.reset(self.model)

        return super().reset(seed=seed, options=options)

    def _update_obs(self, obs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Updates the observations for the environment."""
        obs = super()._update_obs(obs)

        # Update the object observations
        for name, obj in self._objects.items():
            if obj.config.use_as_obs:
                obs[name] = obj.pos

        return obs

    def _update_info(
        self, info: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Updates the info for the environment."""
        info = super()._update_info(info)

        # Update the object info
        info["objects"] = {}
        for name, obj in self._objects.items():
            info["objects"][name] = obj.pos

        return info

    def _compute_reward(
        self,
        terminated: Dict[str, bool],
        truncated: Dict[str, bool],
        info: Dict[str, bool],
    ) -> Dict[str, float]:
        """Computes the reward for the environment.

        Args:
            terminated (Dict[str, bool]): Whether each animal has terminated.
                Termination indicates success (agent has reached the goal).
            truncated (Dict[str, bool]): Whether each animal has truncated.
                Truncation indicates failure (agent has hit the wall or something).
            info (Dict[str, bool]): The info dict for each animal.
        """
        rewards = super()._compute_reward(terminated, truncated, info)

        for name, _ in self.animals.items():
            # Early exits
            if terminated[name] or truncated[name]:
                continue

        return rewards

    @property
    def observation_spaces(self) -> spaces.Space:
        """Creates the observation spaces. Identical to `MjCambrianEnv` but with the
        addition of the object observations, if desired."""

        observation_spaces: spaces.Dict = super().observation_spaces

        # Add the object observations
        for animal_name, animal in self.animals.items():
            if not animal.config.trainable:
                continue

            observation_space: spaces.Dict = observation_spaces.spaces[animal_name]

            for name, obj in self._objects.items():
                if obj.config.use_as_obs:
                    observation_space.spaces[name] = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                    )

        return observation_spaces

    @property
    def objects(self) -> Dict[str, "MjCambrianObject"]:
        return self._objects


class MjCambrianObject:
    def __init__(self, config: MjCambrianObjectConfig, name: str):
        self._config = config
        self._name = name

        self._pos = np.array(self._config.pos)

    def generate_xml(self) -> MjCambrianXML:
        return MjCambrianXML.from_string(self._config.xml)

    def reset(self, model: mj.MjModel) -> np.ndarray:
        """Resets the object in the model. Will update it's pos."""
        body_id = get_body_id(model, f"{self._name}_body")
        assert body_id != -1, f"Body {self._name}_body not found in model"

        model.body_pos[body_id] = self._pos

        return model.body_pos[body_id]

    def is_close(self, pos: np.ndarray) -> bool:
        """Helper function to check if the object is close to a position."""
        return np.linalg.norm(self._pos - pos) < self._config.distance_threshold

    @property
    def pos(self) -> np.ndarray:
        return self._pos

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> MjCambrianObjectConfig:
        return self._config
