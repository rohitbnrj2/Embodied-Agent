from typing import Dict, Any, List
from pathlib import Path
import numpy as np

import mujoco as mj
from gymnasium import spaces

from cambrian_xml import MjCambrianXML
from eye import MjCambrianEye, MjCambrianEyeConfig
from config import MjCambrianAnimalConfig


class MjCambrianAnimal:
    """The animal class is defined as a physics object with eyes.

    In our context, an animal has at least one eye and a body which an eye can be 
    attached to. This class abstracts away the inner workings of the mujoco model itself
    to the xml generation/loading. It uses existing xml files that only define the 
    animals, with which are going to be accumulating into one large xml file that will 
    be loaded into mujoco.

    Args:
        config (MjCambrianAnimalConfig): The configuration for the animal.
    """

    def __init__(self, config: MjCambrianAnimalConfig):
        self.config = config
        self._check_config()

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._eyes: List[MjCambrianEye] = []

        self._create_eyes()

    def _check_config(self):
        """Run some checks/asserts on the config to make sure everything's there. Also,
        we'll update the model path to make sure it's either absolute/relative to
        the execution path or relative to this file."""

        assert self.config.num_eyes > 0, "Must have at least one eye."
        assert self.config.default_eye_config is not None, "No default eye config."

        # Validate the config model path
        model_path = Path(self.config.model_path)
        if model_path.exists():
            pass
        elif (rel_model_path := Path(__file__).parent / model_path).exists():
            self.config.model_path = rel_model_path
        else:
            raise FileNotFoundError(f"Could not find model file {model_path}.")

    def _create_eyes(self):
        """Helper method to create the eyes that are attached to this animal."""

        for i in range(self.config.num_eyes):
            eye_config = MjCambrianEyeConfig(**self.config.default_eye_config)
            eye_config.name = f"{eye_config.name_prefix}_{self.config.name}_{i}"
            self._eyes.append(MjCambrianEye(eye_config))

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the animal. Will generate the xml from the model file
        and then add eyes to it."""
        # Create the xml
        self.xml = MjCambrianXML(self.config.model_path)

        # Add eyes
        for eye in self._eyes:
            self.xml += eye.generate_xml(self.xml, self.config.body_name)

        return self.xml

    def reset(
        self, model: mj.MjModel, data: mj.MjData, init_qpos: np.ndarray
    ) -> Dict[str, Any]:
        """Sets up the animal in the environment. Uses the model/data to update
        positions during the simulation.
        """
        self._model = model
        self._data = data

        body_name = self.config.body_name
        self._body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        assert self._body_id != -1, f"Could not find body with name {body_name}."

        joint_name = self.config.joint_name
        self._joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        assert self._joint_id != -1, f"Could not find joint with name {joint_name}."

        self.qpos = init_qpos
        mj.mj_forward(model, data)  # step here so that the observations are updated

        obs: Dict[str, Any] = {}
        for eye in self._eyes:
            obs[eye.name] = eye.reset(model, data)

        return obs

    def step(self) -> Dict[str, Any]:
        """Simply steps the eyes and returns the observation."""
        obs: Dict[str, Any] = {}
        for eye in self._eyes:
            obs[eye.name] = eye.step()

        return obs

    @property
    def observation_space(self) -> spaces.Dict:
        """The observation space is defined on an animal basis. The `env` should combine
        the observation spaces such that it's supported by stable_baselines3."""
        observation_space: Dict[spaces.Dict] = {}
        for eye in self._eyes:
            observation_space[eye.name] = eye.observation_space
        return spaces.Dict(observation_space)

    @property
    def eyes(self) -> List[MjCambrianEye]:
        return self._eyes

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def qpos(self) -> np.ndarray:
        """Sets the freejoint position of the animal. The freejoint should be at the
        root of the body of the animal. A free joint in mujoco is capable of being
        explicitly positioned using the `qpos` attribute (which is actually pos and
        quat). This property is for accessing. See the setter."""
        return np.asarray(self._data.qpos[self._joint_id : self._joint_id + 3])

    @qpos.setter
    def qpos(self, value: np.ndarray):
        """See the getter for more info. Sets the freejoint qpos of the animal. If you
        want to set the quat, you have to pass the first 3 elements of the array as pos
        and the remaining 4 as the quat (wxyz)."""
        self._data.qpos[self._joint_id : self._joint_id + len(value)] = value
