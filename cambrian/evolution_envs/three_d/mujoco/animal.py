from typing import Dict, Any, List
from pathlib import Path
import numpy as np
from enum import Enum

import mujoco as mj
from gymnasium import spaces

from cambrian_xml import MjCambrianXML
from eye import MjCambrianEye, MjCambrianEyeConfig
from config import MjCambrianAnimalConfig
from utils import safe_index, MjCambrianJoint

class MjCambrianAnimalType(Enum):
    ANT: str = "ant"

class MjCambrianAnimal:
    """The animal class is defined as a physics object with eyes.

    This object serves as an agent in a multi-agent mujoco environment. Therefore,
    it must have a uniquely idenfitiable name.

    In our context, an animal has at least one eye and a body which an eye can be 
    attached to. This class abstracts away the inner workings of the mujoco model itself
    to the xml generation/loading. It uses existing xml files that only define the 
    animals, with which are going to be accumulating into one large xml file that will 
    be loaded into mujoco.

    To support specific animal types, you should define subclasses that include animal
    specific configs (i.e. model_path, num_joints). Furthermore, the main advantage of
    subclassing is for defining the `add_eye` method. This method should position a new
    eye on the animal, which may be unique across animals given geometry. This is 
    important because the XML is required to position the eye and we don't actually know
    the geometry at this time.

    Args:
        config (MjCambrianAnimalConfig): The configuration for the animal.
    """

    def __init__(self, config: MjCambrianAnimalConfig):
        self.config = config
        self._check_config()

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._eyes: Dict[str, MjCambrianEye] = {}

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
            if eye_config.name is None:
                eye_config.name = f"{self.name}_eye_{i}"
            self.eyes[eye_config.name] = MjCambrianEye(eye_config)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the animal. Will generate the xml from the model file
        and then add eyes to it."""
        # Create the xml
        self.xml = MjCambrianXML(self.config.model_path)

        # Add eyes
        for eye in self.eyes.values():
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

        # Root body for the animal
        body_name = self.config.body_name
        self._body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        assert self._body_id != -1, f"Could not find body with name {body_name}."

        # Num of controls
        self._num_ctrl = model.nu
        assert self._num_ctrl > 0, "Model has no controllable dofs."

        # Get the actuator and actuated joint adrs.
        # Mujoco doesn't have a neat way to grab the actuators associated with a 
        # specific agent, so we'll try to grab them dynamically by checking the
        # transmission joint ids (the joint adrs associated with that actuator) and 
        # seeing if that the corresponding joint is on for this animal's body.
        self._joints: List[MjCambrianJoint] = []
        self._actadrs = []
        actuator_trnid = list(model.actuator_trnid[:, 0])
        for jntadr, jnt_bodyid in enumerate(model.jnt_bodyid):
            jnt_rootbodyid = model.body_rootid[jnt_bodyid]
            if jnt_rootbodyid == self._body_id:
                # This joint is associated with this animal's body
                self._joints.append(MjCambrianJoint.create(model, jntadr))

                # Check if this joint is actuated
                if (actadr := safe_index(actuator_trnid, jntadr)) != -1:
                    self._actadrs.append(actadr)
        assert len(self._joints) > 0, f"Body {body_name} has no joints."
        assert len(self._actadrs) > 0, f"Body {body_name} has no actuators."

        self._qposadrs = []
        for joint in self._joints:
            self._qposadrs.extend(range(joint.qposadr, joint.qposadr + joint.numqpos))

        self._qveladrs = []
        for joint in self._joints:
            self._qveladrs.extend(range(joint.qveladr, joint.qveladr + joint.numqvel))

        # This joint is used for positioning the animal in the environment
        joint_name = self.config.joint_name
        self._joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        assert self._joint_id != -1, f"Could not find joint with name {joint_name}."
        self._joint_qposadr = model.jnt_qposadr[self._joint_id]
        self._joint_dofadr = model.jnt_dofadr[self._joint_id]

        self.qpos = init_qpos
        mj.mj_forward(model, data)  # step here so that the observations are updated

        obs: Dict[str, Any] = {}
        for name, eye in self.eyes.items():
            obs[name] = eye.reset(model, data)

        return self._get_obs(obs)

    def step(self, action: List[float]) -> Dict[str, Any]:
        """Simply steps the eyes and returns the observation."""
        obs: Dict[str, Any] = {}
        for name, eye in self.eyes.items():
            obs[name] = eye.step()

        self._data.ctrl[self._actadrs] = action

        return self._get_obs(obs)

    def _get_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Creates the entire obs dict."""
        qpos = self._data.qpos[self._qposadrs]
        qvel = self._data.qvel[self._qveladrs]

        obs["qpos"] = qpos.flat.copy()
        obs["qvel"] = qvel.flat.copy()

        return obs

    @property
    def observation_space(self) -> spaces.Space:
        """The observation space is defined on an animal basis. The `env` should combine
        the observation spaces such that it's supported by stable_baselines3/pettingzoo.

        The animal has three observation spaces:
            - {eye.name}: The eyes observations
            - qpos: The joint positions of the animal. The number of joints is extracted 
            from the model. It's queried using `qpos`.
            - qvel: The joint velocities of the animal. The number of joints is 
            extracted from the model. It's queried using `qvel`.
        """
        observation_space: Dict[spaces.Dict] = {}

        for name, eye in self.eyes.items():
            observation_space[name] = eye.observation_space

        observation_space["qpos"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.config.num_qpos,), dtype=np.float32
        )
        observation_space["qvel"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.config.num_qvel,), dtype=np.float32
        )

        return spaces.Dict(observation_space)

    @property
    def action_space(self) -> spaces.Space:
        """TODO: Probably will be updated after the double reloading change."""
        return spaces.Box(low=-30, high=30, shape=(self.config.num_actuators,), dtype=np.float32)

        bounds = self._model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def eyes(self) -> Dict[str, MjCambrianEye]:
        return self._eyes

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def pos(self) -> np.ndarray:
        """Sets the freejoint position of the animal. The freejoint should be at the
        root of the body of the animal. A free joint in mujoco is capable of being
        explicitly positioned using the `qpos` attribute (which is actually pos and
        quat). This property is for accessing. See the setter.
        
        Use qpos to get _all_ the positions of the animal.
        """
        return np.asarray(self._data.qpos[self._joint_qposadr : self._joint_qposadr + 3])

    @pos.setter
    def pos(self, value: np.ndarray):
        """See the getter for more info. Sets the freejoint qpos of the animal. If you
        want to set the quat, you have to pass the first 3 elements of the array as pos
        and the remaining 4 as the quat (wxyz).
        
        Use qpos to set _all_ the positions of the animal.
        """
        self._data.qpos[self._joint_qposadr : self._joint_qposadr + len(value)] = value

    def create(config: MjCambrianAnimalConfig) -> 'MjCambrianAnimal':
        """Factory method for creating animals. This is used by the environment to
        create animals."""
        type = MjCambrianAnimalType(config.type)

        if type == MjCambrianAnimalType.ANT:
            return MjCambrianAnt(config)
        else:
            raise ValueError(f"Animal type {type} not supported.")

class MjCambrianAnt(MjCambrianAnimal):
    """Defines an ant animal.
    
    See `https://gymnasium.farama.org/environments/mujoco/ant/` for more info.

    This class simply defines some default config attributes that are specific for ants.
    """

    CONFIG = dict(
        body_name="torso",
        joint_name="root",
        num_actuators=8,
        num_qpos=15,
        num_qvel=14,
        model_path="assets/ant.xml",
    )

    def __init__(self, config: MjCambrianAnimalConfig):
        config.update(self.CONFIG)
        super().__init__(config)