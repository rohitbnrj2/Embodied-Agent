from typing import Dict, Any, List, Deque, Tuple, Callable, Self
from enum import Flag, auto
from functools import reduce
from collections import deque
import uuid

import numpy as np
import mujoco as mj
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from cambrian.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.utils import (
    get_body_id,
    get_geom_id,
    get_joint_id,
    MjCambrianJoint,
    MjCambrianActuator,
    MjCambrianGeometry,
    generate_sequence_from_range,
)
from cambrian.utils.cambrian_xml import MjCambrianXML, MjCambrianXMLConfig
from cambrian.utils.base_config import MjCambrianBaseConfig, config_wrapper
from cambrian.utils.logger import get_logger


@config_wrapper
class MjCambrianAnimalConfig(MjCambrianBaseConfig):
    """Defines the config for an animal. Used for type hinting.

    Attributes:
        instance (Callable[[Self, str], "MjCambrianAnimal"]): The class instance for the
            animal. This is used to create the animal.

        xml (MjCambrianXMLConfig): The xml for the animal. This is the xml that will be
            used to create the animal. You should use ${parent:xml} to generate
            named attributes. This will search upwards in the yaml file to find the
            name of the animal.

        body_name (str): The name of the body that defines the main body of the animal.
        joint_name (str): The root joint name for the animal. For positioning (see qpos)
        geom_name (str): The name of the geom that are used for eye placement.

        eyes_lat_range (Tuple[float, float]): The x range of the eye. This is used to
            determine the placement of the eye on the animal. Specified in degrees. This
            is the latitudinal/vertical range of the evenly placed eye about the
            animal's bounding sphere.
        eyes_lon_range (Tuple[float, float]): The y range of the eye. This is used to
            determine the placement of the eye on the animal. Specified in degrees. This
            is the longitudinal/horizontal range of the evenly placed eye about the
            animal's bounding sphere.

        initial_qpos (Dict[int, float]): The initial qpos of the animal. Indices are
            the qpos adr and the values are the initial values. This is used to set the
            initial position of the animal in the environment. The indices are 
            actually calculated as the qpos adr of the joints associated with the 
            animal plus the index specified in the dict.

        use_action_obs (bool): Whether to use the action observation or not. NOTE: If
            the MjCambrianConstantActionWrapper is used, this is not reflected in the
            observation.

        eyes (Dict[str, MjCambrianEyeConfig]): The configs for the eyes.
            The key will be used as the name for the eye.

        mutations_from_parent (Optional[List[str]]): The mutations applied to the child
            (this animal) from the parent. This is unused during mutation; it simply
            is a record of the mutations that were applied to the parent.
    """

    instance: Callable[[Self, str], "MjCambrianAnimal"]

    xml: MjCambrianXMLConfig

    body_name: str
    joint_name: str
    geom_name: str

    eyes_lat_range: Tuple[float, float]
    eyes_lon_range: Tuple[float, float]

    initial_qpos: Dict[int, float]

    use_action_obs: bool

    eyes: Dict[str, MjCambrianEyeConfig]

    mutations_from_parent: List[str]


class MjCambrianAnimal:
    """The animal class is defined as a physics object with eyes.

    This object serves as an agent in a multi-agent mujoco environment. Therefore,
    it must have a uniquely identifiable name.

    In our context, an animal has at any number of eyes and a body which an eye can be
    attached to. This class abstracts away the inner workings of the mujoco model itself
    to the xml generation/loading. It uses existing xml files that only define the
    animals, with which are going to be accumulating into one large xml file that will
    be loaded into mujoco.

    To support specific animal types, you should define subclasses that include animal
    specific configs (i.e. model_path, num_joints).

    Args:
        config (MjCambrianAnimalConfig): The configuration for the animal.
        name (str): The name of the animal. This is used to identify the animal in the
            environment.
    """

    def __init__(self, config: MjCambrianAnimalConfig, name: str):
        self.config = self._check_config(config)
        self._name = name
        self._logger = get_logger()

        self._eyes: Dict[str, MjCambrianEye] = {}

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._initialize()

        # Public attributes
        self.init_pos: np.ndarray = None
        self.last_action: np.ndarray = None

    def _check_config(self, config: MjCambrianAnimalConfig) -> MjCambrianAnimalConfig:
        """Run some checks/asserts on the config to make sure everything's there. Also,
        we'll update the model path to make sure it's either absolute/relative to
        the execution path or relative to this file."""

        assert config.body_name is not None, "No body name specified."
        assert config.joint_name is not None, "No joint name specified."
        assert config.geom_name is not None, "No geom name specified."

        return config

    def _initialize(self):
        """Initialize the animal.

        This method does the following:
            - load the base xml to MjModel
            - parse the geometry
            - place eyes at the appropriate locations
        """
        xml_string = MjCambrianXML.from_config(self.config.xml).to_string()
        model = mj.MjModel.from_xml_string(xml_string)

        self._parse_geometry(model)
        self._parse_actuators(model)

        self._place_eyes()

        del model

    def _parse_geometry(self, model: mj.MjModel):
        """Parse the geometry to get the root body, number of controls, joints, and
        actuators. We're going to do some preprocessing of the model here to get info
        regarding num joints, num controls, etc. This is because we need to know this
        to compute the observation and action spaces, which is needed _before_ actually
        initializing mujoco (but we can't get this information until after mujoco is
        initialized and we don't want to hardcode this for extensibility).

        NOTE:
        - We can't grab the ids/adrs here because they'll be different once we load the
        entire model
        """

        # Num of controls
        assert model.nu > 0, "Model has no controllable actuators."

        # Get number of qpos/qvel/ctrl
        # Just stored for later, like to get the observation space, etc.
        self._numqpos = model.nq
        self._numqvel = model.nv
        self._numctrl = model.nu

        # Create the geometries we will use for eye placement
        geom_id = get_geom_id(model, self.config.geom_name)
        assert geom_id != -1, f"Could not find geom {self.config.geom_name}."
        geom_rbound = model.geom_rbound[geom_id]
        geom_pos = model.geom_pos[geom_id]
        self._geom = MjCambrianGeometry(geom_id, geom_rbound, geom_pos)

    def _parse_actuators(self, model: mj.MjModel):
        """Parse the current model/xml for the actuators.

        We have to do this twice: once on the initial model load to get the ctrl limits
        on the actuators. And then later to acquire the actual ids/adrs.
        """

        # Root body for the animal
        body_name = self.config.body_name
        body_id = get_body_id(model, body_name)
        assert body_id != -1, f"Could not find body with name {body_name}."

        # Mujoco doesn't have a neat way to grab the actuators associated with a
        # specific agent, so we'll try to grab them dynamically by checking the
        # transmission joint ids (the joint adrs associated with that actuator) and
        # seeing if that the corresponding joint is on for this animal's body.
        self._actuators: List[MjCambrianActuator] = []
        for actadr, ((trnid, _), trntype) in enumerate(
            zip(model.actuator_trnid, model.actuator_trntype)
        ):
            if trntype == mj.mjtTrn.mjTRN_JOINT:
                act_bodyid = model.jnt_bodyid[trnid]
            elif trntype == mj.mjtTrn.mjTRN_SITE:
                act_bodyid = model.site_bodyid[trnid]
            else:
                raise NotImplementedError(f'Unsupported trntype "{trntype}".')

            act_rootbodyid = model.body_rootid[act_bodyid]
            if act_rootbodyid == body_id:
                ctrlrange = model.actuator_ctrlrange[actadr]
                self._actuators.append(MjCambrianActuator(actadr, *ctrlrange))

        # Get the joints
        # We use the joints to get the qpos/qvel as observations (joint specific states)
        self._joints: List[MjCambrianJoint] = []
        for jntadr, jnt_bodyid in enumerate(model.jnt_bodyid):
            jnt_rootbodyid = model.body_rootid[jnt_bodyid]
            if jnt_rootbodyid == body_id:
                # This joint is associated with this animal's body
                self._joints.append(MjCambrianJoint.create(model, jntadr))

        assert len(self._joints) > 0, f"Body {body_name} has no joints."
        assert len(self._actuators) > 0, f"Body {body_name} has no actuators."

    def _place_eyes(self):
        """Place the eyes on the animal."""

        for name, eye_config in self.config.eyes.items():
            self._eyes[name] = self._create_eye(eye_config, name)

    def _create_eye(self, config: MjCambrianEyeConfig, name: str) -> MjCambrianEye:
        """Creates an eye with the given config.

        TODO: Rotations are weird. Fix this.
        """
        assert config.coord is not None, "No coord specified."
        lat, lon = np.radians(config.coord)
        lon += np.pi / 2

        default_rot = R.from_euler("z", np.pi / 2)
        pos_rot = default_rot * R.from_euler("yz", [lat, lon])
        rot_rot = R.from_euler("z", lat) * R.from_euler("y", -lon) * default_rot

        geom = self._geom
        config.pos = (pos_rot.apply([-geom.rbound, 0, 0]) + geom.pos).tolist()
        config.quat = rot_rot.as_quat().tolist()
        return MjCambrianEye(config, name)

    def generate_xml(self, idx: int) -> MjCambrianXML:
        """Generates the xml for the animal. Will generate the xml from the model file
        and then add eyes to it.
        """
        xml = MjCambrianXML.from_config(self.config.xml)

        # Set each geom in this animal to be a certain group for rendering utils
        # The group number is the index the animal was created + 2
        # + 2 because the default group used in mujoco is 0 and our animal indexes start
        # at 0 and we'll put our scene stuff on group 1
        for geom in xml.findall(f".//*[@name='{self.config.body_name}']//geom"):
            geom.set("group", str(idx + 2))

        # Add eyes
        for eye in self.eyes.values():
            xml += eye.generate_xml(xml, self.config.body_name)

        return xml

    def apply_action(self, actions: List[float]):
        """Applies the action to the animal. This probably happens before step
        so that the observations reflect the state of the animal after the new action
        is applied.

        It is assumed that the actions are normalized between -1 and 1.
        """
        for action, actuator in zip(actions, self._actuators):
            # Map from -1, 1 to ctrlrange
            action = np.interp(action, [-1, 1], [actuator.low, actuator.high])
            self._data.ctrl[actuator.adr] = action

        # Set the last action class variable
        self.last_action = np.array(actions)

    def reset(self, model: mj.MjModel, data: mj.MjData) -> Dict[str, Any]:
        """Sets up the animal in the environment. Uses the model/data to update
        positions during the simulation.
        """
        self._model = model
        self._data = data

        # Parse actuators
        self._parse_actuators(model)

        # Accumulate the qpos/qvel/act adrs
        self._reset_adrs(model)

        # Set the last_action to the current action
        self.last_action = self._data.ctrl[self._actadrs].copy()

        # Update the animal's qpos
        for idx, val in self.config.initial_qpos.items():
            self.qpos[idx] = val

        # step here so that the observations are updated
        mj.mj_forward(model, data)
        self.init_pos = self.pos.copy()

        obs: Dict[str, Any] = {}
        for name, eye in self.eyes.items():
            obs[name] = eye.reset(model, data)

        return self._update_obs(obs)

    def _reset_adrs(self, model: mj.MjModel):
        """Resets the adrs for the animal. This is used when the model is reloaded."""

        # Root body for the animal
        body_name = self.config.body_name
        self._body_id = get_body_id(model, body_name)
        assert self._body_id != -1, f"Could not find body with name {body_name}."

        # This joint is used for positioning the animal in the environment
        joint_name = self.config.joint_name
        self._joint_id = get_joint_id(model, joint_name)
        assert self._joint_id != -1, f"Could not find joint with name {joint_name}."
        self._joint_qposadr = model.jnt_qposadr[self._joint_id]
        self._joint_dofadr = model.jnt_dofadr[self._joint_id]

        # Accumulate the qpos/qvel/act adrs
        self._qposadrs = []
        for joint in self._joints:
            self._qposadrs.extend(range(joint.qposadr, joint.qposadr + joint.numqpos))
        assert len(self._qposadrs) == self._numqpos

        self._qveladrs = []
        for joint in self._joints:
            self._qveladrs.extend(range(joint.qveladr, joint.qveladr + joint.numqvel))
        assert len(self._qveladrs) == self._numqvel

        self._actadrs = [act.adr for act in self._actuators]
        assert len(self._actadrs) == self._numctrl

    def step(self) -> Dict[str, Any]:
        """Steps the eyes, updates the ctrl inputs, and returns the observation.

        NOTE: the action isn't actually applied here, it's simply passed to be stored
        in the observation, if needed. The action should be applied explicitly with
        `apply_action`.
        """

        obs: Dict[str, Any] = {}
        for name, eye in self.eyes.items():
            obs[name] = eye.step()

        return self._update_obs(obs)

    def _update_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Add additional attributes to the observation."""
        if self.config.use_action_obs:
            obs["action"] = self.last_action

        return obs

    def create_composite_image(self) -> np.ndarray | None:
        """Creates a composite image from the eyes. If there are no eyes, then this
        returns None.

        Will appear as a compound eye. For example, if we have a 3x3 grid of eyes:
            TL T TR
            ML M MR
            BL B BR
        """
        if self.num_eyes == 0:
            return

        from cambrian.renderer import resize_with_aspect_fill

        max_res = (
            max([eye.config.resolution[0] for eye in self.eyes.values()]),
            max([eye.config.resolution[1] for eye in self.eyes.values()]),
        )

        # Sort the eyes based on their lat/lon
        lons = set()
        images = {}
        for eye in self.eyes.values():
            lat, lon = eye.config.coord
            if lat not in images:
                images[lat] = {lon: eye.prev_obs}
            else:
                assert lon not in images[lat]
                images[lat][lon] = eye.prev_obs

        # Construct the composite image
        # Loop through the sorted list of images based on lat/lon
        composite = []
        for lat in sorted(images.keys()):
            row = []
            lons = sorted(images[lat].keys())
            for lon in lons[::-1]:
                row.append(resize_with_aspect_fill(images[lat][lon], *max_res))
            composite.append(np.hstack(row))
        composite = np.vstack(composite)

        if composite.size == 0:
            self._logger.warning(
                f"Animal `{self.name}` observations. "
                "Maybe you forgot to call `render`?."
            )
            return None

        return composite

    @property
    def has_contacts(self) -> bool:
        """Returns whether or not the animal has contacts.

        Walks through all the contacts in the environment and checks if any of them
        involve this animal.
        """
        for contact in self._data.contact:
            geom1 = contact.geom1
            body1 = self._model.geom_bodyid[geom1]
            rootbody1 = self._model.body_rootid[body1]

            geom2 = contact.geom2
            body2 = self._model.geom_bodyid[geom2]
            rootbody2 = self._model.body_rootid[body2]

            if rootbody1 != self._body_id and rootbody2 != self._body_id:
                # Not a contact with this animal
                continue

            return True

        return False

    @property
    def observation_space(self) -> spaces.Space:
        """The observation space is defined on an animal basis. the `env` should combine
        the observation spaces such that it's supported by stable_baselines3/pettingzoo.

        The animal has three observation spaces:
            - {eye.name}: The eyes observations
            - qpos: The joint positions of the animal. The number of joints is extracted
            from the model. It's queried using `qpos`.
            - qvel: The joint velocities of the animal. The number of joints is
            extracted from the model. It's queried using `qvel`.
        """
        observation_space: Dict[Any, spaces.Space] = {}

        for name, eye in self.eyes.items():
            observation_space[name] = eye.observation_space 

        if self.config.use_action_obs:
            observation_space["action"] = spaces.Box(
                low=-1, high=1, shape=(self._numctrl,), dtype=np.float32
            )

        return spaces.Dict(observation_space)

    @property
    def action_space(self) -> spaces.Space:
        """The action space is simply the controllable actuators of the animal."""
        return spaces.Box(low=-1, high=1, shape=(self._numctrl,), dtype=np.float32)

    @property
    def name(self) -> str:
        return self._name

    @property
    def eyes(self) -> Dict[str, MjCambrianEye]:
        return self._eyes

    @property
    def num_eyes(self) -> int:
        return len(self._eyes)

    @property
    def qpos(self) -> np.ndarray:
        """Gets the qpos of the animal. The qpos is the state of the joints defined
        in the animal's xml. This method is used to get the state of the qpos."""
        return self._data.qpos[self._qposadrs].copy()

    @qpos.setter
    def qpos(self, value: np.ndarray[float | None]):
        """Set's the qpos of the animal. The qpos is the state of the joints defined
        in the animal's xml. This method is used to set the state of the qpos. The
        value input is a numpy array where the entries are either values to set
        to the corresponding qpos adr or None. If None, the qpos adr is not
        updated.

        It's allowed for `value` to be less than the total number of joints in the
        animal. If this is the case, only the first `len(value)` joints will be
        updated.
        """
        for qposadr, val in zip(self._qposadrs[: len(value)], value):
            if val is not None:
                self._data.qpos[qposadr] = val

    @property
    def pos(self) -> np.ndarray:
        """Returns the position of the animal in the environment."""
        return self._data.xpos[self._body_id].copy()

    # ==========================

    class MutationType(Flag):
        """Use as bitmask to specify which type of mutation to perform on the animal.

        Example:
        >>> type = MutationType.ADD_EYE
        >>> type = MutationType.REMOVE_EYE | MutationType.EDIT_EYE
        """

        ADD_EYE = auto()
        REMOVE_EYE = auto()
        EDIT_EYE = auto()
        UPDATE_APERTURE = auto()

    @staticmethod
    def mutate(
        config: MjCambrianAnimalConfig,
        mutations: List[str],
        mutation_options: Dict[str, Any],
    ) -> MjCambrianAnimalConfig:
        """Mutates the animal config."""
        logger = get_logger()
        logger.info("Mutating animal...")

        # Store the mutations used to create this animal
        config.mutations_from_parent = mutations

        # Convert the strings to the enum
        mutations = [MjCambrianAnimal.MutationType[m] for m in mutations]
        if len(mutations) == 0:
            return config

        # Randomly select the number of mutations to perform with a skewed dist
        # This will lean towards less total mutations generally
        p = np.exp(-np.arange(len(mutations)))
        num_of_mutations = np.random.choice(np.arange(1, len(p) + 1), p=p / p.sum())
        mutations = np.random.choice(mutations, num_of_mutations, False)
        mutations = reduce(lambda x, y: x | y, mutations)
        logger.info(f"Mutations: {mutations}")

        if MjCambrianAnimal.MutationType.REMOVE_EYE in mutations:
            logger.debug("Removing an eye.")

            if len(config.eye_configs) <= 1:
                logger.info("Cannot remove the last eye. Adding one instead.")
                mutations |= MjCambrianAnimal.MutationType.ADD_EYE
            else:
                # Select an eye at random
                eye_keys = list(config.eye_configs.keys())
                eye_key = np.random.choice(eye_keys)
                del config.eye_configs[eye_key]

        if MjCambrianAnimal.MutationType.ADD_EYE in mutations:
            logger.debug("Adding an eye.")

            # Randomly select an eye to duplicate
            eye_configs = list(config.eye_configs.values())
            new_eye_config: MjCambrianEyeConfig = np.random.choice(eye_configs).copy()
            new_eye_config.name = str(uuid.uuid4())  # make sure the name is unique
            config.eye_configs[new_eye_config.name] = new_eye_config

        if MjCambrianAnimal.MutationType.EDIT_EYE in mutations:
            logger.debug("Editing an eye.")

            # Randomly select an eye to edit
            eye_config: MjCambrianEyeConfig = np.random.choice(
                list(config.eye_configs.values())
            )

            def edit(attrs, low=0.8, high=1.2):
                randn = np.random.uniform(low, high)
                return [int(np.ceil(attr * randn)) for attr in attrs]

            # Each edit (for now) is just taking the current state and multiplying by
            # some random number between 0.8 and 1.2
            eye_config.resolution = edit(eye_config.resolution)
            eye_config.fov = edit(eye_config.fov)

            # If all eyes should be identical, then apply the same edits to all eyes
            if mutation_options.get("all_eyes_identical", False):
                for key in config.eye_configs:
                    config.eye_configs[key].resolution = eye_config.resolution
                    config.eye_configs[key].fov = eye_config.fov
            else:
                config.eye_configs[eye_config.name] = eye_config

        if MjCambrianAnimal.MutationType.UPDATE_APERTURE in mutations:
            logger.debug("Updating aperture.")

            # edit all eyes with the same aperture
            allowed_apertures = np.array(
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32
            )
            aperture = np.random.choice(allowed_apertures)

            # allowed_apertures = np.array([-0.2, 0.1])
            # d_aperture = np.random.choice(allowed_apertures)
            for eye_config in config.eye_configs.values():
                logger.debug(f"aperture_open before: {eye_config.aperture_open}")

                eye_config.aperture_open = aperture
                # config.eye_configs[key].aperture_open += d_aperture
                eye_config.aperture_open = float(
                    np.clip(eye_config.aperture_open, 0.0, 1.0)
                )

                logger.debug(f"aperture_open after: {eye_config.aperture_open}")

        # Update the eye placement, if necessary
        # TODO make official config
        mode = mutation_options.get("eye_placement_mode", "random")
        if mode == "uniform":
            logger.debug("Placing eyes uniformly.")

            # Place the eyes uniformly on the animal. Will redistribute all
            # existing eyes. The number of latitude and longitude bins is
            # proportional to the number of eyes and the ratio of lat and lon
            # ranges.
            num_eyes = len(config.eye_configs)
            lat_range = np.subtract(*config.model_config.eyes_lat_range)
            lon_range = np.subtract(*config.model_config.eyes_lon_range)
            assert (
                lat_range != 0 or lon_range != 0
            ), "Both lat_range and lon_range cannot be 0."

            # Calculate the number of lat and lon bins
            # Handle edge case where lat_range or lon_range is 0
            if lat_range == 0:
                num_lat = 1
                num_lon = num_eyes
            elif lon_range == 0:
                num_lat = num_eyes
                num_lon = 1
            else:
                num_lat = int(np.ceil(np.sqrt(num_eyes)))
                num_lon = int(np.ceil(num_eyes / num_lat))
            lat_bins = generate_sequence_from_range(
                config.model_config.eyes_lat_range, num_lat
            )
            lon_bins = generate_sequence_from_range(
                config.model_config.eyes_lon_range, num_lon
            )

            # Update the coord for each eye
            for i, eye_config in enumerate(config.eye_configs.values()):
                eye_config.coord = [
                    float(lat_bins[i // num_lon]),
                    float(lon_bins[i % num_lon]),
                ]

                # Update the eye name to be the index of the eye in the new list
                eye_config.name = f"{config.name}_eye_{i}"
            config.eye_configs = {eye.name: eye for eye in config.eye_configs.values()}
        elif mode == "random":
            # Assume the eyes are already placed randomly
            logger.debug("Placing eyes randomly (doing nothing).")
        else:
            raise ValueError(f"Invalid eye placement mode: {mode}")

        # Update the number of eyes
        config.num_eyes = len(config.eye_configs)

        return config

    @staticmethod
    def crossover(
        parent1: MjCambrianAnimalConfig,
        parent2: MjCambrianAnimalConfig,
    ) -> MjCambrianAnimalConfig:
        raise NotImplementedError("Crossover not implemented.")
