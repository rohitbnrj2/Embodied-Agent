from math import prod
from typing import Dict, Any, List, Optional, Deque
from enum import Flag, auto
from functools import reduce
from collections import deque

import numpy as np
import mujoco as mj
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from cambrian.evolution_envs.three_d.mujoco.cambrian_xml import MjCambrianXML
from cambrian.evolution_envs.three_d.mujoco.eye import (
    MjCambrianEye,
    MjCambrianEyeConfig,
)
from cambrian.evolution_envs.three_d.mujoco.config import MjCambrianAnimalConfig
from cambrian.evolution_envs.three_d.mujoco.utils import (
    get_include_path,
    get_body_id,
    get_body_name,
    get_geom_id,
    get_joint_id,
    get_geom_name,
    MjCambrianJoint,
    MjCambrianActuator,
    MjCambrianGeometry,
)


class MjCambrianAnimal:
    """The animal class is defined as a physics object with eyes.

    This object serves as an agent in a multi-agent mujoco environment. Therefore,
    it must have a uniquely identifiable name.

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

    Keyword Args:
        verbose (int): The verbosity level. Defaults to 0.
    """

    def __init__(self, config: MjCambrianAnimalConfig, *, verbose: int = 0):
        self.config = self._check_config(config)
        self.verbose = verbose

        self._eyes: Dict[str, MjCambrianEye] = {}
        self._intensity_sensor: MjCambrianEye = None
        self._responsible_for_intensity_sensor: bool = (
            not self.config.use_intensity_obs
            and not self.config.disable_intensity_sensor
        )

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._initialize()

        self._eye_obs: Dict[str, Deque[np.ndarray]] = None
        self._init_pos: np.ndarray = None

    def _check_config(self, config: MjCambrianAnimalConfig) -> MjCambrianAnimalConfig:
        """Run some checks/asserts on the config to make sure everything's there. Also,
        we'll update the model path to make sure it's either absolute/relative to
        the execution path or relative to this file."""

        assert config.model_config.body_name is not None, "No body name specified."
        assert config.model_config.joint_name is not None, "No joint name specified."
        assert config.model_config.geom_name is not None, "No geom name specified."

        return config

    def _initialize(self):
        """Initialize the animal.

        This method does the following:
            - load the base xml to MjModel
            - parse the geometry
            - place eyes at the appropriate locations
        """
        model = mj.MjModel.from_xml_string(self.config.model_config.xml)

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
        geom_id = get_geom_id(model, self.config.model_config.geom_name)
        assert (
            geom_id != -1
        ), f"Could not find geom {self.config.model_config.geom_name}."
        geom_rbound = model.geom_rbound[geom_id]
        geom_pos = model.geom_pos[geom_id]
        self._geom = MjCambrianGeometry(geom_id, geom_rbound, geom_pos)

    def _parse_actuators(self, model: mj.MjModel):
        """Parse the current model/xml for the actuators.

        We have to do this twice: once on the initial model load to get the ctrl limits
        on the actuators. And then later to acquire the actual ids/adrs.
        """

        # Root body for the animal
        body_name = self.config.model_config.body_name
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

        for i, eye_config in enumerate(self.config.eye_configs.values()):
            name = f"{self.name}_eye_{i}"
            self._eyes[name] = self._create_eye(eye_config, name)

        # Add a forward facing eye intensity sensor
        if not self.config.disable_intensity_sensor:
            intensity_sensor_config = self.config.intensity_sensor_config
            intensity_sensor_config.coord = [
                float(np.mean(self.config.model_config.eyes_lat_range)),
                float(np.mean(self.config.model_config.eyes_lon_range)),
            ]
            self._intensity_sensor = self._create_eye(
                intensity_sensor_config,
                f"{self.name}_intensity_sensor",
            )
            if self.config.use_intensity_obs:
                self._eyes[self._intensity_sensor.name] = self._intensity_sensor

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

        config.name = name
        config.pos = (
            pos_rot.apply([-self._geom.rbound, 0, 0]) + self._geom.pos
        ).tolist()
        config.quat = rot_rot.as_quat().tolist()
        return MjCambrianEye(config)

    def generate_xml(self, idx: int) -> MjCambrianXML:
        """Generates the xml for the animal. Will generate the xml from the model file
        and then add eyes to it.
        """
        self.xml = MjCambrianXML.from_string(self.config.model_config.xml)

        # Set each geom in this animal to be a certain group for rendering utils
        # The group number is the index the animal was created + 2
        # + 2 because the default group used in mujoco is 0 and our animal indexes start
        # at 0 and we'll put our scene stuff on group 1
        for geom in self.xml.findall(
            f".//*[@name='{self.config.model_config.body_name}']//geom"
        ):
            geom.set("group", str(idx + 2))

        # Add eyes
        for eye in self.eyes.values():
            self.xml += eye.generate_xml(self.xml, self.config.model_config.body_name)

        # Add the intensity sensor only if it's not included in self.eyes
        if self._responsible_for_intensity_sensor:
            body_name = self.config.model_config.body_name
            self.xml += self._intensity_sensor.generate_xml(self.xml, body_name)

        return self.xml

    def reset(
        self, model: mj.MjModel, data: mj.MjData, init_qpos: np.ndarray
    ) -> Dict[str, Any]:
        """Sets up the animal in the environment. Uses the model/data to update
        positions during the simulation.
        """
        self._model = model
        self._data = data

        # Parse actuators
        self._parse_actuators(model)

        # Accumulate the qpos/qvel/act adrs
        self._reset_adrs(model)

        # Update the animal's position using the freejoint
        self.pos = init_qpos
        # step here so that the observations are updated
        mj.mj_forward(model, data)
        self.init_pos = self.pos.copy()

        self._eye_obs: Dict[str, Deque[np.ndarray]] = {}
        for name, eye in self.eyes.items():
            reset_obs = eye.reset(model, data)

            # The initial obs is a list of black images and the first obs returned
            # by reset
            num_obs = self.config.n_temporal_obs
            init_eye_obs = deque([np.zeros(reset_obs.shape)] * num_obs, maxlen=num_obs)
            init_eye_obs.append(reset_obs)

            self._eye_obs[name] = init_eye_obs

        if self._responsible_for_intensity_sensor:
            self._intensity_sensor.reset(model, data)

        return self._get_obs(np.zeros(self._numctrl))

    def _reset_adrs(self, model: mj.MjModel):
        """Resets the adrs for the animal. This is used when the model is reloaded."""

        # Root body for the animal
        body_name = self.config.model_config.body_name
        self._body_id = get_body_id(model, body_name)
        assert self._body_id != -1, f"Could not find body with name {body_name}."

        # This joint is used for positioning the animal in the environment
        joint_name = self.config.model_config.joint_name
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

    def apply_action(self, action: List[float]):
        """Applies the action to the animal."""
        self._data.ctrl[self._actadrs] = action

    def step(self, action: List[float]) -> Dict[str, Any]:
        """Steps the eyes, updates the ctrl inputs, and returns the observation.

        NOTE: the action isn't actually applied here, it's simply passed to be stored
        in the observation, if needed. The action should be applied explicitly with
        `apply_action`.
        """

        for name, eye in self.eyes.items():
            self._eye_obs[name].append(eye.step())

        if self._responsible_for_intensity_sensor:
            self._intensity_sensor.step()

        return self._get_obs(action)

    def _get_obs(self, action: Optional[List[float]] = None) -> Dict[str, Any]:
        """Creates the entire obs dict."""
        obs: Dict[str, Any] = {}

        for name in self.eyes.keys():
            obs[name] = np.array(self._eye_obs[name])

        if self.config.use_qpos_obs:
            qpos = self._data.qpos[self._qposadrs]
            obs["qpos"] = qpos.flat.copy()

        if self.config.use_qvel_obs:
            qvel = self._data.qvel[self._qveladrs]
            obs["qvel"] = qvel.flat.copy()

        if self.config.use_action_obs:
            assert action is not None, "Action expected."
            obs["action"] = action

        return obs

    def create_composite_image(self) -> np.ndarray | None:
        """Creates a composite image from the eyes. If there are no eyes, then this
        returns None.

        Will appear as a compound eye. For example, if we have a 3x3 grid of eyes:
            TL T TR
            ML M MR
            BL B BR
        """
        from cambrian.evolution_envs.three_d.mujoco.renderer import (
            resize_with_aspect_fill,
        )

        max_res = (
            max([eye.config.resolution[0] for eye in self.eyes.values()]),
            max([eye.config.resolution[1] for eye in self.eyes.values()]),
        )

        # TODO: sort based on lat/lon
        num_horizontal = np.ceil(np.sqrt(len(self.eyes))).astype(int)
        num_vertical = np.ceil(len(self.eyes) / num_horizontal).astype(int)

        images = []
        for i in range(num_vertical):
            images.append([])
            for j in reversed(range(num_horizontal)):
                name = f"{self.name}_eye_{num_vertical * i + j}"

                if name in self._eyes:
                    image = self._eyes[name].last_obs
                else:
                    image = np.zeros((1, 1, 3), dtype=np.uint8)
                images[i].append(resize_with_aspect_fill(image, *max_res))
        images = np.array(images)

        if images.size == 0:
            print(
                f"WARNING: Animal `{self.name}` observations. "
                "Maybe you forgot to call `render`?."
            )
            return None

        return np.vstack([np.hstack(image_row) for image_row in reversed(images)])

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

            body = rootbody = geom = None
            otherbody = otherrootbody = othergeom = None
            if rootbody1 == self._body_id:
                body, rootbody, geom = body1, rootbody1, geom1
                otherbody, otherrootbody, othergeom = body2, rootbody2, geom2
            elif rootbody2 == self._body_id:
                body, rootbody, geom = body2, rootbody2, geom2
                otherbody, otherrootbody, othergeom = body1, rootbody1, geom1
            else:
                # Not a contact with this animal
                continue

            # Verify it's not a ground contact
            groundbody = get_body_id(self._model, "floor")
            if otherrootbody == groundbody:
                continue

            if self.verbose > 1:
                body_name = get_body_name(self._model, body)
                rootbody_name = get_body_name(self._model, rootbody)
                geom_name = get_geom_name(self._model, geom)

                otherbody_name = get_body_name(self._model, otherbody)
                otherrootbody_name = get_body_name(self._model, otherrootbody)
                othergeom_name = get_geom_name(self._model, othergeom)

                print("Detected contact:")
                print(
                    f"\t1 (body :: rootbody :: geom): "
                    f"{body_name} :: {rootbody_name} :: {geom_name}"
                )
                print(
                    f"\t2 (body :: rootbody :: geom): "
                    f"{otherbody_name} :: {otherrootbody_name} :: {othergeom_name}"
                )

            return True

        return False

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

        n_temporal_obs = self.config.n_temporal_obs
        for name, eye in self.eyes.items():
            observation_space[name] = spaces.Box(
                0, 255, shape=(n_temporal_obs, *eye.resolution, 3), dtype=np.uint8
            )

        if self.config.use_qpos_obs:
            observation_space["qpos"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._numqpos,), dtype=np.float32
            )

        if self.config.use_qvel_obs:
            observation_space["qvel"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._numqvel,), dtype=np.float32
            )

        if self.config.use_action_obs:
            actlow = np.array([act.low for act in self._actuators])
            acthigh = np.array([act.high for act in self._actuators])
            observation_space["action"] = spaces.Box(
                low=actlow, high=acthigh, shape=(self._numctrl,), dtype=np.float32
            )

        return spaces.Dict(observation_space)

    @property
    def action_space(self) -> spaces.Space:
        """The action space is simply the controllable actuators of the animal."""
        actlow = np.array([act.low for act in self._actuators])
        acthigh = np.array([act.high for act in self._actuators])
        return spaces.Box(low=actlow, high=acthigh, dtype=np.float32)

    @property
    def eyes(self) -> Dict[str, MjCambrianEye]:
        return self._eyes

    @property
    def num_pixels(self) -> int:
        try:
            return self._num_pixels
        except AttributeError:
            self._num_pixels = 0
            for eye in self._eyes.values():
                self._num_pixels += prod(eye.resolution)
            return self._num_pixels

    @property
    def intensity_sensor(self) -> MjCambrianEye:
        return self._intensity_sensor

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
        return self._data.qpos[self._joint_qposadr : self._joint_qposadr + 2].copy()

    @pos.setter
    def pos(self, value: np.ndarray):
        """See the getter for more info. Sets the freejoint qpos of the animal. If you
        want to set the quat, you have to pass the first 3 elements of the array as pos
        and the remaining 4 as the quat (wxyz).

        Use qpos to set _all_ the positions of the animal.
        """
        self._data.qpos[self._joint_qposadr : self._joint_qposadr + len(value)] = value

    @property
    def init_pos(self) -> np.ndarray:
        """Returns the initial position of the animal."""
        return self._init_pos

    @init_pos.setter
    def init_pos(self, value: np.ndarray):
        """Sets the initial position of the animal."""
        self._init_pos = value

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

    @staticmethod
    def mutate(
        config: MjCambrianAnimalConfig,
        default_eye_config: MjCambrianEyeConfig,
        *,
        mutations: Optional[MutationType] = None,
        verbose: int = 0,
    ) -> MjCambrianAnimalConfig:
        """Mutates the animal config."""
        if verbose > 1:
            print("Mutating animal...")

        if not mutations:
            # The mutation options are all the possible mutations
            mutation_options = [
                MjCambrianAnimal.MutationType[m] for m in config.mutation_options
            ]

            # Randomly select the number of mutations to perform with a skewed dist
            # This will lean towards less total mutations generally
            p = np.exp(-np.arange(len(mutation_options)))
            num_of_mutations = np.random.choice(np.arange(1, len(p) + 1), p=p / p.sum())
            mutations = np.random.choice(mutation_options, num_of_mutations, False)
            mutations = reduce(lambda x, y: x | y, mutations)

            if verbose > 1:
                print(f"Number of mutations: {num_of_mutations}")

        if verbose > 1:
            print(f"Mutations: {mutations}")

        if MjCambrianAnimal.MutationType.REMOVE_EYE in mutations:
            if verbose > 1:
                print("Removing a eye.")

            if len(config.eye_configs) <= 1:
                print("Cannot remove the last eye. Adding one instead.")
                mutations |= MjCambrianAnimal.MutationType.ADD_EYE
            else:
                eye_config = np.random.choice(list(config.eye_configs.values()))
                del config.eye_configs[eye_config.name]

        if MjCambrianAnimal.MutationType.ADD_EYE in mutations:
            if verbose > 1:
                print("Adding a eye.")

            new_eye_config: MjCambrianEyeConfig = default_eye_config.copy()
            new_eye_config.name = f"{config.name}_eye_{len(config.eye_configs)}"
            new_eye_config.coord = [
                np.random.uniform(*config.model_config.eyes_lat_range),
                np.random.uniform(*config.model_config.eyes_lon_range),
            ]
            config.eye_configs[new_eye_config.name] = new_eye_config

        if MjCambrianAnimal.MutationType.EDIT_EYE in mutations:
            if verbose > 1:
                print("Editing an eye.")

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

        if verbose > 2:
            print(f"Mutated animal: \n{config}")

        return config

    @staticmethod
    def crossover(
        parent1: MjCambrianAnimalConfig,
        parent2: MjCambrianAnimalConfig,
        *,
        verbose: int = 0,
    ) -> MjCambrianAnimalConfig:
        raise NotImplementedError("Crossover not implemented.")


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from config import MjCambrianConfig
    from utils import MjCambrianArgumentParser

    parser = MjCambrianArgumentParser(description="Animal Test")

    parser.add_argument("title", type=str, help="Title of the demo.")

    parser.add_argument("--save", action="store_true", help="Save the demo")
    parser.add_argument("--mutate", action="store_true", help="Mutate the animal")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plot", action="store_true", help="Plot the demo")
    action.add_argument("--viewer", action="store_true", help="Launch the viewer")
    action.add_argument("--speed-test", action="store_true", help="Speed test")

    args = parser.parse_args()

    config: MjCambrianConfig = MjCambrianConfig.load(
        args.config, overrides=args.overrides
    )
    animal_config: MjCambrianAnimalConfig = list(
        config.env_config.animal_configs.values()
    )[0]

    if args.mutate:
        mutations = reduce(lambda x, y: x | y, MjCambrianAnimal.MutationType)
        animal_config = MjCambrianAnimal.mutate(
            animal_config,
            config.evo_config.spawning_config.default_eye_config,
            mutations=mutations,
            verbose=2,
        )
    animal = MjCambrianAnimal(animal_config)

    env_xml = MjCambrianXML(get_include_path("models/test.xml"))
    model = mj.MjModel.from_xml_string(str(env_xml + animal.generate_xml(0)))
    data = mj.MjData(model)

    animal.reset(model, data, [-3, 0])

    if args.speed_test:
        print("Starting speed test...")
        num_frames = 200
        t0 = time.time()
        for i in range(num_frames):
            print(i)
            animal.step(np.zeros(animal.action_space.shape))
            mj.mj_step(model, data)
        t1 = time.time()
        print(f"Rendered {num_frames} frames in {t1 - t0} seconds.")
        print(f"Average FPS: {num_frames / (t1 - t0)}")
        exit()

    if args.viewer:
        from renderer import MjCambrianRenderer

        renderer_config = config.env_config.renderer_config
        renderer_config.render_modes = ["human", "rgb_array"]
        renderer_config.camera_config.lookat = [-3, 0, 0.25]
        renderer_config.camera_config.elevation = -20
        renderer_config.camera_config.azimuth = 10
        renderer_config.camera_config.distance = model.stat.extent * 2.5

        renderer = MjCambrianRenderer(renderer_config)
        renderer.reset(model, data)

        renderer.viewer.scene_option.flags[mj.mjtVisFlag.mjVIS_CAMERA] = True
        renderer.viewer.model.vis.scale.camera = 1.0

        i = 0
        while renderer.is_running():
            print(f"Step {i}")
            renderer.render()
            mj.mj_step(model, data)
            i += 1

            if i == 600 and args.save:
                filename = args.title.lower().replace(" ", "_")
                renderer.record = True
                renderer.render()
                print(f"Saving to {filename}...")
                renderer.save(filename, save_types=["png"])
                renderer.record = False
                break

        exit()

    plt.imshow(animal.create_composite_image())
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    if args.plot or args.save:
        plt.title(args.title)
        plt.subplots_adjust(wspace=0, hspace=0)

    if args.save:
        filename = f"{args.title.lower().replace(' ', '_')}.png"
        print(f"Saving to {filename}...")

        # save the figure without the frame
        plt.axis("off")
        plt.savefig(filename, bbox_inches="tight", dpi=300)

    if args.plot and not args.save:
        fig_manager = plt.get_current_fig_manager()
        fig_manager.full_screen_toggle()
        plt.show()
