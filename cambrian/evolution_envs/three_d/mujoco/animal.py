from typing import Dict, Any, List
import numpy as np
from enum import Enum

import mujoco as mj
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from cambrian_xml import MjCambrianXML
from eye import MjCambrianEye, MjCambrianEyeConfig
from config import MjCambrianAnimalConfig
from utils import (
    get_model_path,
    MjCambrianJoint,
    MjCambrianActuator,
    MjCambrianGeometry,
)


class MjCambrianAnimalType(Enum):
    ANT: str = "ant"
    SWIMMER: str = "swimmer"
    SKYDIO: str = "skydio"
    POINT: str = "point"


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

    ANIMAL_SPECIFIC_CONFIG = dict()

    def __init__(self, config: MjCambrianAnimalConfig):
        config.update(self.ANIMAL_SPECIFIC_CONFIG)
        self.config = config
        self._check_config()

        self._eyes: Dict[str, MjCambrianEye] = {}
        self.intensity_sensor: MjCambrianEye = None

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._initialize()

        self._init_pos: np.ndarray = None

    def _check_config(self):
        """Run some checks/asserts on the config to make sure everything's there. Also,
        we'll update the model path to make sure it's either absolute/relative to
        the execution path or relative to this file."""

        assert self.config.body_name is not None, "No body name specified."
        assert self.config.joint_name is not None, "No joint name specified."
        assert self.config.geom_name is not None, "No geom name specified."
        assert self.config.default_eye_config is not None, "No default eye config."

        self.config.model_path = get_model_path(self.config.model_path)

    def _initialize(self):
        """Initialize the animal.

        This method does the following:
            - load the base xml to MjModel
            - parse the geometry
            - place eyes at the appropriate locations
        """
        model = mj.MjModel.from_xml_path(str(self.config.model_path))

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
        geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, self.config.geom_name)
        assert geom_id != -1, f"Could not find geom with name {self.config.geom_name}."
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
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
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
        """Place the eyes on the animal.

        The current algorithm for eye placement is as follows. We first choose a random
        geometry of the user specified geoms. The eyes are then placed randomly
        on that geometry's bounding sphere (`rbound`). The limits are specified
        by the animal's config. The eye's are restricted along the latitudes
        to be placed within `eye.config.latrange` (probably 60 degrees or something) and
        along the longitudes to be placed within `eye.config.longrange`.

        NOTE:
        - For animal-specific eye placement, you should override this method.
        - `eye.[lat|long]range` are in degrees.

        TODO: Why are the transformations so weird?
        TODO: Have a way to change the placement method, like rectangular or custom
            shape
        """

        eyes_lat_range = np.radians(self.config.eyes_lat_range)
        if self.config.num_eyes_lat == 1:
            latitudes = [np.sum(eyes_lat_range) / 2]
        else:
            latitudes = np.linspace(*eyes_lat_range, self.config.num_eyes_lat)

        eyes_lon_range = np.radians(self.config.eyes_lon_range)
        if self.config.num_eyes_lon == 1:
            longitudes = [np.sum(eyes_lon_range) / 2]
        else:
            longitudes = np.linspace(*eyes_lon_range, self.config.num_eyes_lon)

        # The default rotation to have the camera point forward
        default_rot = R.from_euler("z", np.pi / 2)

        for lat_idx in range(self.config.num_eyes_lat):
            for lon_idx in range(self.config.num_eyes_lon):
                # Create the eye
                eye_config = MjCambrianEyeConfig(**self.config.default_eye_config)
                if eye_config.name is None:
                    i = lat_idx * self.config.num_eyes_lon + lon_idx
                    eye_config.name = f"{self.name}_eye_{i}"
                eye = MjCambrianEye(eye_config)

                # Get the geometry to place the eye on
                latitude = latitudes[lat_idx]
                longitude = longitudes[lon_idx] + np.pi / 2

                # TODO: why is this transformation so weird? Make it into one
                pos_rot = default_rot * R.from_euler("yz", [latitude, longitude])
                rot_rot = R.from_euler("z", latitude)
                rot_rot *= R.from_euler("y", -longitude)
                rot_rot *= default_rot

                # Calc the pos/quat of the eye
                pos = pos_rot.apply([-self._geom.rbound, 0, 0]) + self._geom.pos
                quat = rot_rot.as_quat()

                # Must be space separated strings for the xml
                eye.config.pos = pos
                eye.config.quat = quat

                self._eyes[eye.name] = eye

        # Add a forward facing eye intensity sensor
        # TODO: FIX, ugly
        fov = [120, 10]
        latitude = np.radians(fov[1]) / 2
        longitude = np.pi / 2
        pos_rot = default_rot * R.from_euler("yz", [latitude, longitude])
        rot_rot = R.from_euler("z", latitude)
        rot_rot *= R.from_euler("y", -longitude)
        rot_rot *= default_rot
        pos = pos_rot.apply([-self._geom.rbound, 0, 0]) + self._geom.pos
        quat = rot_rot.as_quat()
        self.intensity_sensor = MjCambrianEye(
            MjCambrianEyeConfig(
                name=f"intensity_sensor_{self.name}",
                pos=pos,
                quat=quat,
<<<<<<< Updated upstream
                resolution=[4, 4], # if other eyes are > 3x3, this needs to be because of bug in sb3 (see is_image_space_channels_first)
=======
                resolution=[4, 4],
>>>>>>> Stashed changes
                fov=[120, 10],
            )
        )
        self._eyes[self.intensity_sensor.name] = self.intensity_sensor

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the animal. Will generate the xml from the model file
        and then add eyes to it.
        """
        idx = self.config.idx

        # Update the names to have idx as a suffix
        self.config.body_name = self.config.body_name.format(uid=idx)
        self.config.joint_name = self.config.joint_name.format(uid=idx)
        self.config.geom_name = self.config.geom_name.format(uid=idx)

        # Create the xml and update the names in the xml to be unique
        # Each name/target/site/etc. should have a fstring-like tag that is evaluated
        # here. It should be implemented using the python built-in .format and use the
        # keyword 'uid'
        # TODO Write better docs here/above
        temp_xml = MjCambrianXML(self.config.model_path)
        self.xml = MjCambrianXML.from_string(str(temp_xml).format(uid=idx))

        # Set each geom in this animal to be a certain group for rendering utils
        # The group number is the index the animal was created + 2
        # + 2 because the default group used in mujoco is 0 and our animal indexes start
        # at 0
        for geom in self.xml.findall(f".//*[@name='{self.config.body_name}']//geom"):
            geom.set("group", str(idx + 1))

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

        # This joint is used for positioning the animal in the environment
        joint_name = self.config.joint_name
        self._joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        assert self._joint_id != -1, f"Could not find joint with name {joint_name}."
        self._joint_qposadr = model.jnt_qposadr[self._joint_id]
        self._joint_dofadr = model.jnt_dofadr[self._joint_id]

        # Parse actuators
        self._parse_actuators(model)

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

        # Update the animal's position using the freejoint
        self.pos = init_qpos
        # step here so that the observations are updated
        mj.mj_forward(model, data)
        self.init_pos = self.pos.copy()

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
        if self.config.use_qpos_obs:
            qpos = self._data.qpos[self._qposadrs]
            obs["qpos"] = qpos.flat.copy()

        if self.config.use_qvel_obs:
            qvel = self._data.qvel[self._qveladrs]
            obs["qvel"] = qvel.flat.copy()

        return obs

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

            rootbody = geom = None
            otherrootbody = othergeom = None
            if rootbody1 == self._body_id:
                rootbody, geom = rootbody1, geom1
                otherrootbody, othergeom = rootbody2, geom2
            elif rootbody2 == self._body_id:
                rootbody, geom = rootbody2, geom2
                otherrootbody, othergeom = rootbody1, geom1
            else:
                # Not a contact with this animal
                continue

            # Verify it's not a self collision
            if rootbody == otherrootbody:
                continue

            # Verify it's not a ground contact
            groundbody = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_BODY, "floor")
            if otherrootbody == groundbody:
                continue

            # body1_name = mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_BODY, body1)
            # body2_name = mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_BODY, body2)
            # print(f"Detected contact between {body1_name} and {body2_name}!")

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

        for name, eye in self.eyes.items():
            observation_space[name] = eye.observation_space

        if self.config.use_qpos_obs:
            observation_space["qpos"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._numqpos,), dtype=np.float32
            )
        if self.config.use_qvel_obs:
            observation_space["qvel"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._numqvel,), dtype=np.float32
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

    def create(config: MjCambrianAnimalConfig) -> "MjCambrianAnimal":
        """Factory method for creating animals. This is used by the environment to
        create animals."""
        type = MjCambrianAnimalType(config.type)

        if type == MjCambrianAnimalType.ANT:
            return MjCambrianAnt(config)
        elif type == MjCambrianAnimalType.SWIMMER:
            return MjCambrianSwimmer(config)
        elif type == MjCambrianAnimalType.SKYDIO:
            return MjCambrianSkydioX2(config)
        elif type == MjCambrianAnimalType.POINT:
            return MjCambrianPoint(config)
        else:
            raise ValueError(f"Animal type {type} not supported.")


class MjCambrianAnt(MjCambrianAnimal):
    """Defines an ant animal.

    See `https://gymnasium.farama.org/environments/mujoco/ant/` for more info.

    This class simply defines some default config attributes that are specific for ants.
    """

    ANIMAL_SPECIFIC_CONFIG = dict(
        model_path="models/ant.xml",
        body_name="torso_{uid}",
        joint_name="root_{uid}",
        geom_name="torso_geom_{uid}",
        eyes_lat_range=[-30, 30],
        eyes_lon_range=[-120, 120],
    )


class MjCambrianSwimmer(MjCambrianAnimal):
    """Defines an swimmer animal.

    See `https://gymnasium.farama.org/environments/mujoco/swimmer/` for more info.

    This class simply defines some default config attributes that are specific for
    swimmers.
    """

    ANIMAL_SPECIFIC_CONFIG = dict(
        model_path="models/swimmer.xml",
        body_name="torso_{uid}",
        joint_name="slider1_{uid}",
        geom_name="frontbody_{uid}",
        eyes_lat_range=[1, 60],
        eyes_lon_range=[-120, 120],
    )


class MjCambrianSkydioX2(MjCambrianAnimal):
    """Defines a skydio-x2 drone "animal".

    See [here](https://github.com/google-deepmind/mujoco_menagerie/tree/main/skydio_x2)
    for more info.
    """

    ANIMAL_SPECIFIC_CONFIG = dict(
        model_path="models/skydio-x2.xml",
        body_name="x2_{uid}",
        joint_name="x2-freejoint_{uid}",
        geom_name="mesh_{uid}",
        eyes_lat_range=[-30, 30],
        eyes_lon_range=[-120, 120],
    )


class MjCambrianPoint(MjCambrianAnimal):
    """Defines a point animal.

    See `https://robotics.farama.org/envs/maze/point_maze/`.
    """

    ANIMAL_SPECIFIC_CONFIG = dict(
        model_path="models/point.xml",
        body_name="particle_{uid}",
        joint_name="ball_x_{uid}",
        geom_name="particle_geom_{uid}",
        eyes_lat_range=[-30, 30],
        eyes_lon_range=[-60, 60],
    )


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from config import MjCambrianConfig

    parser = argparse.ArgumentParser(description="Animal Test")

    parser.add_argument("config_path", type=str, help="Path to the config file.")
    parser.add_argument("title", type=str, help="Title of the demo.")
    parser.add_argument("--plot", action="store_true", help="Plot the demo")
    parser.add_argument("--save", action="store_true", help="Save the demo")

    args = parser.parse_args()

    config = MjCambrianConfig.load(args.config_path)
    config.animal_config.name = "animal"
    config.animal_config.idx = 0
    animal = MjCambrianAnimal.create(config.animal_config)

    env_xml = MjCambrianXML(get_model_path(config.env_config.scene_path))
    model = mj.MjModel.from_xml_string(str(env_xml + animal.generate_xml()))
    data = mj.MjData(model)

    X_NUM, Y_NUM = config.animal_config.num_eyes_lat, config.animal_config.num_eyes_lon
    if args.plot or args.save:
        fig, ax = plt.subplots(Y_NUM, X_NUM, figsize=(Y_NUM, X_NUM))
        if X_NUM == 1 and Y_NUM == 1:
            ax = np.array([[ax]])
        ax = np.flipud(ax)
        assert X_NUM == Y_NUM

    eyes = list(animal.eyes.values())
    obs = animal.reset(model, data, [0, 0, 0.5])
    for i in range(X_NUM):
        for j in range(Y_NUM):
            eye = eyes[i * X_NUM + j]
            image = obs[eye.name]

            if args.plot or args.save:
                ax[i, j].imshow(image.transpose(1, 0, 2))

                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])

    if args.plot or args.save:
        fig.suptitle(args.title)
        plt.subplots_adjust(wspace=0, hspace=0)

    if args.save:
        # save the figure without the frame
        plt.axis("off")
        plt.savefig(f"{args.title}.png", bbox_inches="tight", dpi=300)

    if args.plot:
        fig_manager = plt.get_current_fig_manager()
        fig_manager.full_screen_toggle()
        plt.show()
