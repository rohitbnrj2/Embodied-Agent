from typing import Dict, Any, Tuple, List
from prodict import Prodict
from pathlib import Path
import yaml


def read_yaml(filename: Path | str) -> Dict:
    return yaml.safe_load(Path(filename).read_text())


def write_yaml(config: Any, filename: Path | str):
    with open(filename, "w") as f:
        yaml.dump(config, f)


def load_config(config_file: Path | str | Prodict) -> "MjCambrianConfig":
    if isinstance(config_file, (Path, str)):
        config_file = Prodict.from_dict(read_yaml(config_file))
    return config_file


class MjCambrianTrainingConfig(Prodict):
    """Settings for the training process. Used for type hinting.

    Attributes:
        logdir (str): The directory to log training data to.
        exp_name (str): The name of the experiment. Used to name the logging
            subdirectory.
        total_timesteps (int): The total number of timesteps to train for.
        ppo_checkpoint_path (Path | str | None): The path to the ppo checkpoint to
            load. If None, training will start from scratch.
        check_freq (int): The frequency at which to evaluate the model.
        batch_size (int): The batch size to use for training.
        n_steps (int): The number of steps to take per training batch.
        seed (int): The seed to use for training.
        verbose (int): The verbosity level for the training process.
    """

    logdir: Path | str
    exp_name: str
    ppo_checkpoint_path: Path | str | None
    total_timesteps: int
    check_freq: int
    batch_size: int
    n_steps: int
    seed: int
    verbose: int


class MjCambrianMazeConfig(Prodict):
    """Defines a map config. Used for type hinting.

    Attributes:
        name (str): The name of the map. See
            `cambrian.evolution_envs.three_d.mujoco.maps`
        size_scaling (float): The maze scaling for the continuous coordinates in the
            MuJoCo simulation.
        height (float): The height of the walls in the MuJoCo simulation.
        init_goal_pos (Tuple[float, float]): The initial position of the goal in the 
            maze. If unset, will be randomly generated.
        use_target_light_source (bool): Whether to use a target light source or not. If
            False, the default target site will be used (a red sphere). Otherwise, a
            light source will be used. The light source is simply a spot light facing 
            down. If unset (i.e. None), this field will set to the opposite of the
            `use_directional_light` field in `MjCambrianEnvConfig`.
        use_headlight (bool): Whether to use a headlight or not. The headlight in 
            mujoco is basically a first-person light that illuminates the scene directly
            in front of the camera. If False (the default), no headlight is used. This
            should be set to True during visualization/eval/testing.
    """

    name: str
    size_scaling: float
    height: float
    init_goal_pos: Tuple[float, float]
    use_target_light_source: bool
    use_headlight: bool

    def init(self):
        self.name = "U_MAZE"
        self.size_scaling = 1.0
        self.height = 0.5
        self.use_headlight = False


class MjCambrianEnvConfig(Prodict):
    """Defines a config for the cambrian environment. Used for type hinting.

    Attributes:
        num_animals (int): The number of animals to spawn in the env.
        scene_path (str | Path): The path to the scene file. This is the file that
            contains the xml for the environment. The path is either absolute, relative
            to the execution path or relative to the
            cambrian.evolution_envs.three_d.mujoco
        use_directional_light (bool): Whether to use a directional light or not. If 
            True, a directional light will be instantiated in the model that illuminates
            the entire scene. Otherwise, no global illuminating light will be created. 
            Setting to False should be used in the case that the animal is trying to 
            navigate to a light source. Furthermore, if False, the maze xml will be 
            updated such that the target site is a light source instead of a red sphere
            (this behavior can be overwritten using the `target_as_light_source` field 
            in `MjCambrianMazeConfig`).

        model_path (Path | str): The path to the mujoco model file.
        frame_skip (int): The number of mujoco simulation steps per `gym.step()` call.
        render_mode (str): The render mode to use.
        width (int): The width of the rendered image.
        height (int): The height of the rendered image.
        camera_name (str): The name of the camera to use for eval rendering.

        maze_config (MjCambrianMazeConfig): The config for the maze.
    """

    num_animals: int
    scene_path: Path | str
    use_directional_light: bool

    # ============
    # Defined based on `MujocoEnv`

    model_path: Path | str
    frame_skip: int
    render_mode: str
    width: int
    height: int
    camera_name: str

    # ============

    maze_config: MjCambrianMazeConfig

    # ============

    def init(self):
        """Initializes the config with defaults."""
        self.use_directional_light = True
        self.frame_skip = 10
        self.width = 480
        self.height = 480
        self.camera_name = "track"


class MjCambrianEyeConfig(Prodict):
    """Defines the config for an eye. Used for type hinting.

    Attributes:
        name (str): Placeholder for the name of the eye. If set, used directly. If
            unset, the name is set to `{animal.name}_eye_{eye_index}`.
        mode (str): The mode of the camera. Should always be "fixed". See the mujoco
            documentation for more info.
        pos (str): The initial position of the camera. Fmt: "x y z".
        quat (str): The initial rotation of the camera. Fmt: "w x y z".
        resolution (str): The width and height of the rendered image.
            Fmt: "width height". NOTE: only available in 2.3.8.

        filter_size (Tuple[int, int]): The psf filter size. This is convoluted across
            the image, so the actual resolution of the image is plus filter_size / 2
    """

    name: str
    mode: str
    pos: str
    quat: str
    resolution: str

    filter_size: Tuple[int, int]

    def init(self):
        """Set defaults."""
        self.resolution = "1 1"
        self.mode = "fixed"
        self.pos = "0 0 0"
        self.quat = "1 0 0 0"

        self.filter_size = [23, 23]

    def to_xml_kwargs(self) -> Dict:
        return dict(
            name=self.name,
            mode=self.mode,
            pos=self.pos,
            quat=self.quat,
            resolution=self.resolution,
        )


class MjCambrianAnimalConfig(Prodict):
    """Defines the config for an animal. Used for type hinting.

    Attributes:
        type (str): The type of animal. Used to determine which animal subclass to
            create.
        name (str): The name of the animal. Used to uniquely name the animal and its
            eyes. Defaults to `{type}_{i}` where i is the index at which the animal was
            created.
        idx (int): The index of the animal as it was created in the env. Used to 
            uniquely change the animal's xml elements. Placeholder; should be set by
            the env.


        model_path (Path | str): The path to the mujoco model file for the animal.
            Either absolute, relative to execution path or relative to
            cambrian.evolution_envs.three_d.mujoco.animal.py file. This will probably
            be set through a MjCambrianAnimal subclass.
        body_name (str): The name of the body that defines the main body of the animal.
            This will probably be set through a MjCambrianAnimal subclass.
        joint_name (str): The root joint name for the animal. For positioning (see qpos)
            This will probably be set through a MjCambrianAnimal subclass.
        geom_names (List[str]): The names of the geoms that are used for eye
            placement. If a geom is present in the list, it's aab will be parsed from
            the xml and eyes will be placed on the surface of the aabb based on some
            criteria.

        init_pos (Tuple[float, float]): The initial position of the animal. If unset,
            the animal's position at each reset is generated randomly using the
            `maze.generate_reset_pos` method.

        num_eyes_lat (int): The number of eyes to place latitudinally/vertically.
        num_eyes_lon (int): The number of eyes to place longitudinally/horizontally.
        eyes_lat_range (Tuple[float, float]): The x range of the eye. This is used to 
            determine the placement of the eye on the animal. Specified in radians. This
            is the latitudinal/vertical range of the evenly placed eye about the 
            animal's bounding sphere.
        eyes_lon_range (Tuple[float, float]): The y range of the eye. This is used to 
            determine the placement of the eye on the animal. Specified in radians. This
            is the longitudinal/horizontal range of the evenly placed eye about the 
            animal's bounding sphere.
        default_eye_config (EyeConfig): The default eye config to use for the eyes.
    """

    type: str
    name: str
    idx: int

    model_path: Path | str
    body_name: str
    joint_name: str
    geom_names: List[str]

    init_pos: Tuple[float, float]

    num_eyes_lat: int
    num_eyes_lon: int
    eyes_lat_range: Tuple[float, float]
    eyes_lon_range: Tuple[float, float]
    default_eye_config: MjCambrianEyeConfig

    def init(self):
        """Initializes the config with defaults."""
        self.default_eye_config = MjCambrianEyeConfig()


class MjCambrianConfig(Prodict):
    training_config: MjCambrianTrainingConfig
    env_config: MjCambrianEnvConfig
    animal_config: MjCambrianAnimalConfig

    @classmethod
    def load(cls, config: Path | str | Prodict) -> "MjCambrianConfig":
        if isinstance(config, (Path, str)):
            config = cls.from_yaml(config)
        else:
            config = cls(**config)
        return config

    @classmethod
    def from_yaml(cls, filename: Path | str) -> "MjCambrianConfig":
        """Helper method to load a config from a yaml file. This is required so defaults
        are passed correctly to sub-configs"""
        config = cls.from_dict(read_yaml(filename))
        config.training_config = MjCambrianTrainingConfig(**config.training_config)
        config.env_config = MjCambrianEnvConfig(**config.env_config)
        config.animal_config = MjCambrianAnimalConfig(**config.animal_config)

        return config
