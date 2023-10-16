from typing import Dict, Any, Tuple
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
    """

    name: str
    size_scaling: float
    height: float

    def init(self):
        self.name = "U_MAZE"
        self.size_scaling = 1.0
        self.height = 0.5


class MjCambrianEnvConfig(Prodict):
    """Defines a config for the cambrian environment. Used for type hinting.

    Attributes:
        num_animals (int): The number of animals to spawn in the env.
        model_path (Union[Path, str]): The path to the mujoco model file.
        frame_skip (int): The number of mujoco simulation steps per `gym.step()` call.
        render_mode (str): The render mode to use.
        width (int): The width of the rendered image.
        height (int): The height of the rendered image.
        camera_name (str): The name of the camera to use for rendering. Mutually
        exclusive with `camera_id`.
        camera_id (int): The id of the camera to use for rendering. Mutually exclusive
        with `camera_name`.
        maze_config (MjCambrianMazeConfig): The config for the maze.
    """

    num_animals: int

    # ============
    # Defined based on `MujocoEnv`

    model_path: Path | str
    frame_skip: int
    render_mode: str
    width: int
    height: int
    camera_name: str
    camera_id: int

    # ============

    maze_config: MjCambrianMazeConfig

    # ============

    def init(self):
        """Initializes the config with defaults."""
        self.frame_skip = 10
        self.width = 480
        self.height = 480
        self.camera_name = "track"


class MjCambrianEyeConfig(Prodict):
    """Defines the config for an eye. Used for type hinting.

    Attributes:
        name (str): Placeholder for the name of the eye. See `name_prefix` for more
        info. If set directly, `name_prefix` is ignored.
        name_prefix (str): The name of the eye. Used to uniquely identify the eye. The
        resolved name will be `{name_prefix}_{eye_index}_{animal_name}`. Therefore, the
        `name_prefix` doesn't necessarily need to be unique. The resolved name should
        then be accessible through `name`.
        Fmt: "width height".
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
    name_prefix: str
    mode: str
    pos: str
    quat: str
    resolution: str

    filter_size: Tuple[int, int]

    def init(self):
        """Set defaults."""
        self.name_prefix = "eye"
        self.resolution = "1 1"
        self.mode = "fixed"
        self.pos = "0 0 0"
        self.quat = "1 0 0 0"

        self.filter_size = [23, 23]


class MjCambrianAnimalConfig(Prodict):
    """Defines the config for an animal. Used for type hinting.

    Attributes:
        name (str): The name of the animal. Used to uniquely name the animal and its
        eyes.
        body_name (str): The name of the body that defines the main body of the animal.
        joint_name (str): The root joint name for the animal. For positioning (see qpos)
        model_path (Union[Path, str]): The path to the mujoco model file for the animal.
        Either absolute, relative to execution path or relative to
        cambrian.evolution_envs.three_d.mujoco.animal.py file.
        num_eyes (int): The number of eyes to add to the animal.
        default_eye_config (EyeConfig): The default eye config to use for the eyes.
    """

    name: str
    body_name: str
    joint_name: str

    model_path: Path | str

    num_eyes: int
    default_eye_config: MjCambrianEyeConfig

    def init(self):
        """Initializes the config with defaults."""
        self.body_name = "torso"
        self.joint_name = "root"

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