from types import UnionType
from typing import Dict, Any, Tuple, Optional, Union, get_args, get_origin, List, TypeVar, Generic
import yaml
from dataclasses import dataclass, asdict, fields, is_dataclass, replace
from pathlib import Path
from functools import reduce
from copy import deepcopy
import mergedeep
import numpy as np

dataclass = dataclass(kw_only=True)


def _get_underlying_type(field_type: type) -> type:
    """Get the underlying type of a field. This is useful for dealing with Union
    types."""
    origin = get_origin(field_type)
    if origin is Union:
        args = get_args(field_type)
        return reduce(lambda x, y: x | y if y is not type(None) else x, args)
    else:
        return field_type


def _is_iterable(field_type: type) -> bool:
    is_iterable = False
    try:
        iter(field_type)
        is_iterable = True
    except TypeError:
        try:
            iter(get_origin(field_type))
            is_iterable = True
        except TypeError:
            pass
    return is_iterable

T = TypeVar("T")

@dataclass
class MjCambrianBaseConfig(Generic[T]):
    """Base config for all configs. This is an abstract class."""

    includes: Optional[List[Path | str]] = None

    @classmethod
    def load(cls, config: Path | str | T) -> T:
        if isinstance(config, (Path, str)):
            config = cls.from_yaml(config)
        else:
            assert isinstance(config, cls)
        return config

    @classmethod
    def from_yaml(cls, path: Path | str) -> T:
        """Load the config from a yaml file. Will call `from_dict` with the output form
        `yaml.safe_load`."""
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def write_to_yaml(self, path: Path | str):
        """Write the config to a yaml file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls: T, dict: Dict[str, Any]):
        """
        Recursively converts a dictionary into a dataclass instance.

        This function supports nested data structures, which means that if a dataclass
        contains fields that are themselves dataclasses, it will instantiate those as
        well.

        This will also do some rudimentary type checking and validation. For example,
        if a field is a dataclass, it will check that the dictionary contains the
        correct fields to instantiate that dataclass. It will also check that the types
        of the fields in the dictionary match the types of the fields in the dataclass.

        Args:
            cls (BaseConfig): The config to which the dictionary should be converted to.
            dict (Dict[str, Any]): The dictionary containing the data to be converted
                into a dataclass instance.
        """
        if "includes" in dict:
            for include in dict["includes"]:
                with open(include, "r") as f:
                    dict = mergedeep.merge(yaml.safe_load(f), dict)

        field_types = {
            f.name: _get_underlying_type(f.type) for f in fields(cls)
        }
        field_values = {}
        for field_name, field_value in dict.items():
            if field_name not in field_types:
                raise AttributeError(f"Unknown field '{field_name}' in {cls.__name__}")
            if is_dataclass(field_types[field_name]):
                assert issubclass(field_types[field_name], MjCambrianBaseConfig)
                # Recursively convert nested dictionaries into their corresponding
                # dataclasses
                field_values[field_name] = field_types[field_name].from_dict(
                    field_value
                )
            else:
                field_values[field_name] = field_value
        try:
            return cls(**field_values)
        except TypeError as e:
            raise TypeError(
                f"Error instantiating {cls.__name__} from dictionary: {e}"
            ) from e

    def merge(self, other: Dict[str, Any] | T):
        if issubclass(type(other), MjCambrianBaseConfig):
            other = other.to_dict()

        for field_name, field_value in other.items():
            if field_name not in self.__annotations__:
                raise AttributeError(
                    f"Unknown field '{field_name}' in {self.__class__.__name__}"
                )

            if is_dataclass(self.__annotations__[field_name]):
                assert issubclass(
                    self.__annotations__[field_name], MjCambrianBaseConfig
                )
                # Recursively convert nested dictionaries into their corresponding
                # dataclasses
                if getattr(self, field_name) is None:
                    self_value = self.__annotations__[field_name].from_dict(field_value)
                else:
                    self_value = getattr(self, field_name).merge(field_value)
                setattr(self, field_name, self_value)
            else:
                setattr(self, field_name, field_value)

    def to_dict(self) -> Dict[str, Any]:
        """Return the dataclass as a dictionary."""
        return asdict(self)

    def copy(self) -> T:
        return self.update()

    def update(self, **kwargs) -> T:
        return replace(deepcopy(self), **kwargs)

    def __contains__(self, key: str) -> bool:
        """Check if the dataclass contains a field with the given name."""
        return key in self.__annotations__

    def __post_init__(self):
        """Check that the types of the fields match the types of the dataclass."""

        def _throw_error(name: str, value: Any, expected_type: type):
            raise TypeError(
                f"The field `{name}` was assigned to type `{type(value).__name__}` instead of `{expected_type}`"
            )

        for name, field_type in self.__annotations__.items():
            field_type = _get_underlying_type(field_type)
            value = self.__dict__[name]
            if value is None:
                continue

            origin = get_origin(field_type)
            if _is_iterable(field_type):
                # TODO: Not going to type check iterables
                pass
            elif origin is Union or origin is UnionType:
                args = get_args(field_type)
                if not any(
                    isinstance(value, arg) for arg in args if arg is not type(None)
                ):
                    _throw_error(name, value, field_type)
            elif not isinstance(value, field_type):
                _throw_error(name, value, field_type)


@dataclass
class MjCambrianTrainingConfig(MjCambrianBaseConfig[T]):
    """Settings for the training process. Used for type hinting.

    Attributes:
        logdir (str): The directory to log training data to.
        exp_name (str): The name of the experiment. Used to name the logging
            subdirectory.
        ppo_checkpoint_path (Optional[Path | str]): The path to the ppo checkpoint to
            load. If None, training will start from scratch.
        total_timesteps (int): The total number of timesteps to train for.
        check_freq (int): The frequency at which to evaluate the model.
        batch_size (Optional[int)): The batch size to use for training. If None,
            calculated as `n_steps * n_envs // n_epochs`.
        n_steps (int): The number of steps to take per training batch.
        n_envs (int): The number of environments to use for training.
        max_episode_steps (int): The maximum number of steps per episode.
        seed (int): The seed to use for training.
        verbose (int): The verbosity level for the training process.
    """

    logdir: Path | str
    exp_name: str
    ppo_checkpoint_path: Optional[Path | str] = None
    total_timesteps: int
    check_freq: int
    batch_size: Optional[int] = None
    n_steps: int
    n_envs: int
    max_episode_steps: int
    seed: int
    verbose: int


@dataclass
class MjCambrianMazeConfig(MjCambrianBaseConfig[T]):
    """Defines a map config. Used for type hinting.

    Attributes:
        name (str): The name of the map. See
            `cambrian.evolution_envs.three_d.mujoco.maps`
        size_scaling (float): The maze scaling for the continuous coordinates in the
            MuJoCo simulation.
        height (float): The height of the walls in the MuJoCo simulation.
        init_goal_pos (Optional[Tuple[float, float]]): The initial position of the goal
            in the maze. If unset, will be randomly generated.
        use_target_light_source (bool): Whether to use a target light source or not. If
            False, the default target site will be used (a red sphere). Otherwise, a
            light source will be used. The light source is simply a spot light facing
            down. If unset (i.e. None), this field will set to the opposite of the
            `use_directional_light` field in `MjCambrianEnvConfig`.
    """

    name: str
    size_scaling: float
    height: float
    init_goal_pos: Optional[Tuple[float, float]] = None
    use_target_light_source: bool


@dataclass
class MjCambrianEnvConfig(MjCambrianBaseConfig[T]):
    """Defines a config for the cambrian environment.

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
        use_headlight (bool): Whether to use a headlight or not. The headlight in
            mujoco is basically a first-person light that illuminates the scene directly
            in front of the camera. If False (the default), no headlight is used. This
            should be set to True during visualization/eval/testing.

        reward_fn_type (str): The reward function type to use. See
            `MjCambrianEnv._RewardType` for options.

        use_goal_obs (bool): Whether to use the goal observation or not.
        terminate_at_goal (bool): Whether to terminate the episode when the animal
            reaches the goal or not.
        distance_to_goal_threshold (float): The distance to the goal at which the
            animal is assumed to be "at the goal".

        model_path (Optional[Path | str]): The path to the mujoco model file. Set at
            runtime to the generated xml file.
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
    use_headlight: bool

    reward_fn_type: str

    use_goal_obs: bool
    terminate_at_goal: bool
    distance_to_goal_threshold: float

    # ============
    # Defined based on `MujocoEnv`

    model_path: Optional[Path | str] = None
    frame_skip: int
    render_mode: str
    width: int
    height: int
    camera_name: str

    # ============

    maze_config: MjCambrianMazeConfig

    # ============


@dataclass
class MjCambrianEyeConfig(MjCambrianBaseConfig[T]):
    """Defines the config for an eye. Used for type hinting.

    Attributes:
        name (Optional[str]): Placeholder for the name of the eye. If set, used 
            directly. If unset, the name is set to `{animal.name}_eye_{eye_index}`.
        mode (str): The mode of the camera. Should always be "fixed". See the mujoco
            documentation for more info.
        pos (Optional[Tuple[float, float, float]]): The initial position of the camera. 
            Fmt: xyz
        quat (Optional[Tuple[float, float, float, float]]): The initial rotation of the 
            camera. Fmt: wxyz.
        resolution (Tuple[int, int]): The width and height of the rendered image.
            Fmt: width height.
        fovy (Optinoal[float]): The vertical field of view of the camera.
        focal (Optional[Tuple[float, float]]): The focal length of the camera.
            Fmt: focal_x focal_y.
        focalpixel (Optional[Tuple[int, int]]): The focal length of the camera in 
            pixels.
        sensorsize (Optional[Tuple[float, float]]): The sensor size of the camera.
            Fmt: sensor_x sensor_y.

        fov (Optional[Tuple[float, float]]): Independent of the `fovy` field in the MJCF 
            xml. Used to calculate the sensorsize field. Specified in degrees. Mutually 
            exclusive with `fovy`. If `focal` is unset, it is set to 1, 1. Will override
            `sensorsize`, if set.

        filter_size (Optional[Tuple[int, int]]): The psf filter size. This is 
            convolved across the image, so the actual resolution of the image is plus 
            filter_size / 2
    """

    name: Optional[str] = None
    mode: str
    pos: Optional[Tuple[float, float, float]] = None
    quat: Optional[Tuple[float, float, float, float]] = None
    resolution: Optional[Tuple[int, int]] = None
    fovy: Optional[float] = None
    focal: Optional[Tuple[float, float]] = None
    sensorsize: Optional[Tuple[float, float]] = None

    fov: Optional[Tuple[float, float]] = None

    filter_size: Optional[Tuple[int, int]] = None

    def to_xml_kwargs(self) -> Dict[str, Any]:
        kwargs = dict()

        def set_if_not_none(key: str, val: Any):
            if val is not None:
                if isinstance(val, (list, tuple, np.ndarray)):
                    val = " ".join(map(str, val))
                kwargs[key] = val

        set_if_not_none("name", self.name)
        set_if_not_none("mode", self.mode)
        set_if_not_none("pos", self.pos)
        set_if_not_none("quat", self.quat)
        set_if_not_none("resolution", self.resolution)
        set_if_not_none("fovy", self.fovy)
        set_if_not_none("focal", self.focal)
        set_if_not_none("sensorsize", self.sensorsize)

        return kwargs


@dataclass
class MjCambrianAnimalConfig(MjCambrianBaseConfig[T]):
    """Defines the config for an animal. Used for type hinting.

    Attributes:
        name (Optional[str]): The name of the animal. Used to uniquely name the animal
            and its eyes. Defaults to `{animal_name}_eye_{lat_idx}_{lon_idx}`
        idx (Optional[int]): The index of the animal as it was created in the env. Used 
            to uniquely change the animal's xml elements. Placeholder; should be set by
            the env.

        model_path (Path | str): The path to the mujoco model file for the animal.
            Either absolute, relative to execution path or relative to
            cambrian.evolution_envs.three_d.mujoco.animal.py file. This will probably
            be set through a MjCambrianAnimal subclass.
        body_name (str): The name of the body that defines the main body of the animal.
            This will probably be set through a MjCambrianAnimal subclass.
        joint_name (str): The root joint name for the animal. For positioning (see qpos)
            This will probably be set through a MjCambrianAnimal subclass.
        geom_name (str): The name of the geom that are used for eye placement.

        init_pos (Tuple[float, float]): The initial position of the animal. If unset,
            the animal's position at each reset is generated randomly using the
            `maze.generate_reset_pos` method.

        use_qpos_obs (bool): Whether to use the qpos observation or not.
        use_qvel_obs (bool): Whether to use the qvel observation or not.
        use_intensity_obs (bool): Whether to use the intensity sensor
            observation.

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
        default_eye_config (MjCambrianEyeConfig): The default eye config to use for the
            eyes.
        intensity_sensor_config (MjCambrianEyeConfig): The eye config to use for the
            intensity sensor.
    """

    name: Optional[str] = None
    idx: Optional[int] = None

    model_path: Path | str
    body_name: str
    joint_name: str
    geom_name: str

    init_pos: Optional[Tuple[float, float]] = None

    use_qpos_obs: bool
    use_qvel_obs: bool
    use_intensity_obs: bool

    num_eyes_lat: int
    num_eyes_lon: int
    eyes_lat_range: Tuple[float, float]
    eyes_lon_range: Tuple[float, float]
    default_eye_config: MjCambrianEyeConfig
    intensity_sensor_config: MjCambrianEyeConfig


@dataclass
class MjCambrianConfig(MjCambrianBaseConfig["MjCambrianConfig"]):
    training_config: MjCambrianTrainingConfig
    env_config: MjCambrianEnvConfig
    animal_config: MjCambrianAnimalConfig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataclass/YAML Tester")

    parser.add_argument("base", type=str, help="Path to base config file")

    args = parser.parse_args()

    config = MjCambrianConfig.from_yaml(args.base)

    print(config)
    print(config.to_dict())
