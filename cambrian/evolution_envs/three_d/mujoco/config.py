from types import UnionType
from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
    Union,
    get_args,
    get_origin,
    List,
    TypeVar,
    Generic,
)
import yaml
from dataclasses import dataclass, asdict, fields, is_dataclass, replace
from pathlib import Path
from functools import reduce, partial
from copy import deepcopy
import mergedeep
import numpy as np

dataclass = partial(dataclass, kw_only=True, repr=False)


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


def _list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(list, _list_representer)
yaml.add_representer(tuple, _list_representer)


def convert_overrides_to_dict(overrides: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Convert an override list (probably passed through the command line) that
    is in the form (key, dot.separated.nested.value) to a dictionary that `from_dict`
    expects."""

    overrides_dict: Dict[str, Any] = {}
    for k, v in overrides:
        d = overrides_dict

        keys = k.split(".")
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = yaml.safe_load(v)

    return overrides_dict


T = TypeVar("T")


@dataclass
class MjCambrianBaseConfig(Generic[T]):
    """Base config for all configs. This is an abstract class."""

    @classmethod
    def load(cls, config: Path | str | T, *, overrides: Dict[str, Any] = {}) -> T:
        """Load the config from a yaml file or another config. If overrides are passed,
        they are merged into the config.

        Args:
            config (Path | str | MjCambrianBaseConfig): The config to load. If a path or
                string is passed, it is assumed to be a path to a yaml file. If a config
                is passed, it is assumed to be a config of the same type as this config.
            overrides (Dict[str, Any]): The overrides to merge into the config.
        """
        if isinstance(config, (Path, str)):
            config = cls.from_yaml(config, overrides=overrides)
        else:
            # TODO: not working
            # assert isinstance(type(config), cls), (
            #     f"Expected config to be of type {cls.__name__}, but got "
            #     f"{type(config).__name__}"
            # )
            config.merge(overrides)
        return config

    @classmethod
    def from_yaml(cls, path: Path | str, *, overrides: Dict[str, Any] = {}) -> T:
        """Load the config from a yaml file. Will call `from_dict` with the output form
        `yaml.safe_load`.

        Args:
            path (Path | str): The path to the yaml file.
            overrides (Dict[str, Any]): The overrides to merge into the config.
        """
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        data = mergedeep.merge(data, overrides)

        return cls.from_dict(data)

    def write_to_yaml(self, path: Path | str):
        """Write the config to a yaml file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls: T, dict: Dict[str, Any]) -> T:
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
        if dict is None:
            # early exist if dict is None
            # this happens when a recursive call is made and the field is None/Optional
            return None

        field_types = {f.name: _get_underlying_type(f.type) for f in fields(cls)}
        field_values = {}
        for field_name, field_value in dict.items():
            if field_name not in field_types:
                raise AttributeError(f"Unknown field '{field_name}' in {cls.__name__}")
            if is_dataclass(field_types[field_name]) and field_value:
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
        """Merge this config with a passed dictionary or config. This will recursively
        merge nested dictionaries/configs, as well.

        Args:
            other (Dict[str, Any] | MjCambrianBaseConfig): The dictionary to merge with.
                If a dataclass is passed, it will be converted to a dictionary first.
        """
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

    def to_dict(self, *, remove_nones: bool = False) -> Dict[str, Any]:
        """Return the dataclass as a dictionary.

        Keyword Args:
            remove_nones (bool): Whether to remove fields that are None or not. If 
                False, will include all fields, even if they are None.
        """

        def remove_nones_fn(d: Dict[str, Any]) -> Dict[str, Any]:
            if isinstance(d, dict):
                return {k: remove_nones_fn(v) for k, v in d.items() if v is not None}
            return d

        return asdict(self) if not remove_nones else remove_nones_fn(asdict(self))

    def copy(self, **kwargs) -> T:
        """Copy the config such that it is a new instance."""
        return self.update(**kwargs)

    def update(self, **kwargs) -> T:
        """Update the config with the given kwargs. This is a shallow update, meaning it
        will only replace attributes at this class level, not nested."""
        return replace(deepcopy(self), **kwargs)

    def setdefault(self, field: str, default: Any) -> Any:
        """Assign the default value to the field if it is not already set. Like
        `dict.setdefault`."""
        if not hasattr(self, field) or getattr(self, field) is None:
            setattr(self, field, default)
        return getattr(self, field)

    def __contains__(self, key: str) -> bool:
        """Check if the dataclass contains a field with the given name."""
        return key in self.__annotations__

    def __str__(self):
        return yaml.dump(self.to_dict())

    def __post_init__(self):
        """Check that the types of the fields match the types of the dataclass."""

        def _throw_error(name: str, value: Any, expected_type: type):
            raise TypeError(
                f"The field `{name}` was assigned to type `{type(value).__name__}` "
                f"instead of `{expected_type}`"
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
class MjCambrianTrainingConfig(MjCambrianBaseConfig["MjCambrianTrainingConfig"]):
    """Settings for the training process. Used for type hinting.

    Attributes:
        logdir (str): The directory to log training data to.
        exp_name (Optional[str]): The name of the experiment. Used to name the logging
            subdirectory. If unset, will set to the name of the config file.
        checkpoint_path (Optional[Path | str]): The path to the model checkpoint to
            load. If None, training will start from scratch.
        policy_path (Optional[Path | str]): The path to the policy checkpoint to load.
            Should be a `.pt` file that was saved using MjCambrianModel.save_policy.

        total_timesteps (int): The total number of timesteps to train for.
        max_episode_steps (int): The maximum number of steps per episode.
        n_steps (int): The number of steps to take per training batch.

        learning_rate (float): The learning rate to use for training. NOTE: sb3 default
            is 3e-4.
        n_epochs (int): The number of epochs to use for training. NOTE: sb3 default is
            10.
        gae_lambda (float): The lambda value to use for the generalized advantage
            estimation. NOTE: sb3 default is 0.95.
        batch_size (Optional[int)): The batch size to use for training. If None,
            calculated as `n_steps * n_envs // n_epochs`.
        n_envs (int): The number of environments to use for training.

        eval_freq (int): The frequency at which to evaluate the model.
        reward_threshold (float): The reward threshold at which to stop training.
        max_no_improvement_evals (int): The maximum number of evals to take without
            improvement before stopping training.
        min_no_improvement_evals (int): The minimum number of evaluations to perform
            before stopping training if max_no_improvement_steps is reached.

        seed (int): The seed to use for training.
        verbose (int): The verbosity level for the training process.
    """

    logdir: Path | str
    exp_name: Optional[str] = None
    checkpoint_path: Optional[Path | str] = None
    policy_path: Optional[Path | str] = None

    total_timesteps: int
    max_episode_steps: int
    n_steps: int

    learning_rate: float
    n_epochs: int
    gae_lambda: float
    batch_size: Optional[int] = None
    n_envs: int

    eval_freq: int
    reward_threshold: float
    max_no_improvement_evals: int
    min_no_improvement_evals: int

    seed: int
    verbose: int

    def __post_init__(self):
        self.setdefault("batch_size", self.n_steps * self.n_envs // self.n_epochs)

        super().__post_init__()


@dataclass
class MjCambrianMazeConfig(MjCambrianBaseConfig["MjCambrianMazeConfig"]):
    """Defines a map config. Used for type hinting.

    Attributes:
        map (List[List[str]]): The map to use for the maze. It's a 2D array where
            each element is a string and corresponds to a "pixel" in the map. See
            `maze.py` for info on what different strings mean. 
        maze_path (Path | str): The path to the maze xml file. This is the file that
            contains the xml for the maze. The path is either absolute, relative to the
            execution path or relative to the
            cambrian.evolution_envs.three_d.mujoco.maze.py file

        size_scaling (float): The maze scaling for the continuous coordinates in the
            MuJoCo simulation.
        height (float): The height of the walls in the MuJoCo simulation.

        use_target_light_sources (bool): Whether to use a target light sources or not. 
            If False, the colored target sites will be used (e.g. a red sphere). 
            Otherwise, a light source will be used. The light source is simply a spot 
            light facing down. If unset (i.e. None), this field will set to the 
            opposite of the `use_directional_light` field in `MjCambrianEnvConfig`.

        init_goal_pos (Optional[Tuple[float, float]]): The initial position of the 
            goal in the maze. If unset, will be randomly generated.
        eval_goal_pos (Optional[Tuple[float, float]]): The evaluation position of the
            goal in the maze. If unset, will be randomly generated.

        use_adversary (bool): Whether to use an adversarial target or not. If
            True, a second target will be created which is deemed adversarial. Also,
            the target's will be given high frequency textures which correspond to 
            whether a target is adversarial or the true goal. This is done in hopes of 
            having the animal learn to see high frequency input.
        init_adversary_pos (Optional[Tuple[float, float]]): The initial position
            of the adversary target in the maze. If unset, will be randomly generated.
        eval_adversary_pos (Optional[Tuple[float, float]]): The evaluation
            position of the adversary target in the maze. If unset, will be randomly
            generated.
    """

    map: List[List[str]]
    maze_path: Path | str

    size_scaling: float
    height: float

    use_target_light_sources: bool

    init_goal_pos: Optional[Tuple[float, float]] = None
    eval_goal_pos: Optional[Tuple[float, float]] = None

    use_adversary: bool
    init_adversary_pos: Optional[Tuple[float, float]] = None
    eval_adversary_pos: Optional[Tuple[float, float]] = None


@dataclass
class MjCambrianCameraConfig(MjCambrianBaseConfig["MjCambrianCameraConfig"]):
    """Defines a camera config. Used for type hinting. This is a wrapper of
    mj.mjvCamera that is used to configure the camera in the viewer.

    Attributes:
        type (Optional[int]): The type of camera.
        fixedcamid (Optional[int]): The id of the camera to use.
        trackbodyid (Optional[int]): The id of the body to track.

        lookat (Optional[Tuple[float, float, float]]): The point to look at.
        distance (Optional[float]): The distance from the camera to the lookat point.
        azimuth (Optional[float]): The azimuth angle.
        elevation (Optional[float]): The elevation angle.

        typename (Optional[str]): The type of camera as a string. Mutually exclusive
            with type. Converted to mjtCamera with
            getattr(..., f"mjCAMERA_{typename.upper()}")
        fixedcamname (Optional[str]): The name of the camera. Mutually exclusive with
            fixedcamid. Used to determine the fixedcamid using mj.mj_name2id.
        trackbodyname (Optional[str]): The name of the body to track. Mutually exclusive
            with trackbodyid. Used to determine the trackbodyid using mj.mj_name2id.

        distance_factor (Optional[float]): The distance factor. This is used to
            calculate the distance from the camera to the lookat point. If unset, no
            scaling will be applied.
    """

    type: Optional[int] = None
    fixedcamid: Optional[int] = None
    trackbodyid: Optional[int] = None

    lookat: Optional[Tuple[float, float, float]] = None
    distance: Optional[float] = None
    azimuth: Optional[float] = None
    elevation: Optional[float] = None

    typename: Optional[str] = None
    fixedcamname: Optional[str] = None
    trackbodyname: Optional[str] = None

    distance_factor: Optional[float] = None


@dataclass
class MjCambrianRendererConfig(MjCambrianBaseConfig["MjCambrianRendererConfig"]):
    """The config for the renderer. Used for type hinting.

    A renderer corresponds to a single camera. The renderer can then view the scene in
    different ways, like offscreen (rgb_array) or onscreen (human).

    Attributes:
        render_modes (List[str]): The render modes to use for the renderer. See
            `MjCambrianRenderer.metadata["render.modes"]` for options.

        max_geom (Optional[int]): The maximum number of geoms to render.

        width (int): The width of the rendered image. For onscreen renderers, if this
            is set, the window cannot be resized. Must be set for offscreen renderers.
        height (int): The height of the rendered image. For onscreen renderers, if this
            is set, the window cannot be resized. Must be set for offscreen renderers.

        resizable (Optional[bool]): Whether the window is resizable or not. This only
            applies to onscreen renderers.
        fullscreen (Optional[bool]): Whether to render in fullscreen or not. If True,
            the width and height are ignored and the window is rendered in fullscreen.
            This is only valid for onscreen renderers.
        fps (Optional[int]): The fps to render videos at.

        camera_config (Optional[MjCambrianCameraConfig]): The camera config to use for
            the renderer.

        use_shared_context (bool): Whether to use a shared context or not.
            If True, the renderer will share a context with other renderers. This is
            useful for rendering multiple renderers at the same time. If False, the
            renderer will create its own context. This is computationally expensive if
            there are many renderers.
    """

    render_modes: List[str]

    max_geom: Optional[int] = None

    width: Optional[int] = None
    height: Optional[int] = None

    resizable: Optional[bool] = None
    fullscreen: Optional[bool] = None
    fps: Optional[int] = None

    camera_config: Optional[MjCambrianCameraConfig] = None

    use_shared_context: bool = None


@dataclass
class MjCambrianEnvConfig(MjCambrianBaseConfig["MjCambrianEnvConfig"]):
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
        truncate_on_contact (bool): Whether to truncate the episode when the animal
            makes contact with an object or not.
        distance_to_target_threshold (float): The distance to the target at which the
            animal is assumed to be "at the target".
        frame_skip (int): The number of mujoco simulation steps per `gym.step()` call.

        use_renderer (bool): Whether to use the renderer. Should set to False if
            `render` will never be called. Defaults to True. This is useful to reduce
            the amount of vram consumed by non-rendering environments.
        add_overlays (bool): Whether to add overlays or not. 
        overlay_width (Optional[float]): The width of _each_ rendered overlay that's
            placed on the render output. This is primarily for debugging. If unset,
            no overlay will be added. This is a percentage!! It's the percentage of
            the total width of the render output.
        overlay_height (Optional[float]): The height of _each_ rendered overlay that's
            placed on the render output. This is primarily for debugging. If unset,
            no overlay will be added. This is a percentage!! It's the percentage of
            the total height of the render output.
        renderer_config (MjCambrianViewerConfig): The default viewer config to
            use for the mujoco viewer.

        maze_config (MjCambrianMazeConfig): The config for the maze.
    """

    num_animals: int
    scene_path: Path | str
    use_directional_light: bool
    use_headlight: bool

    reward_fn_type: str

    use_goal_obs: bool
    terminate_at_goal: bool
    truncate_on_contact: bool
    distance_to_target_threshold: float

    frame_skip: int

    use_renderer: bool
    add_overlays: bool
    overlay_width: Optional[float] = None
    overlay_height: Optional[float] = None
    renderer_config: MjCambrianRendererConfig

    maze_config: MjCambrianMazeConfig


@dataclass
class MjCambrianEyeConfig(MjCambrianBaseConfig["MjCambrianEyeConfig"]):
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

        renderer_config (MjCambrianRendererConfig): The renderer config to use for the
            underlying renderer. The width and height of the renderer will be set to the
            padded resolution (resolution + filter_size) of the eye.
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

    renderer_config: MjCambrianRendererConfig

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
class MjCambrianAnimalConfig(MjCambrianBaseConfig["MjCambrianAnimalConfig"]):
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
        use_action_obs (bool): Whether to use the action observation or not.

        enforce_2d (bool): Whether to enforce 2d eye placement. If True, all eyes
            will be placed on the same plane and they will all have a vertical 
            resolution of 1.

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

        disable_intensity_sensor (bool): Whether to disable the intensity sensor or not.
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
    use_action_obs: bool

    enforce_2d: bool

    num_eyes_lat: int
    num_eyes_lon: int
    eyes_lat_range: Tuple[float, float]
    eyes_lon_range: Tuple[float, float]
    default_eye_config: MjCambrianEyeConfig

    disable_intensity_sensor: bool
    intensity_sensor_config: MjCambrianEyeConfig


@dataclass
class MjCambrianGenerationConfig(MjCambrianBaseConfig["MjCambrianGenerationConfig"]):
    """Config for a generation. Used for type hinting.

    Attributes:
        rank (int): The rank of the generation. A rank is a unique identifier assigned 
            to each process, where a processes is an individual evo runner running on a
            separate computer. In the context of a cluster, each node that is running
            an evo job is considered one rank, where the rank number is a unique int.
        generation (int): The generation number. This is used to uniquely identify the
            generation.
    """

    rank: int
    generation: int

    def to_path(self) -> Path:
        return Path(f"generation_{self.generation}") / f"rank_{self.rank}"

@dataclass
class MjCambrianPopulationConfig(MjCambrianBaseConfig["MjCambrianPopulationConfig"]):
    """Config for a population. Used for type hinting.

    Attributes:
        size (int): The population size. This represents the number of agents that 
            should be trained at any one time.
        num_top_performers (int): The number of top performers to use in the new agent
            selection. Either in cross over or in mutation, these top performers are
            used to generate new agents.

        replication_type (str): The type of replication to use. See
            `MjCambrianReplicationType` for options.
    """

    size: int
    num_top_performers: int

    replication_type: str


@dataclass
class MjCambrianEvoConfig(MjCambrianBaseConfig["MjCambrianEvoConfig"]):
    """Config for evolutions. Used for type hinting.

    Attributes:
        max_n_envs (int): The maximum number of environments to use for
            parallel training. Will set `n_envs` for each training process to 
            `max_n_envs // population size`.
        total_timesteps (int): The total number of timesteps to train for. 
            Will set `total_timesteps` for each training process to `total_timesteps //
            n_envs`.


        num_generations (int): The number of generations to run for.

        population_config (MjCambrianPopulationConfig): The config for the population.

        generation_config (Optional[MjCambrianGenerationConfig]): The config for the 
            current generation. Will be set by the evolution runner.
        parent_generation_config (Optional[MjCambrianGenerationConfig]): The config 
            for the parent generation. Will be set by the evolution runner. If None, 
            that means that the current generation is the first generation (i.e. no 
            parent).

        environment_variables (Optional[Dict[str, str]]): The environment variables to 
            set for the training process. 
    """

    max_n_envs: int
    total_timesteps: int
    
    num_generations: int

    population_config: MjCambrianPopulationConfig

    generation_config: Optional[MjCambrianGenerationConfig] = None
    parent_generation_config: Optional[MjCambrianGenerationConfig] = None

    environment_variables: Optional[Dict[str, str]] = None


@dataclass
class MjCambrianConfig(MjCambrianBaseConfig["MjCambrianConfig"]):
    includes: Optional[Dict[str, Path | str]] = None

    training_config: MjCambrianTrainingConfig
    env_config: MjCambrianEnvConfig
    animal_config: MjCambrianAnimalConfig
    evo_config: Optional[MjCambrianEvoConfig] = None

    @classmethod
    def from_dict(cls: T, dict: Dict[str, Any]) -> T:
        """Overrides the base class method to handle includes."""
        includes = dict.pop("includes", None)
        if includes is not None:
            for include in reversed(includes.values()):
                with open(include, "r") as f:
                    dict = mergedeep.merge(yaml.safe_load(f), dict)
            dict = cls.from_dict(dict).to_dict()

        return super().from_dict(dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataclass/YAML Tester")

    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action="append",
        type=lambda v: v.split("="),
        help="Override config values. Do <dot separated yaml config>=<value>",
        default=[],
    )

    args = parser.parse_args()

    overrides = convert_overrides_to_dict(args.overrides)
    config = MjCambrianConfig.from_yaml(args.config, overrides=overrides)

    print(config)
    print(config.to_dict())
