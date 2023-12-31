from typing import Dict, Any, Tuple, Optional, List, TypeVar, Type
from collections.abc import Iterable
from dataclasses import dataclass, replace, field
from pathlib import Path
from copy import deepcopy
from enum import Flag, auto, Enum

import yaml
from omegaconf import OmegaConf, DictConfig, Node, SCMode

from cambrian.evolution_envs.three_d.mujoco.utils import get_include_path

# ==================== Global Config ====================


def extend(*args: List[DictConfig], _parent_: DictConfig, _node_: Node):
    """This resolver is used to extend a config with another config. To use, set the
    value to ${extend: ${<dotlist>.<key>}}. This will extend the current config with the
    config at the given dotlist. It also accepts multiple configs to extend with, just
    separate with a comma."""
    # Explicitly set extend to None so it doesn't get resolved again when merging
    _parent_[_node_._key()] = None

    for interpolation in args:
        # First move overrides from _parent_ to interpolation since parent is higher
        # priority. Then move interpolation back into parent to get all the defaults.
        # TODO: this makes interpolation happen a lot, slow
        interpolation = OmegaConf.merge(interpolation, _parent_)
        OmegaConf.unsafe_merge(_parent_, interpolation)
    return None


def include(interpolation: DictConfig, type: Optional[str] = "DictConfig"):
    """This resolver takes a filepath and returns the config at that filepath.
    See `get_include_path` for info on how the path is resolved."""
    interpolation: Path = get_include_path(interpolation)
    assert interpolation.exists(), f"File {interpolation} does not exist"
    assert interpolation.is_file(), f"File {interpolation} is not a file"

    if type == "str":
        with open(interpolation) as f:
            return f.read()
    elif type == "DictConfig":
        return OmegaConf.load(interpolation)
    else:
        raise ValueError(f"Unknown type {type}")


def parent(type: Optional[str] = "key", *, _parent_: DictConfig):
    """This resolver is used to access a parent config. To use, set the value to
    ${parent: ${<dotlist>.<key>}}. This will access the parent config at the given
    dotlist."""
    if type == "key":
        return _parent_._key() if _parent_._key() is not None else "PARENT"
    else:
        raise ValueError(f"Unknown type {type}")


# TODO: I would guess hydra does the extend/include part with defaults
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("extend", extend, replace=True)
OmegaConf.register_new_resolver("include", include, replace=True)
OmegaConf.register_new_resolver("parent", parent, replace=True)

list_repr = "tag:yaml.org,2002:seq"
yaml.add_representer(list, lambda d, seq: d.represent_sequence(list_repr, seq, True))
yaml.add_representer(tuple, lambda d, seq: d.represent_sequence(list_repr, seq, True))


# =======================================================


# TODO: move to python3.11 so we can use Self
T = TypeVar("T", bound="MjCambrianBaseConfig")


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianBaseConfig:
    """Base config for all configs. This is an abstract class.

    Attributes:
        extend (Optional[str]): The config to extend this config with.
            This is useful for splitting configs into multiple files. When overriding,
            you can override the extend path directly using the key. Should be
            implemented like this: `extends: ${extend: ${<dotlist>.<key>}}`. You can
            specify multiple includes by passing a list, like
            `extends: ${extend: ${<dotlist>.<key>}, ${<dotlist>.<key>}}`. Lastly, you
            can also include another file and extend it by using the `include`
            resolver. Ex: `extends: ${extend: ${include: <path>}}`.
    """

    extend: Optional[Dict[str, Any]] = field(default=None, init=False)

    @classmethod
    def load(
        cls: T,
        config: Path | str | T,
        *,
        overrides: List[List[str]] = [],
        instantiate: bool = True,
    ) -> T | DictConfig:
        """Load a config. Accepts a path to a yaml file or a config object.

        Args:
            config (Path | str | T): The config to load. Can be a path to a yaml file,
                a yaml string or a config object.

        Keyword Args:
            overrides (List[List[str]]): A list of overrides to apply to the config.
                Each override is a list of strings of the form `key=value`. This is
                passed to OmegaConf.from_dotlist. Defaults to [].
            instantiate (bool): Whether to instantiate the config or not. If True, the
                config will be converted to an object. If False, the config will be
                converted to a dictionary. Defaults to False.

        Returns:
            T | DictConfig: The loaded config. If `instantiate` is True, the config
                will be an object of type T. Otherwise, it will be a
                `OmegaConf.DictConfig`.
        """
        if isinstance(config, (Path, str)):
            filename = Path(config)
            config = OmegaConf.load(filename)
            OmegaConf.register_new_resolver(
                "filename", lambda: filename.stem, replace=True
            )

        overrides = OmegaConf.from_dotlist(overrides)
        OmegaConf.unsafe_merge(config, overrides, overrides)

        if instantiate:
            return cls.instantiate(config)
        else:
            return config

    @classmethod
    def instantiate(cls: Type[T], config: DictConfig) -> T:
        """Convert a config to an object.

        We need to merge config in twice so that extend will work properly
        Not sure why, is kinda of slow. Should be fixed in next OmegaConf version.
        Real issue is we modify the config in place in resolvers, which is
        currently an issue with OmegaConf.resolve, but not an issue with merge.

        TODO: is the doubling really necessary?
        """
        config = OmegaConf.merge(cls, config, config)
        OmegaConf.resolve(config)  # resolves included strings
        return OmegaConf.to_container(config, resolve=False, throw_on_missing=True, structured_config_mode=SCMode.INSTANTIATE)
        return OmegaConf.to_object(config)

    def save(self, path: Path | str):
        """Save the config to a yaml file."""
        OmegaConf.save(self, path)

    def copy(self: T, **kwargs) -> T:
        """Copy the config such that it is a new instance."""
        return deepcopy(self).update(**kwargs)

    def update(self: T, **kwargs) -> T:
        """Update the config with the given kwargs. This is a shallow update, meaning it
        will only replace attributes at this class level, not nested."""
        return replace(self, **kwargs)

    def merge_with_dotlist(self: T, dotlist: List[str]) -> T:
        """Merge the config with the given dotlist. This is a shallow merge, meaning it
        will only replace attributes at this class level, not nested."""
        return self.update(**OmegaConf.from_dotlist(dotlist))

    def setdefault(self: T, key: str, default: Any) -> Any:
        """Assign the default value to the key if it is not already set. Like
        `dict.setdefault`."""
        if not hasattr(self, key) or getattr(self, key) is None:
            setattr(self, key, default)
        return getattr(self, key)

    def __contains__(self: T, key: str) -> bool:
        """Check if the dataclass contains a key with the given name."""
        return key in self.__annotations__

    def __str__(self):
        return OmegaConf.to_yaml(self)


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianGenerationConfig(MjCambrianBaseConfig):
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


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianTrainingConfig(MjCambrianBaseConfig):
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


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianMazeConfig(MjCambrianBaseConfig):
    """Defines a map config. Used for type hinting.

    Attributes:
        name (str): The name of the maze. Used to uniquely name the maze.

        map (List[List[str]]): The map to use for the maze. It's a 2D array where
            each element is a string and corresponds to a "pixel" in the map. See
            `maze.py` for info on what different strings mean.
        xml (str): The xml for the maze. This is the xml that will be used to
            create the maze.

        difficulty (int): The difficulty of the maze. This is used to determine the
            selection probability of the maze when the mode is set to "DIFFICULTY".
            The value should be set between 0 and 100, where 0 is the easiest and 100
            is the hardest.

        size_scaling (float): The maze scaling for the continuous coordinates in the
            MuJoCo simulation.
        height (float): The height of the walls in the MuJoCo simulation.

        use_target_light_sources (bool): Whether to use a target light sources or not.
            If False, the colored target sites will be used (e.g. a red sphere).
            Otherwise, a light source will be used. The light source is simply a spot
            light facing down.

        wall_texture_map (Dict[str, str]): The mapping from texture id to texture name.
            The keyword "default" is required for walls denoted simply as 1 or W. 
            Other walls are specified as 1/W:<texture id>.

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

    name: str

    map: List[List]
    xml: str

    difficulty: int

    size_scaling: float
    height: float

    use_target_light_sources: bool

    wall_texture_map: Dict[str, str]

    init_goal_pos: Optional[Tuple[float, float]] = None
    eval_goal_pos: Optional[Tuple[float, float]] = None

    use_adversary: bool
    init_adversary_pos: Optional[Tuple[float, float]] = None
    eval_adversary_pos: Optional[Tuple[float, float]] = None


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianCameraConfig(MjCambrianBaseConfig):
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

        typename (Optional[str]): The type of camera as a string. Takes presidence over
            type. Converted to mjtCamera with mjCAMERA_{typename.upper()}.
        fixedcamname (Optional[str]): The name of the camera. Takes presidence over
            fixedcamid. Used to determine the fixedcamid using mj.mj_name2id.
        trackbodyname (Optional[str]): The name of the body to track. Takes presidence
            over trackbodyid. Used to determine the trackbodyid using mj.mj_name2id.

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


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianRendererConfig(MjCambrianBaseConfig):
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

        fullscreen (Optional[bool]): Whether to render in fullscreen or not. If True,
            the width and height are ignored and the window is rendered in fullscreen.
            This is only valid for onscreen renderers.

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

    fullscreen: Optional[bool] = None

    camera_config: Optional[MjCambrianCameraConfig] = None

    use_shared_context: bool


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianEyeConfig(MjCambrianBaseConfig):
    """Defines the config for an eye. Used for type hinting.

    Attributes:
        name (Optional[str]): Placeholder for the name of the eye. If set, used
            directly. If unset, the name is set to `{animal.name}_eye_{eye_index}`.

        mode (str): The mode of the camera. Should always be "fixed". See the mujoco
            documentation for more info.
        resolution (Tuple[int, int]): The width and height of the rendered image.
            Fmt: width height.
        fov (Tuple[float, float]): Independent of the `fovy` field in the MJCF
            xml. Used to calculate the sensorsize field. Specified in degrees. Mutually
            exclusive with `fovy`. If `focal` is unset, it is set to 1, 1. Will override
            `sensorsize`, if set.
        filter_size (Tuple[int, int]): The psf filter size. This is
            convolved across the image, so the actual resolution of the image is plus
            filter_size / 2.
        coord (Optional[Tuple[float, float]]): The x and y coordinates of the eye.
            This is used to determine the placement of the eye on the animal.
            Specified in degrees. Mutually exclusive with `pos` and `quat`. This attr
            isn't actually used by eye, but by the animal. The eye has no knowledge
            of the geometry it's trying to be placed on. Fmt: lat lon

        pos (Optional[Tuple[float, float, float]]): The initial position of the camera.
            Fmt: xyz
        quat (Optional[Tuple[float, float, float, float]]): The initial rotation of the
            camera. Fmt: wxyz.
        fovy (Optional[float]): The vertical field of view of the camera.
        focal (Optional[Tuple[float, float]]): The focal length of the camera.
            Fmt: focal_x focal_y.
        sensorsize (Optional[Tuple[float, float]]): The sensor size of the camera.
            Fmt: sensor_x sensor_y.

        renderer_config (MjCambrianRendererConfig): The renderer config to use for the
            underlying renderer. The width and height of the renderer will be set to the
            padded resolution (resolution + filter_size) of the eye.
    """

    name: Optional[str] = None

    mode: str
    resolution: Tuple[int, int]
    fov: Tuple[float, float]
    filter_size: Tuple[int, int]

    pos: Optional[Tuple[float, float, float]] = None
    quat: Optional[Tuple[float, float, float, float]] = None
    fovy: Optional[float] = None
    focal: Optional[Tuple[float, float]] = None
    sensorsize: Optional[Tuple[float, float]] = None

    coord: Optional[Tuple[float, float]] = None

    renderer_config: MjCambrianRendererConfig

    def to_xml_kwargs(self) -> Dict[str, Any]:
        kwargs = dict()

        def set_if_not_none(key: str, val: Any):
            if val is not None:
                if isinstance(val, Iterable) and not isinstance(val, str):
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


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianAnimalModelConfig(MjCambrianBaseConfig):
    """Defines the config for an animal model. Used for type hinting.

    Attributes:
        xml (str): The xml for the animal model. This is the xml that will be used to
            create the animal model. You should use ${..name} to generate named
            attributes.
        body_name (str): The name of the body that defines the main body of the animal.
            This will probably be set through a MjCambrianAnimal subclass.
        joint_name (str): The root joint name for the animal. For positioning (see qpos)
            This will probably be set through a MjCambrianAnimal subclass.
        geom_name (str): The name of the geom that are used for eye placement.

        eyes_lat_range (Tuple[float, float]): The x range of the eye. This is used to
            determine the placement of the eye on the animal. Specified in radians. This
            is the latitudinal/vertical range of the evenly placed eye about the
            animal's bounding sphere.
        eyes_lon_range (Tuple[float, float]): The y range of the eye. This is used to
            determine the placement of the eye on the animal. Specified in radians. This
            is the longitudinal/horizontal range of the evenly placed eye about the
            animal's bounding sphere.
    """

    xml: str
    body_name: str
    joint_name: str
    geom_name: str

    eyes_lat_range: Tuple[float, float]
    eyes_lon_range: Tuple[float, float]


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianAnimalConfig(MjCambrianBaseConfig):
    """Defines the config for an animal. Used for type hinting.

    Attributes:
        name (str): The name of the animal. Used to uniquely name the animal
            and its eyes.

        init_pos (Tuple[float, float]): The initial position of the animal. If unset,
            the animal's position at each reset is generated randomly using the
            `maze.generate_reset_pos` method.

        model_config (MjCambrianAnimalModelConfig): The config for the animal model.

        use_qpos_obs (bool): Whether to use the qpos observation or not.
        use_qvel_obs (bool): Whether to use the qvel observation or not.
        use_intensity_obs (bool): Whether to use the intensity sensor observation.
        use_action_obs (bool): Whether to use the action observation or not.
        n_temporal_obs (int): The number of temporal observations to use.

        mutation_options (List[str]): The mutation options to use for the animal. See
            `MjCambrianAnimal.MutationType` for options.

        eye_configs (Dict[str, MjCambrianEyeConfig]): The configs for the eyes.
            The key will be used as the default name for the eye, unless explicitly
            set in the eye config.

        disable_intensity_sensor (bool): Whether to disable the intensity sensor or not.
        intensity_sensor_config (MjCambrianEyeConfig): The eye config to use for the
            intensity sensor.

        parent_generation_config (Optional[MjCambrianGenerationConfig]): The config for
            the parent generation. Will be set by the evolution runner. If None, that
            means that the current generation is the first generation (i.e. no parent).
    """

    name: str

    model_config: Optional[MjCambrianAnimalModelConfig] = None

    init_pos: Optional[Tuple[float, float]] = None

    use_qpos_obs: bool
    use_qvel_obs: bool
    use_intensity_obs: bool
    use_action_obs: bool
    n_temporal_obs: int

    mutation_options: List[str]

    eye_configs: Dict[str, MjCambrianEyeConfig] = field(default_factory=dict)

    disable_intensity_sensor: bool
    intensity_sensor_config: MjCambrianEyeConfig

    parent_generation_config: Optional[MjCambrianGenerationConfig] = None


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianEnvConfig(MjCambrianBaseConfig):
    """Defines a config for the cambrian environment.

    Attributes:
        xml (str): The xml for the scene. This is the xml that will be used to
            create the environment.
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

        maze_selection_criteria (Dict[str, Any]): The mode to use for choosing
            the maze. The `mode` key is required and must be set to a 
            `MazeSelectionMode`. See `MazeSelectionMode` for other params or 
            `maze.py` for more info.
        maze_config ([Dict[str, MjCambrianMazeConfig]): The configs for the mazes. Each
            maze will be loaded into the scene and the animal will be placed in a maze
            at each reset. The maze will be chosen based on the `maze_choice_mode`
            field.
        eval_maze_configs (Optional[Dict[str, MjCambrianMazeConfig]]): The 
            configs for the evaluation mazes. If unset, the one evaluation maze will 
            be chosen using the maze selection criteria. 

        animal_configs (Dict[str, MjCambrianAnimalConfig]): The configs for the animals.
            The key will be used as the default name for the animal, unless explicitly
            set in the animal config.
    """

    xml: str
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

    class MazeSelectionMode(Enum):
        """The mode to use for choosing the maze. See `maze.py` for more info.

        NOTE: the `mode` key is required for the criteria dict. other keys are passed
        as kwargs to the selection method.

        Ex:
            # Choose a random maze
            maze_selection_criteria: 
                mode: RANDOM

            # Choose a maze based on difficulty
            maze_selection_criteria: 
                mode: DIFFICULTY
                schedule: true

            # From the command line
            -o ...maze_selection_criteria="{mode: DIFFICULTY, schedule: true}"
            # or simply
            -o ...maze_selection_criteria.mode=RANDOM

        Attributes:
            RANDOM (str): Choose a random maze.
            DIFFICULTY (str): Choose a maze based on difficulty. A maze is chosen
                based on the prob exp(-difficulty). By default, easier mazes will
                be selected based on a skewed distribution. If `schedule` is passed
                as a kwarg and set to true, the probability of choosing more difficult
                mazes increases over time with a linear schedule.
            NAMED (str): Choose a maze based on name. `name` must be passed as a kwarg
                to the selection method.
            CYCLE (str): Cycle through the mazes. The mazes are cycled through in
                the order they are defined in the config.

            EVAL (str): Use the mazes in `eval_maze_configs`. The mazes are cycled,
                similar to CYCLE. `eval_maze_configs` cannot be None.
        """

        RANDOM: str = "random"
        DIFFICULTY: str = "difficulty"
        NAMED: str = "named"
        CYCLE: str = "cycle"

        EVAL: str = "eval"

    maze_selection_criteria: Dict[str, Any]
    maze_configs: Dict[str, MjCambrianMazeConfig]
    eval_maze_configs: Optional[Dict[str, MjCambrianMazeConfig]] = None 

    animal_configs: Dict[str, MjCambrianAnimalConfig] = field(default_factory=dict)


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianPopulationConfig(MjCambrianBaseConfig):
    """Config for a population. Used for type hinting.

    Attributes:
        size (int): The population size. This represents the number of agents that
            should be trained at any one time.
        num_top_performers (int): The number of top performers to use in the new agent
            selection. Either in cross over or in mutation, these top performers are
            used to generate new agents.
    """

    size: int
    num_top_performers: int


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianSpawningConfig(MjCambrianBaseConfig):
    """Config for spawning. Used for type hinting.

    Attributes:
        num_mutations (int): The number of mutations to perform on the
            default config to generate the initial population. The actual number of
            mutations is calculated using random.randint(1, init_num_mutations).

        replication_type (str): The type of replication to use. See
            `ReplicationType` for options.
    """

    init_num_mutations: int

    class ReplicationType(Flag):
        """Use as bitmask to specify which type of replication to perform on the animal.

        Example:
        >>> # Only mutation
        >>> type = ReplicationType.MUTATION
        >>> # Both mutation and crossover
        >>> type = ReplicationType.MUTATION | ReplicationType.CROSSOVER
        """

        MUTATION = auto()
        CROSSOVER = auto()

    replication_type: str

    default_animal_config: Optional[MjCambrianAnimalConfig] = None
    default_eye_config: Optional[MjCambrianEyeConfig] = None


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianEvoConfig(MjCambrianBaseConfig):
    """Config for evolutions. Used for type hinting.

    Attributes:
        max_n_envs (int): The maximum number of environments to use for
            parallel training. Will set `n_envs` for each training process to
            `max_n_envs // population size`.

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

    num_generations: int

    population_config: MjCambrianPopulationConfig
    spawning_config: MjCambrianSpawningConfig

    generation_config: Optional[MjCambrianGenerationConfig] = None

    environment_variables: Dict[str, str]


@dataclass(kw_only=True, repr=False, slots=True, eq=False, match_args=False)
class MjCambrianConfig(MjCambrianBaseConfig):
    """The base config for the mujoco cambrian environment. Used for type hinting.

    Attributes:
        training_config (MjCambrianTrainingConfig): The config for the training process.
        env_config (MjCambrianEnvConfig): The config for the environment.
        animal_config (MjCambrianAnimalConfig): The config for the animal.
        evo_config (Optional[MjCambrianEvoConfig]): The config for the evolution
            process. If None, the environment will not be run in evolution mode.
    """

    training_config: MjCambrianTrainingConfig
    env_config: MjCambrianEnvConfig
    evo_config: Optional[MjCambrianEvoConfig] = None


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Dataclass/YAML Tester")

    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument(
        "-o",
        "--override",
        "--overrides",
        dest="overrides",
        action="extend",
        nargs="+",
        type=str,
        help="Override config values. Do <config>.<key>=<value>",
        default=[],
    )

    parser.add_argument("--no-resolve", action="store_true", help="Don't resolve config")
    parser.add_argument("-q", "--quiet", action="store_true", help="Run in quiet mode")

    args = parser.parse_args()

    t0 = time.time()
    config = MjCambrianConfig.load(args.config, overrides=args.overrides, instantiate=not args.no_resolve)
    t1 = time.time()

    print(f"Loaded config in {t1 - t0:.4f} seconds")

    if not args.quiet:
        if args.no_resolve:
            print(OmegaConf.to_yaml(config))
        else:
            print(config)
