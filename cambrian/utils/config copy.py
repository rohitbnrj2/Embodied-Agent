from typing import Dict, Any, Tuple, Optional, List, Self, Iterable, Callable, TypeAlias
import os
from dataclasses import dataclass, field
from pathlib import Path
from enum import Flag, auto
from functools import partial, cache, wraps
from abc import ABC
from typing_extensions import Literal

from omegaconf import OmegaConf, DictConfig, ListConfig, MissingMandatoryValue
import hydra_zen as zen
import torch

import mujoco as mj
import hydra_zen.typing

class CustomBuilds(zen.BuildsFn[hydra_zen.typing.CustomConfigType]):
    @classmethod
    def _make_hydra_compatible(
        cls, value, **kw
    ) -> Any:
        # Take some value and return a Hydra-compatible config for it.
        return super()._make_hydra_compatible(value, **kw)


class Key:
    __slots__ = ("obj", "key", "name")

    def __init__(self, obj: "MjCambrianDictConfig", name: str):
        self.obj = obj
        self.key = getattr(obj, "_key", lambda: None)() or ""
        self.name = name

    def __hash__(self):
        return hash((self.key, self.name))

    def __eq__(self, other):
        if isinstance(other, Key):
            return (self.key, self.name) == (other.key, other.name)
        return NotImplemented


def cache_key(fn):
    @cache
    def key_fn(key: Key):
        return fn(key.obj, key.name)

    @wraps(fn)
    def wrapper(self: "MjCambrianDictConfig", name: str):
        return key_fn(Key(self, name))

    return wrapper


class MjCambrianContainerConfig(ABC):
    _config: DictConfig | ListConfig

    def __init__(
        self,
        content: DictConfig | ListConfig | Self,
        /,
        *args,
        structured: Optional["MjCambrianBaseConfig"] = None,
        config: Optional[DictConfig | ListConfig] = None,
        **kwargs,
    ):
        self.__dict__["_config"] = config or content
        if structured:
            content = self.instantiate(content, structured=structured, *args, **kwargs)
        super().__init__(content, *args, **kwargs)

    def instantiate(
        self,
        config: DictConfig | ListConfig | Self,
        structured: "MjCambrianConfig",
        *args,
        **kwargs,
    ) -> Self:
        config = OmegaConf.merge(structured, zen.instantiate(config, *args, **kwargs))
        if keys := OmegaConf.missing_keys(config):
            config._format_and_raise(
                key=next(iter(keys)),
                value=None,
                cause=MissingMandatoryValue("Missing mandatory value"),
            )
        if isinstance(config, DictConfig):
            config = MjCambrianDictConfig(config, config=self._config)
        elif isinstance(config, ListConfig):
            config = MjCambrianListConfig(config, config=self._config)
        return config

    @cache_key
    def __getattr__(self, name: str):
        attr = super().__getattr__(name)
        config = getattr(self._config, name, None)
        if isinstance(attr, DictConfig):
            return MjCambrianDictConfig(attr, config=config)
        elif isinstance(attr, ListConfig):
            return MjCambrianListConfig(attr, config=config)
        return attr

    @cache_key
    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_container(self) -> Dict[str, Any]:
        return OmegaConf.to_container(self._config)

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(self._config)

    def save(self, path: Path | str):
        """Save the config to a yaml file."""
        with open(path, "w") as f:
            f.write(self.to_yaml())

    def __str__(self) -> str:
        return self.to_yaml()


class MjCambrianDictConfig(MjCambrianContainerConfig, DictConfig):
    pass


class MjCambrianListConfig(MjCambrianContainerConfig, ListConfig):
    pass


OmegaConf.register_new_resolver("eval", eval, replace=True)


def search(
    key: str | None = None,
    /,
    mode: Optional[str] = "value",
    *,
    depth: int = 0,
    _parent_: DictConfig,
) -> Any:
    """This method will recursively search up the parent chain for the key and return
    the value. If the key is not found, will raise a KeyError.

    For instance, a heavily nested value might want to access a value some level
    higher but it may be hazardous to use relative paths (i.e. ${..key}) since
    the config may be changed. Instead, we'll search up for a specific key to set the
    value to. Helpful for setting unique names for an object in a nested config.

    NOTE: This technically uses hidden attributes (i.e. _parent).

    Args:
        key (str | None): The key to search for. Could be none (like when mode is
            "parent_key").
        mode (Optional[str]): The mode to use. Defaults to "value". Available modes:
            - "value": Will return the value of the found key. Key must be set.
            - "parent_key": Will return the parent's key. If key is None, won't do
                any recursion and will return the parent's key.
        depth (int, optional): The depth of the search. Used internally
            in this method and unsettable from the config. Avoids checking the parent
            key.
        _parent_ (DictConfig): The parent config to search in.
    """
    if _parent_ is None:
        # Parent will be None if we're at the top level
        raise KeyError(f"Key {key} not found in parent chain.")

    if mode == "value":
        if key in _parent_:
            # If the key is in the parent, we'll return the value
            return _parent_[key]
        else:
            # Otherwise, we'll keep searching up the parent chain
            return search(key, mode=mode, depth=depth + 1, _parent_=_parent_._parent)
    elif mode == "parent_key":
        if key is None:
            # If the key is None, we'll return the parent's key
            assert _parent_._key() is not None, "Parent key is None."
            return _parent_._key()

        if depth != 0 and isinstance(_parent_, DictConfig) and key in _parent_:
            # If we're at a key that's not the parent and the parent has the key we're
            # looking for, we'll return the parent
            return search(None, mode=mode, depth=depth + 1, _parent_=_parent_)
        else:
            # Otherwise, we'll keep searching up the parent chain
            return search(key, mode=mode, depth=depth + 1, _parent_=_parent_._parent)

OmegaConf.register_new_resolver("search", search, replace=True)
OmegaConf.register_new_resolver(
    "parent", partial(search, mode="parent_key"), replace=True
)


def config_wrapper(cls=None, /, dataclass_kwargs: Dict[str, Any] | None = ...):
    """This is a wrapper of the dataclass decorator that adds the class to the hydra
    store.

    The hydra store is used to construct structured configs from the yaml files.
    NOTE: Only some primitive datatypes are supported by Hydra/OmegaConf.

    Args:
        dataclass_kwargs (Dict[str, Any] | None): The kwargs to pass to the dataclass
            decorator. If unset, will use the defaults. If set to None, the class
            will not be wrapped as a dataclass.
    """

    # Update the kwargs for the dataclass with some defaults
    # NOTE: Can't use slots: https://github.com/python/cpython/issues/90562
    default_dataclass_kwargs = dict(repr=False, eq=False, slots=True, kw_only=True)
    if dataclass_kwargs is ...:
        # Set to the default dataclass kwargs
        dataclass_kwargs = default_dataclass_kwargs
    elif isinstance(dataclass_kwargs, dict):
        # Update the default dataclass kwargs with the given dataclass kwargs
        dataclass_kwargs = {**default_dataclass_kwargs, **dataclass_kwargs}

    def wrapper(cls):
        if dataclass_kwargs is not None:
            # new_cls = zen.hydrated_dataclass(dict, **dataclass_kwargs)(cls)
            cls = dataclass(cls, **dataclass_kwargs)

        # Add to the hydra store
        # By adding it to the zen store rather than the hydra store directly, we can
        # support partial types (as in types that are not allowed by OmegaConf/hydra).
        # For instance, if we want to type hint a class or function, this would not be
        # allowed by OmegaConf/hydra. But by adding it to the zen store, we can support
        # these types.
        if (None, cls.__name__) not in zen.store:
            (new_cls,) = (zen.builds(cls, populate_full_signature=True),)
            zen.store(
                new_cls,
                name=cls.__name__,
                zen_dataclass={**dataclass_kwargs, "cls_name": new_cls.__name__},
                builds_bases=(cls,),
            )

        return cls

    if cls is None:
        return wrapper
    return wrapper(cls)


def mujoco_wrapper(instance, **kwargs):
    """This wrapper will wrap a mujoco class and convert it into a dataclass which we
    can use to build structured configs. Mujoco classes don't have __init__ methods,
    so we'll use the __dict__ to get the fields of the class.

    Should be called as follows:
    _target_: cambrian.utils.config.mujoco_wrapper
    instance:
        _target_: <mujoco_class>
    """

    def setattrs(instance, **kwargs):
        try:
            for key, value in kwargs.items():
                setattr(instance, key, value)
        except Exception as e:
            raise ValueError(
                f"In mujoco_wrapper, got error when setting attribute "
                f"{key=} to {value=}: {e}"
            )
        return instance

    if isinstance(instance, partial):
        # If the instance is a partial, we'll setup a wrapper such that once the
        # partial is actually instantiated, we'll set the attributes of the instance
        # with the kwargs.
        partial_instance = instance
        config_kwargs = kwargs

        def wrapper(*args, **kwargs):
            instance = partial_instance(*args, **kwargs)
            return setattrs(instance, **config_kwargs)

        return wrapper
    else:
        return setattrs(instance, **kwargs)

def mujoco_flags_wrapper(instance, key, flag_type, **flags):
    def setattrs(instance, key, flag_type, **flags):
        attr = getattr(instance, key)
        for flag, value in flags.items():
            flag = getattr(flag_type, flag)
            attr[flag] = value
        return attr

    if isinstance(instance, partial):
        partial_instance = instance
        config_key = key
        config_type = flag_type
        config_flags = flags

        def wrapper(*args, **kwargs):
            instance = partial_instance(*args, **kwargs)
            return setattrs(instance, config_key, config_type, **config_flags)

        return wrapper
    else:
        return setattrs(instance, key, flag_type, **flags)


@config_wrapper
class MjCambrianBaseConfig(MjCambrianDictConfig):
    """Base config for all configs.

    NOTE: This class inherits from MjCambrianDictConfig which is a subclass of 
    DictConfig. There are issues with inheriting from DictConfig and instantiating an
    instance using the hydra instantiate or omegaconf.to_object methods. So these
    classes aren't meant to be instantiated, but are used for type hinting and
    validation of the config files.

    Attributes:
        custom (Optional[Dict[Any, str]]): Custom data to use. This is useful for
            code-specific logic (i.e. not in yaml files) where you want to store
            data that is not necessarily defined in the config.
    """

    custom: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def instantiate(
        cls, dict_config: DictConfig, **kwargs
    ) -> Self | MjCambrianDictConfig:
        return MjCambrianDictConfig(dict_config, structured=cls, **kwargs)


MjCambrianXMLConfig: TypeAlias = Any
"""Actual type: List[Dict[str, Self]]

The actual type is a nested list of dictionaries that's recursively defined. We use a
list here because then we can have non-unique keys.

This defines a custom xml config. This can be used to define custom xmls which 
are built from during the initialization phase of the environment. The config is 
structured as follows:

```yaml
parent_key1:
    - child_key1: 
        - attr1: val1
        - attr2: val2
    - child_key2:
        - attr1: val1
        - attr2: val2
child_key1:
    - child_key2:
        - attr1: ${parent_key1.child_key1.attr2}
- child_key2:
    - child_key3: ${parent_key1.child_key1}
```

which will construct an xml that looks like:

```xml
<parent_key1>
    <child_key1 attr1="val1" attr2="val2">
        <child_key2 attr1="val2"/>
    </child_key1>
    <child_key2>
        <attr1>val1</attr1>
        <attr2>val2</attr2>
        <child_key3 attr1="val1" attr2="val2">
    </child_key2>
</parent_key1>
```

This is a verbose representation for xml files. This is done
to allow interpolation through hydra/omegaconf in the xml files and without the need
for a complex xml parser omegaconf resolver.

TODO: I think this type (minus the Self) is supported as of OmegaConf issue #890.
"""

MjCambrianActivationFn: TypeAlias = torch.nn.Module
"""Actual type: torch.nn.Module"""

MjCambrianRewardFn: TypeAlias = Any
"""Actual type: Callable[[MjCambrianAnimal, Dict[str, Any], ...], float]"""

MjCambrianMazeSelectionFn: TypeAlias = Any
"""Actual type: Callable[[MjCambrianEnv, ...], MjCambrianMaze]"""

MjCambrianModelType: TypeAlias = Any
"""Actual type: MjCambrianModel"""

MjCambrianCallbackListType: TypeAlias = Any
"""Actual type: MjCambrianCallbackList"""


@config_wrapper
class MjCambrianTrainingConfig(MjCambrianBaseConfig):
    """Settings for the training process. Used for type hinting.

    Attributes:
        total_timesteps (int): The total number of timesteps to train for.
        max_episode_steps (int): The maximum number of steps per episode.
        n_envs (int): The number of parallel environments to use for training.

        model (MjCambrianModelType): The model to use for training.
        callbacks (MjCambrianCallbackListType): The callbacks to use for training.
    """

    total_timesteps: int
    max_episode_steps: int
    n_envs: int

    model: MjCambrianModelType
    callbacks: MjCambrianCallbackListType


@config_wrapper
class MjCambrianMazeConfig(MjCambrianBaseConfig):
    """Defines a map config. Used for type hinting.

    Attributes:
        ref (Optional[str]): Reference to a named maze config. Used to share walls and
            other geometries/assets. A check will be done to ensure the walls are
            identical between configs.

        map (List[List[str]]): The map to use for the maze. It's a 2D array where
            each element is a string and corresponds to a "pixel" in the map. See
            `maze.py` for info on what different strings mean.
        xml (str): The xml for the maze. This is the xml that will be used to
            create the maze.

        difficulty (float): The difficulty of the maze. This is used to determine
            the selection probability of the maze when the mode is set to "DIFFICULTY".
            The value should be set between 0 and 100, where 0 is the easiest and 100
            is the hardest.

        size_scaling (float): The maze scaling for the continuous coordinates in the
            MuJoCo simulation.
        height (float): The height of the walls in the MuJoCo simulation.
        flip (bool): Whether to flip the maze or not. If True, the maze will be
            flipped along the x-axis.
        smooth_walls (bool): Whether to smooth the walls such that they are continuous
            appearing. This is an approximated as a spline fit to the walls.

        hide_targets (bool): Whether to hide the target or not. If True, the target
            will be hidden.
        use_target_light_sources (bool): Whether to use a target light sources or not.
            If False, the colored target sites will be used (e.g. a red sphere).
            Otherwise, a light source will be used. The light source is simply a spot
            light facing down.

        wall_texture_map (Dict[str, List[str]]): The mapping from texture id to
            texture names. Textures in the list are chosen at random. If the list is of
            length 1, only one texture will be used. A length >= 1 is required.
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

    ref: Optional[str] = None

    map: Optional[str] = None
    xml: MjCambrianXMLConfig

    difficulty: float

    size_scaling: float
    height: float
    flip: bool
    smooth_walls: bool

    hide_targets: bool
    use_target_light_sources: bool

    wall_texture_map: Dict[str, List[str]]

    init_goal_pos: Optional[Tuple[float, float]] = None
    eval_goal_pos: Optional[Tuple[float, float]] = None

    use_adversary: bool
    init_adversary_pos: Optional[Tuple[float, float]] = None
    eval_adversary_pos: Optional[Tuple[float, float]] = None

@config_wrapper
class MjCambrianRendererConfig(MjCambrianBaseConfig):
    """The config for the renderer. Used for type hinting.

    A renderer corresponds to a single camera. The renderer can then view the scene in
    different ways, like offscreen (rgb_array) or onscreen (human).

    Attributes:
        render_modes (List[str]): The render modes to use for the renderer. See
            `MjCambrianRenderer.metadata["render.modes"]` for options.

        maxgeom (Optional[int]): The maximum number of geoms to render.

        width (int): The width of the rendered image. For onscreen renderers, if this
            is set, the window cannot be resized. Must be set for offscreen renderers.
        height (int): The height of the rendered image. For onscreen renderers, if this
            is set, the window cannot be resized. Must be set for offscreen renderers.

        fullscreen (Optional[bool]): Whether to render in fullscreen or not. If True,
            the width and height are ignored and the window is rendered in fullscreen.
            This is only valid for onscreen renderers.

        camera (Optional[MjCambrianCameraConfig]): The camera config to use for
            the renderer.
        scene_options (Optional[Dict[str, Any]]): The scene options to use for the
            renderer. Keys are the name of the option as defined in MjvOption. For
            array options (like `flags`), the value should be another dict where the
            keys are the indices/mujoco enum keys and the values are the values to set.

        use_shared_context (bool): Whether to use a shared context or not.
            If True, the renderer will share a context with other renderers. This is
            useful for rendering multiple renderers at the same time. If False, the
            renderer will create its own context. This is computationally expensive if
            there are many renderers.
    """

    render_modes: List[str]

    width: Optional[int] = None
    height: Optional[int] = None

    fullscreen: Optional[bool] = None

    camera: Optional[mj.MjvCamera] = None
    scene: Optional[Any] = None
    scene_options: Optional[Any] = None

    use_shared_context: bool


@config_wrapper
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
            `sensorsize`, if set. Fmt: fovx fovy.

        enable_optics (bool): Whether to enable optics or not.
        enable_aperture (bool): Whether to enable the aperture or not.
        enable_lens (bool): Whether to enable the lens or not.
        enable_phase_mask (bool): Whether to enable the phase mask or not.

        scene_angular_resolution: The angular resolution of the scene. This is used to
            determine the field of view of the scene. Specified in degrees.
        pixel_size: The pixel size of the sensor in meters.
        sensor_resolution (Tuple[int, int]): TODO
        add_noise (bool): TODO
        noise_std (float): TODO
        aperture_open (float): The aperture open value. This is the radius of the
            aperture. The aperture is a circle that is used to determine which light
            rays to let through. Only used if `enable_aperture` is True. Must be
            between 0 and 1.
        aperture_radius (float): The aperture radius value.
        wavelengths (Tuple[float, float, float]): The wavelengths to use for the
            intensity sensor. Fmt: wavelength_1 wavelength_2 wavelength_3
        depth_bins (int): The number of depth bins to use for the depth dependent psf.

        load_height_mask_from_file (bool): Whether to load the height mask from file or
            not. If True, the height mask will be loaded from the file specified in
            `height_mask_from_file`. If False, the psf wil be randomized or set to zeros
            using `randomize_psf_init`.
        height_mask_from_file (Optional[str]): The path to the height mask file to load.
        randomize_psf_init (bool): Whether to randomize the psf or not. If True, the psf
            will be randomized. If False, the psf will be set to zeros. Only used if
            `load_height_mask_from_file` is False.
        zernike_basis_path (Optional[str]): The path to the zernike basis file to load.
        psf_filter_size (Tuple[int, int]): The psf filter size. This is
            convolved across the image, so the actual resolution of the image is plus
            psf_filter_size / 2. Only used if `load_height_mask_from_file` is False.
            Otherwise the psf filter size is determined by the height mask.
        refractive_index (float): The refractive index of the eye.
        min_phi_defocus (float): TODO
        max_phi_defocus (float): TODO

        load_height_mask_from_file (bool): Whether to load the height mask from file or
            not. If True, the height mask will be loaded from the file specified in
            `height_mask_from_file`. If False, the psf wil be randomized or set to zeros
            using `randomize_psf_init`.
        height_mask_from_file (Optional[str]): The path to the height mask file to load.
        randomize_psf_init (bool): Whether to randomize the psf or not. If True, the psf
            will be randomized. If False, the psf will be set to zeros. Only used if
            `load_height_mask_from_file` is False.
        zernike_basis_path (Optional[str]): The path to the zernike basis file to load.

        psf_filter_size (Tuple[int, int]): The psf filter size. This is
            convolved across the image, so the actual resolution of the image is plus
            psf_filter_size / 2. Only used if `load_height_mask_from_file` is False.
            Otherwise the psf filter size is determined by the height mask.
        refractive_index (float): The refractive index of the eye.
        depth_bins (int): The number of depth bins to use for the depth sensor.
        min_phi_defocus (float): The minimum depth to use for the depth sensor.
        max_phi_defocus (float): The maximum depth to use for the depth sensor.
        wavelengths (Tuple[float, float, float]): The wavelengths to use for the
            intensity sensor. Fmt: wavelength_1 wavelength_2 wavelength_3
        #### Optics Params

        pos (Optional[Tuple[float, float, float]]): The initial position of the camera.
            Fmt: xyz
        quat (Optional[Tuple[float, float, float, float]]): The initial rotation of the
            camera. Fmt: wxyz.
        fovy (Optional[float]): The vertical field of view of the camera.
        focal (Optional[Tuple[float, float]]): The focal length of the camera.
            Fmt: focal_x focal_y.
        sensorsize (Optional[Tuple[float, float]]): The sensor size of the camera.
            Fmt: sensor_x sensor_y.

        coord (Tuple[float, float]): The x and y coordinates of the eye.
            This is used to determine the placement of the eye on the animal.
            Specified in degrees. Mutually exclusive with `pos` and `quat`. This attr
            isn't actually used by eye, but by the animal. The eye has no knowledge
            of the geometry it's trying to be placed on. Fmt: lat lon

        renderer (MjCambrianRendererConfig): The renderer config to use for the
            underlying renderer. The width and height of the renderer will be set to the
            padded resolution (resolution + int(psf_filter_size/2)) of the eye.
    """

    name: Optional[str] = None

    mode: str
    resolution: Tuple[int, int]
    fov: Tuple[float, float]

    enable_optics: bool
    enable_aperture: bool
    enable_lens: bool
    enable_phase_mask: bool

    scene_resolution: Tuple[int, int]
    scene_angular_resolution: float
    pixel_size: float
    sensor_resolution: Tuple[int, int]
    add_noise: bool
    noise_std: float
    aperture_open: float
    aperture_radius: float
    wavelengths: Tuple[float, float, float]
    depth_bins: int

    load_height_mask_from_file: bool
    height_mask_from_file: Optional[str] = None
    randomize_psf_init: bool
    zernike_basis_path: Optional[str] = None
    psf_filter_size: Tuple[int, int]
    refractive_index: float
    min_phi_defocus: float
    max_phi_defocus: float

    pos: Optional[Tuple[float, float, float]] = None
    quat: Optional[Tuple[float, float, float, float]] = None
    fovy: Optional[float] = None
    focal: Optional[Tuple[float, float]] = None
    sensorsize: Optional[Tuple[float, float]] = None

    coord: Tuple[float, float]

    renderer: MjCambrianRendererConfig

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


@config_wrapper
class MjCambrianAnimalConfig(MjCambrianBaseConfig):
    """Defines the config for an animal. Used for type hinting.

    Attributes:
        xml (str): The xml for the animal. This is the xml that will be used to create
            the animal. You should use ${parent:xml} to generate named attributes. This
            will search upwards in the yaml file to find the name of the animal.

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

        initial_state (Optional[List[float | None]]): The initial state of the animal.
            The length of the list should be equal or less than the number of qpos
            variables that correspond to the joint defined by `joint_name`. If less
            than the number of qpos variables, the remaining qpos variables will be
            unchanged. If None, the intial state will be generated randomly using
            `maze.generate_reset_pos` method.
        constant_actions (Optional[List[float | None]]): The constant velocity to use for
            the animal. If not None, the len(constant_actions) must equal number of
            actuators defined in the model. For instance, if there are 3 actuators
            defined and it's desired to have the 2nd actuator be constant, then
            constant_actions = [None, 0, None]. If None, no constant action will be
            applied.

        use_action_obs (bool): Whether to use the action observation or not.
        use_init_pos_obs (bool): Whether to use the initial position observation or not.
        use_current_pos_obs (bool): Whether to use the current position observation or
            not.
        n_temporal_obs (int): The number of temporal observations to use.

        eyes (Dict[str, MjCambrianEyeConfig]): The configs for the eyes.
            The key will be used as the name for the eye.

        mutations_from_parent (Optional[List[str]]): The mutations applied to the child
            (this animal) from the parent. This is unused during mutation; it simply
            is a record of the mutations that were applied to the parent.
    """

    xml: MjCambrianXMLConfig

    body_name: str
    joint_name: str
    geom_name: str

    eyes_lat_range: Tuple[float, float]
    eyes_lon_range: Tuple[float, float]

    initial_state: Optional[List[float | None]] = None
    constant_actions: Optional[List[float | None]] = None

    use_action_obs: bool
    use_init_pos_obs: bool
    use_current_pos_obs: bool
    n_temporal_obs: int

    eyes: Dict[str, MjCambrianEyeConfig]

    mutations_from_parent: List[str]


@config_wrapper
class MjCambrianEnvConfig(MjCambrianBaseConfig):
    """Defines a config for the cambrian environment.

    Attributes:
        xml (MjCambrianXMLConfig): The xml for the scene. This is the xml that will be
            used to create the environment. See `MjCambrianXMLConfig` for more info.

        reward_fn (MjCambrianRewardFn): The reward function type to use. See the
            `MjCambrianRewardFn` for more info.

        use_goal_obs (bool): Whether to use the goal observation or not.
        terminate_at_goal (bool): Whether to terminate the episode when the animal
            reaches the goal or not.
        truncate_on_contact (bool): Whether to truncate the episode when the animal
            makes contact with an object or not.
        distance_to_target_threshold (float): The distance to the target at which the
            animal is assumed to be "at the target".
        action_penalty (float): The action penalty when it moves.
        adversary_penalty (float): The adversary penalty when it goes to the wrong target.
        contact_penalty (float): The contact penalty when it contacts the wall.
        force_exclusive_contact_penalty (bool): Whether to force exclusive contact
            penalty or not. If True, the contact penalty will be used exclusively for
            the reward. If False, the contact penalty will be used in addition to the
            calculated reward.

        frame_skip (int): The number of mujoco simulation steps per `gym.step()` call.

        add_overlays (bool): Whether to add overlays or not.
        clear_overlays_on_reset (bool): Whether to clear the overlays on reset or not.
            Consequence of setting to False is that if `add_position_tracking_overlay`
            is True and mazes change between evaluations, the sites will be drawn on top
            of each other which may not be desired. When record is False, the overlays
            are always cleared.
        renderer (Optional[MjCambrianViewerConfig]): The default viewer config to
            use for the mujoco viewer. If unset, no renderer will be used. Should
            set to None if `render` will never be called. This may be useful to
            reduce the amount of vram consumed by non-rendering environments.

        eval_overrides (Optional[Dict[str, Any]]): Key/values to override the default
            env during evaluation. Applied during evaluation only. Merged directly
            with the env. The actual datatype is Self/MjCambrianEnvConfig but all
            attributes are optional. NOTE: This dict is only applied at reset,
            meaning mujoco xml changes will not be reflected in the eval episode.

        mazes (Dict[str, MjCambrianMazeConfig]): The configs for the mazes. Each
            maze will be loaded into the scene and the animal will be placed in a maze
            at each reset.
        maze_selection_fn (MjCambrianMazeSelectionFn): The function to use to select
            the maze. The function will be called at each reset to select the maze
            to use. See `MjCambrianMazeSelectionFn` and `maze.py` for more info.

        animals (List[MjCambrianAnimalConfig]): The configs for the animals.
            The key will be used as the default name for the animal, unless explicitly
            set in the animal config.
    """

    xml: MjCambrianXMLConfig

    reward_fn: MjCambrianRewardFn

    use_goal_obs: bool
    terminate_at_goal: bool
    truncate_on_contact: bool
    distance_to_target_threshold: float
    action_penalty: float
    adversary_penalty: float
    contact_penalty: float
    force_exclusive_contact_penalty: bool

    frame_skip: int

    add_overlays: bool
    clear_overlays_on_reset: bool
    renderer: Optional[MjCambrianRendererConfig] = None

    eval_overrides: Optional[Dict[str, Any]] = None

    mazes: Dict[str, MjCambrianMazeConfig]
    maze_selection_fn: MjCambrianMazeSelectionFn

    animals: Dict[str, MjCambrianAnimalConfig]


@config_wrapper
class MjCambrianPopulationConfig(MjCambrianBaseConfig):
    """Config for a population. Used for type hinting.

    Attributes:
        size (int): The population size. This represents the number of agents that
            should be trained at any one time. This is independent to the number of
            parallel envs; an agent represents a single model, where we launch many
            parallel envs to improve training. This number represents the former.
    """

    size: int


@config_wrapper
class MjCambrianSpawningConfig(MjCambrianBaseConfig):
    """Config for spawning. Used for type hinting.

    Attributes:
        init_num_mutations (int): The number of mutations to perform on the
            default config to generate the initial population. The actual number of
            mutations is calculated using random.randint(1, init_num_mutations).
        num_mutations (int): The number of mutations to perform on the parent
            generation to generate the new generation. The actual number of mutations
            is calculated using random.randint(1, num_mutations).
        mutations (List[str]): The mutation options to use for the animal. See
            `MjCambrianAnimal.MutationType` for options.
        mutation_options (Optional[Dict[str, Any]]): The options to use for
            the mutations.

        load_policy (bool): Whether to load a policy or not. If True, the parent's
            saved policy will be loaded and used as the starting point for the new
            generation. If False, the child will be trained from scratch.

        replication_type (str): The type of replication to use. See
            `ReplicationType` for options.
    """

    init_num_mutations: int
    num_mutations: int
    mutations: List[str]
    mutation_options: Optional[Dict[str, Any]] = None

    load_policy: bool

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


@config_wrapper
class MjCambrianEvoConfig(MjCambrianBaseConfig):
    """Config for evolutions. Used for type hinting.

    Attributes:
        num_nodes (int): The number of nodes used for the evolution process. By default,
            this should be 1. And then if multiple evolutions are run in parallel, the
            number of nodes should be set to the number of evolutions.
        max_n_envs (int): The maximum number of environments to use for
            parallel training. Will set `n_envs` for each training process to
            `max_n_envs // population size`.
        num_generations (int): The number of generations to run for.

        population (MjCambrianPopulationConfig): The config for the population.
        spawning (MjCambrianSpawningConfig): The config for the spawning process.

        rank (int): The rank of the current evolution. Will be set by the evolution
            runner. A rank is essentially the id of the current evolution process.
        generation (int): The generation of the current evolution. Will be set by the
            evolution runner.

        generation (Optional[MjCambrianGenerationConfig]): The config for the
            current generation. Will be set by the evolution runner.
        parent_generation (Optional[MjCambrianGenerationConfig]): The config for
            the parent generation. Will be set by the evolution runner. If None, that
            means that the current generation is the first generation (i.e. no parent).

        top_performers (Optional[List[str]]): The top performers from the parent
            generation. This was used to select an animal to spawn an offspring from.
            Used for parsing after the fact.

        environment_variables (Optional[Dict[str, str]]): The environment variables to
            set for the training process.
    """

    num_nodes: int
    max_n_envs: int
    num_generations: int

    population: MjCambrianPopulationConfig
    spawning: MjCambrianSpawningConfig

    # rank: int
    # generation: int

    # parent_rank: int
    # parent_generation: int

    top_performers: Optional[List[str]] = None

    environment_variables: Dict[str, str]


# @config_wrapper
class MjCambrianConfig(MjCambrianBaseConfig):
    """The base config for the mujoco cambrian environment. Used for type hinting.

    Attributes:
        logdir (str): The directory to log training data to.
        expname (str): The name of the experiment. Used to name the logging
            subdirectory. If unset, will set to the name of the config file.

        seed (int): The base seed used when initializing the default thread/process.
            Launched processes should use this seed value to calculate their own seed
            values. This is used to ensure that each process has a unique seed.

        training (MjCambrianTrainingConfig): The config for the training process.
        env (MjCambrianEnvConfig): The config for the environment.
        evo (Optional[MjCambrianEvoConfig]): The config for the evolution
            process. If None, the environment will not be run in evolution mode.
        logging (Optional[Dict[str, Any]]): The config for the logging process.
            Passed to `logging.config.dictConfig`.
    """

    logdir: str
    expname: str

    seed: int

    training: MjCambrianTrainingConfig
    env: MjCambrianEnvConfig
    evo: Optional[MjCambrianEvoConfig] = None
    logging: Optional[Dict[str, Any]] = None


def setup_hydra(main_fn: Optional[Callable[["MjCambrianConfig"], None]] = None, /):
    """This function is the main entry point for the hydra application.

    Args:
        main_fn (Callable[["MjCambrianConfig"], None]): The main function to be called
            after the hydra configuration is parsed.
    """
    import hydra

    zen.store.add_to_hydra_store()

    def hydra_argparse_override(fn: Callable, /):
        """This function allows us to add custom argparse parameters prior to hydra
        parsing the config.

        We want to set some defaults for the hydra config here. This is a workaround
        in a way such that we don't

        Note:
            Augmented from hydra discussion #2598.
        """
        import sys
        import argparse

        parser = argparse.ArgumentParser()
        parsed_args, unparsed_args = parser.parse_known_args()

        # By default, argparse uses sys.argv[1:] to search for arguments, so update
        # sys.argv[1:] with the unparsed arguments for hydra to parse (which uses
        # argparse).
        sys.argv[1:] = unparsed_args

        return fn if fn is not None else lambda fn: fn

    @hydra_argparse_override
    @hydra.main(
        version_base=None, config_path=f"{os.getcwd()}/configs", config_name="base"
    )
    def main(cfg: DictConfig):
        config = MjCambrianConfig.instantiate(cfg)

        main_fn(config)
        pass

    main()


if __name__ == "__main__":
    import time

    t0 = time.time()

    def main(config: MjCambrianConfig):
        # print(config)
        config.save("config.yaml")
        pass

    setup_hydra(main)
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s")
