from typing import (
    Dict,
    Any,
    Optional,
    Self,
    Type,
    KeysView,
    ItemsView,
    ValuesView,
    List,
    Iterator,
)
from dataclasses import field, dataclass, fields, make_dataclass
from pathlib import Path
from functools import partial
import enum

import numpy as np
import hydra_zen as zen
from hydra.core.config_store import ConfigStore
from omegaconf import (
    OmegaConf,
    DictConfig,
    ListConfig,
    MissingMandatoryValue,
    Node,
    MISSING,
)
from omegaconf.errors import ConfigKeyError


# =============================================================================
# Config classes and methods


class MjCambrianContainerConfig:
    """This is a wrapper around the OmegaConf DictConfig and ListConfig classes.

    Internally, hydra use OmegaConf to parse yaml/config files. OmegaConf
    uses an internal class DictConfig (and ListConfig for lists) to represent the
    dictionary data types. This is immutable and inheritance isn't easy, so this class
    allows us to wrap the DictConfig and ListConfig classes to add some additional
    methods. OmegaConf uses functional-style programming, where the OmegaConf class
    provides methods with which you pass a DictConfig or ListConfig instance to.
    Instead, we wrap the DictConfig and ListConfig classes to provide object-oriented
    programming, where we can call methods on the instance itself, as well as
    additional custom methods.

    We'll keep around two instances of DictConfig or ListConfig: `_config` and
    `_content`. `_config` is the original, uninstantiated config. This is strictly yaml
    and does not include any instantiated objects. `_content` is the instantiated
    config. When getting an attribute, we'll get the attribute from `_content` and
    return the wrapped instance by this class. `_config` is used to export in a human-
    readable format to a yaml file.

    Args:
        content (DictConfig | ListConfig): The instantiated config.

    Keyword Args:
        config (Optional[DictConfig | ListConfig]): The original, uninstantiated
            config. If unset, will use the content as the config.
    """

    _config: DictConfig | ListConfig
    _content: DictConfig | ListConfig

    def __init__(
        self,
        content: DictConfig | ListConfig,
        /,
        config: Optional[DictConfig | ListConfig] = None,
    ):
        # Must use __dict__ to set the attributes since we're overriding the
        # __getattr__ method. If config is None, we'll just set it to the content.
        self.__dict__["_content"] = content
        self.__dict__["_config"] = config or content

    @classmethod
    def instantiate(
        cls,
        config: DictConfig | ListConfig,
        *,
        as_container: bool = False,
        **kwargs,
    ) -> Self | DictConfig | ListConfig:
        """Instantiate the config using the structured config. Will check for missing
        keys and raise an error if any are missing."""
        # Resolve the config prior to instantiation. This will resolve any high level
        # interpolations before instantation such that the interpolations which are
        # configs in themselves will be the uninstantiated config. This is important
        # since the instantiated config has objects that aren't picklable.
        OmegaConf.resolve(config)

        # First instantiate the config (will replace _target_ with the actual class)
        # And then merge the structured config with the instantiated config to give it
        # validation.
        content: DictConfig | ListConfig = zen.instantiate(config, **kwargs)

        # This is redundant for most cases, but instantiated logic may return strings
        # which may have interpolations. We'll explicitly call resolve to resolve these
        # interpolations.
        OmegaConf.resolve(content)

        # Check for missing values. Error message will only show the first missing key.
        if keys := OmegaConf.missing_keys(content):
            content._format_and_raise(
                key=next(iter(keys)),
                value=None,
                cause=MissingMandatoryValue("Missing mandatory value"),
            )

        if as_container:
            return content
        else:
            config = config.copy()
            OmegaConf.set_struct(config, False)
            return MjCambrianContainerConfig(content, config=config)

    @classmethod
    def load(
        cls, *args, instantiate: bool = True, as_container: bool = False, **kwargs
    ) -> Self | DictConfig | ListConfig:
        """Wrapper around OmegaConf.load to instantiate the config."""
        loaded = OmegaConf.load(*args, **kwargs)
        if instantiate:
            return cls.instantiate(loaded, as_container=as_container)
        else:
            return loaded

    @classmethod
    def create(
        cls, *args, instantiate: bool = True, as_container: bool = False, **kwargs
    ) -> Self | DictConfig | ListConfig:
        """Wrapper around OmegaConf.create to instantiate the config."""
        created = OmegaConf.create(*args, **kwargs)
        if instantiate:
            return cls.instantiate(created, as_container=as_container)
        else:
            return created

    def select(self, key: str, *, use_instantiated: bool = False, **kwargs) -> Any:
        """This is a wrapper around OmegaConf.select to select a key from the config.

        NOTE: By default, this will use the uninstantiated config object to select the
        key. Pass `use_instantiated=True` to use the instantiated config object.
        """
        config = self._content if use_instantiated else self._config
        return OmegaConf.select(config, key, **kwargs)

    def get_type(self, **kwargs) -> Type[Any]:
        """Wrapper around OmegaConf.get_type to get the type of the config."""
        return OmegaConf.get_type(self._content, **kwargs)

    def to_container(
        self, *, use_instantiated: bool = True, **kwargs
    ) -> Dict[Any, Any]:
        """Wrapper around OmegaConf.to_container to convert the config to a
        dictionary."""
        if use_instantiated:
            return OmegaConf.to_container(self._content, **kwargs)
        else:
            return OmegaConf.to_container(self._config, **kwargs)

    def to_yaml(self) -> str:
        """Wrapper around OmegaConf.to_yaml to convert the config to a yaml string.
        Adds some custom representers."""
        import yaml

        def str_representer(dumper: yaml.Dumper, data: str):
            style = None
            if "\n" in data:
                # Will use the | style for multiline strings.
                style = "|"
            elif data == MISSING:
                # Will wrap ??? in quotes, yaml doesn't like this otherwise
                style = '"'
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)

        def flag_representer(dumper: yaml.Dumper, data: enum.Flag):
            data = "|".join([m.name for m in type(data) if m in data])
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        dumper = yaml.CDumper
        dumper.add_representer(str, str_representer)
        dumper.add_multi_representer(enum.Flag, flag_representer)
        return yaml.dump(
            self.to_container(use_instantiated=False),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            Dumper=dumper,
        )

    def save(self, path: Path | str):
        """Saves the config to a yaml file."""
        with open(path, "w") as f:
            f.write(self.to_yaml())

    def keys(self) -> KeysView[Any]:
        """Wrapper of the keys method to return the keys of the content."""
        return self._content.keys()

    def values(self) -> ValuesView[Any]:
        """Wrapper of the values method to return the values of the content as a
        MjCambrianContainerConfig if the item is a OmegaConf config."""
        values: List[Any] = []

        for key, value in self._content.items():
            if OmegaConf.is_config(value):
                # Grab the config without validation. This uses internal methods.
                config = self._config._get_child(key, False, False)
                values.append(MjCambrianContainerConfig(value, config=config))
            else:
                values.append(value)
        return values

    def items(self) -> ItemsView[Any, Any]:
        """Wrapper of the items method to return the items of the content as a
        MjCambrianContainrConfig if the item is a OmegaConf config."""
        items: List[Dict[Any, Any]] = []

        for key, value in self._content.items():
            if OmegaConf.is_config(value):
                # Grab the config without validation. This uses internal methods.
                config = self._config._get_child(key, False, False)
                items.append((key, MjCambrianContainerConfig(value, config=config)))
            else:
                items.append((key, value))
        return items

    def copy(self) -> Self:
        """Wrapper around the copy method to return a new instance of this class."""
        content, config = self._content.copy(), self._config.copy()
        return MjCambrianContainerConfig(content, config=config)

    def __getattr__(self, name: str) -> Self | Any:
        """Get the attribute from the content and return the wrapped instance. If the
        attribute is a DictConfig or ListConfig, we'll wrap it in this class.

        TODO: Possible bug: if an attribute is instantiated with _target_ and the
        target returns a Dict, it will be wrapped by OmegaConf as a DictConfig. This
        means {_target_: ...} will be returned rather than the actual instantiated
        Dict object.
        """
        return self._get_impl(name)

    def __getitem__(self, key: Any) -> Self | Any:
        """Get the item from the content and return the wrapped instance. If the item is
        a DictConfig or ListConfig, we'll wrap it in this class."""
        return self._get_impl(key, is_getattr=False)

    def __setattr__(self, name: str, value: Any):
        """Set the attribute in the content. These are fairly slow, but we'll keep them
        using OmegaConf logic since it does validation."""
        if isinstance(value, MjCambrianContainerConfig):
            setattr(self._content, name, value._content)
            setattr(self._config, name, value._config)
        else:
            setattr(self._content, name, value)
            setattr(self._config, name, value)

    def __setitem__(self, key: Any, value: Any):
        """Set the item in the content. These are fairly slow, but we'll keep them using
        OmegaConf logic since it does validation."""
        if isinstance(value, MjCambrianContainerConfig):
            self._content[key] = value._content
            self._config[key] = value._config
        else:
            self._content[key] = value
            self._config[key] = value

    def __iter__(self) -> Iterator[Any]:
        """Only supported by ListConfig. Wrapper around the __iter__ method to return
        the iterator of the content. Will convert any DictConfig or ListConfig to this
        class."""
        for content, config in zip(self._content, self._config):
            if OmegaConf.is_config(content):
                yield MjCambrianContainerConfig(content, config=config)
            else:
                yield content

    def __len__(self) -> int:
        """Only supported by ListConfig. Wrapper around the __len__ method to return the
        length of the content."""
        return len(self._content)

    def __getstate__(self) -> Dict[str, Any]:
        """Get the state of the object to pickle it. Will only pickle the config since
        it's not guaranteed the instances in content can be pickled."""
        return self.to_container(use_instantiated=False, resolve=True)

    def __setstate__(self, state: Dict[str, Any]):
        """Set the state of the object from the pickled state."""
        config = OmegaConf.create(state)
        content = self.instantiate(config, as_container=True)
        self.__init__(content, config=config)

    def __str__(self) -> str:
        return self.to_yaml()

    # ===========
    # Internal utils

    def _get_impl(self, key: Any, default_value: Any = ..., *, is_getattr: bool = True):
        """Perform access of the underlying data structures. This is an optimized
        version which uses internal methods of OmegaConf. This is similar to
        DictConfig._get_impl, but optimized for this use case. We know it out configs
        won't be structured, so there are minor optimizations we can do.

        Args:
            key (Any): The key to access.
            default_value (Any): The default value to return if the key is not found.
        """
        content: Node
        config: Node
        is_config: bool
        if isinstance(self._content, ListConfig):
            # ListConfig only accepts integers as keys
            if is_getattr:
                try:
                    key = int(key)
                except ValueError:
                    self._content._format_and_raise(
                        key,
                        None,
                        AttributeError("ListConfig does not support attribute access"),
                    )
            content = self._content.__dict__["_content"][key]
            if is_config := OmegaConf.is_config(content):
                config = self._config.__dict__["_content"][key]
        else:  # DictConfig
            # Normalize the key. This will convert the key to allowed dictionary keys, like
            # converts 0, 1 if the key type is bool. Basically validates the key.
            key = self._content._validate_and_normalize_key(key)

            # Check that the key exists
            if key not in self._content.__dict__["_content"] and is_getattr:
                if default_value is ...:
                    self._content._format_and_raise(
                        key, None, ConfigKeyError(f"Key not found: {key!s}")
                    )
            # TODO: use [key] instead of get since it's faster
            content = self._content.__dict__["_content"].get(key, default_value)
            if is_config := OmegaConf.is_config(content):
                config = self._config.__dict__["_content"].get(key, default_value)

        # If the content is a config, we'll wrap it in this class
        if is_config:
            return MjCambrianContainerConfig(content, config=config)
        else:
            return content._value()


class MjCambrianDictConfig(MjCambrianContainerConfig, DictConfig):
    """This is a wrapper around the OmegaConf DictConfig class.

    It is intended that this class never actually be instantiated. Config classes
    should inherit from this class (or a base config class should inherit from this)
    such that when duck typing, all the methods of DictConfig and
    MjCambrianContainerConfig are available.
    """

    pass


def config_wrapper(cls=None, /, **kwargs):
    """This is a wrapper of the dataclass decorator that adds the class to the hydra
    store.

    The hydra store is used to construct structured configs from the yaml files.

    We'll also do some preprocessing of the dataclass fields such that all type hints
    are supported by hydra. Hydra only supports a certain subset of types, so we'll
    convert the types to supported types using the _sanitized_type method from
    hydra_zen.

    Keyword Args:
        kw: The kwargs to pass to the dataclass decorator. The following defaults
            are set:
            - repr: False
            - eq: False
            - slots: True
            - kw_only: True
    """

    # Update the kwargs for the dataclass with some defaults
    default_dataclass_kw = dict(repr=False, eq=False, slots=True, kw_only=True)
    kwargs = {**default_dataclass_kw, **kwargs}

    def wrapper(cls):
        # Preprocess the fields to convert the types to supported types
        # Only certain primitives are supported by hydra/OmegaConf, so we'll convert
        # these types to supported types using the _sanitized_type method from hydra_zen
        # We'll just include the fields that are defined in this class and not in a base
        # class.
        cls = dataclass(cls, **kwargs)

        new_fields = []
        for f in fields(cls):
            new_fields.append((f.name, zen.DefaultBuilds._sanitized_type(f.type), f))

        # Create the new dataclass with the sanitized types
        kwargs["bases"] = cls.__bases__
        hydrated_cls = make_dataclass(cls.__name__, new_fields, **kwargs)

        # Add to the hydra store
        ConfigStore().store(cls.__name__, hydrated_cls)

        return hydrated_cls

    if cls is None:
        return wrapper
    return wrapper(cls)


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

    custom: Optional[Dict[str, Any]] = field(default_factory=dict, init=False)


# =============================================================================
# OmegaConf resolvers


def register_new_resolver(*args, replace: bool = True, **kwargs):
    """Wrapper around OmegaConf.register_new_resolver to register a new resolver.
    Defaults to replacing the resolver if it already exists (opposite of the default
    in OmegaConf)."""
    OmegaConf.register_new_resolver(*args, replace=replace, **kwargs)


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
        raise ConfigKeyError(f"Key {key} not found in parent chain.")

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


register_new_resolver("search", search)
register_new_resolver("parent", partial(search, mode="parent_key"))
register_new_resolver("eval", lambda src: eval(src, {}, {"np": np}))


# =============================================================================
# Utilities for config loading


def instance_wrapper(*, instance: Type[Any], **kwargs):
    """This utility method will wrap a class instance to help with setting class
    attributes after initialization.

    Some classes, for instance, don't include all attributes in the constructor; this
    method will postpone setting these attributes until after __init__ is called and
    just set the attributes directly with setattr.

    This is intended to be called from a yaml config file like so:

    ```yaml
    obj_to_instantiate:
        _target_: <path_to>.instance_wrapper
        instance:
            _target_: <class>

            # these will be passed to the __init__ method
            _args_: [arg1, arg2]

            # these will be passed to the __init__ method as kwargs
            init_arg1: value1
            init_arg2: value2

        # these will be set as attributes after the __init__ method
        set_arg1: value1
        set_arg2: value2
    ```

    At instantiate time, init args are not always known. As such, you can leverage
    hydras partial instantiation logic, as well. Under the hood, the instance_wrapper
    method will wrap the partial instance created by hydra such that when it's
    constructor is actually called, the attributes will be set.

    ```yaml
    partial_obj_to_instantiate:
        _target_: <path_to>.instance_wrapper
        instance:
            _target_: <class>
            _partial_: True

            # these will be passed to the __init__ method
            _args_: [arg1, arg2]

            # these will be passed to the __init__ method as kwargs
            init_arg1: value1
            init_arg2: value2
            init_arg3: '???' # this is unknown at instantiate time and can be set later

        # these will be set as attributes after the __init__ method
        set_arg1: value1
        set_arg2: value2
    ```

    Args:
        instance (Type[Any]): The class instance to wrap.

    Keyword Args:
        kwargs: The attributes to set on the instance.
    """

    def setattrs(instance, **kwargs):
        try:
            for key, value in kwargs.items():
                # Special case if value is the wrapper in flag_wrapper
                if callable(value):
                    value = value()
                setattr(instance, key, value)
        except Exception as e:
            raise ValueError(f"Error when setting attribute {key=} to {value=}: {e}")
        return instance

    if isinstance(instance, partial):
        # If the instance is a partial, we'll setup a wrapper such that once the
        # partial is actually instantiated, we'll set the attributes of the instance
        # with the kwargs.
        partial_instance = instance
        config_kwargs = kwargs

        def wrapper(*args, **kwargs):
            # First instantiate the partial
            instance = partial_instance(*args, **kwargs)
            # Then set the attributes
            return setattrs(instance, **config_kwargs)

        return wrapper
    else:
        return setattrs(instance, **kwargs)


def instance_flag_wrapper(
    *,
    instance: Type[Any],
    key: str,
    flag_type: Optional[Type[Any]] = None,
    eval_flags: Optional[bool] = False,
    **flags,
):
    """This utility method will wrap a class instance to help with setting class
    attributes after initialization. As opposed to instance_wrapper, this method will
    set attribute flags on the instance. This is particularly useful for mujoco enums,
    which are stored in a list.

    This is intended to be called from a yaml config file and to be used in conjunction
    with the instance_wrapper method.

    ```yaml
    obj_to_instantiate:
        _target_: <path_to>.instance_wrapper
        instance:
            _target_: <class>

        # these will be set as flags on the instance
        flags:
            _target_: <path_to>.instance_flag_wrapper
            instance: ${..instance}                     # get the instance
            key: ${parent:}                             # gets the parent key; "flags"
            flag_type:
                _target_: <class>                       # the class of the flag

            # These will be set like so:
            # obj_to_instantiate.key[flag1] = value1
            # obj_to_instantiate.key[flag2] = value2
            # ...
            flag1: value1
            flag2: value2
            flag3: value3
    ```

    This also works for partial instances.

    Args:
        instance (Type[Any]): The class instance to wrap.
        key (str): The key to set the flags on.
        flag_type (Optional[Type[Any]]): The class of the flag. If unset, will use the
            flag directly.
        eval_flags (Optional[bool]): Whether to evaluate the flags. If True, will
            call eval on the flags. This is helpful if you want to use slices.
            Default: False. NOTE: this is note safe and should be used with caution.

    Keyword Args:
        flags: The flags to set on the instance.
    """

    def setattrs(instance, key, flag_type, return_instance, **flags):
        """Set the attributes on the instance."""
        attr = getattr(instance, key)
        for flag, value in flags.items():
            flag = getattr(flag_type, flag, flag)
            if eval_flags:
                flag = eval(flag)
            attr[flag] = value
        return attr if not return_instance else instance

    if isinstance(instance, partial):
        partial_instance = instance
        config_key = key
        config_type = flag_type
        config_flags = flags

        def wrapper(*args, **kwargs):
            # First instantiate the partial
            instance = partial_instance(*args, **kwargs)
            # Then set the attributes
            return setattrs(instance, config_key, config_type, True, **config_flags)

        return wrapper
    else:
        return setattrs(instance, key, flag_type, False, **flags)


class MjCambrianFlagWrapperMeta(enum.EnumMeta):
    """This is a simple metaclass to allow for the use of the | operator to combine
    flags. This means you can simply put `flag1 | flag2` in the yaml file and it will
    be combined into a single flag.

    The following forms are supported and any combination thereof:
    - flag1 | flag2 | flag3 | ...
    - flag1|flag2|flag3|...
    - flag1
    """

    def __getitem__(cls, item):
        if isinstance(item, str) and "|" in item:
            from functools import reduce

            items = [cls.__getitem__(i.strip()) for i in item.split("|")]
            return reduce(lambda x, y: x | y, items)
        return super().__getitem__(item)
