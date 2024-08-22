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
    Tuple,
)
from dataclasses import field
from pathlib import Path
import enum
from contextlib import contextmanager

import hydra_zen as zen
from hydra.core.utils import setup_globals
from hydra.core.hydra_config import HydraConfig
from omegaconf import (
    OmegaConf,
    DictConfig,
    ListConfig,
    MissingMandatoryValue,
    Node,
    MISSING,
)
from omegaconf.errors import ConfigKeyError

from cambrian.utils import safe_eval
from cambrian.utils.config.utils import config_wrapper, glob

# =============================================================================
# Global stuff

setup_globals()

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
    _config_is_content: bool
    _instantiated: bool

    def __init__(
        self,
        content: DictConfig | ListConfig,
        /,
        config: Optional[DictConfig | ListConfig] = None,
        instantiated: bool = False,
    ):
        # Must use __dict__ to set the attributes since we're overriding the
        # __getattr__ method. If config is None, we'll just set it to the content.
        self.__dict__["_content"] = content
        self.__dict__["_config"] = config or content
        self.__dict__["_config_is_content"] = config is None
        self.__dict__["_instantiated"] = instantiated

    @property
    def content(self) -> DictConfig | ListConfig:
        """The instantiated config."""
        return self._content

    @classmethod
    def instantiate(
        cls,
        config: DictConfig | ListConfig,
        *,
        as_container: bool = False,
        resolve: bool = True,
        throw_on_missing: bool = True,
        is_struct: bool = True,
        is_readonly: bool = True,
        **kwargs,
    ) -> Self | DictConfig | ListConfig:
        """Instantiate the config using the structured config. Will check for missing
        keys and raise an error if any are missing."""
        # First instantiate the config (will replace _target_ with the actual class)
        # And then merge the structured config with the instantiated config to give it
        # validation.
        content: DictConfig | ListConfig = zen.instantiate(config, **kwargs)

        # This is redundant for most cases, but instantiated logic may return strings
        # which may have interpolations. We'll explicitly call resolve to resolve these
        # interpolations.
        if resolve:
            OmegaConf.resolve(content)

        # Check for missing values. Error message will only show the first missing key.
        if throw_on_missing and (keys := OmegaConf.missing_keys(content)):
            content._format_and_raise(
                key=next(iter(keys)),
                value=None,
                cause=MissingMandatoryValue("Missing mandatory value"),
            )

        # Disable the ability to set new keys.
        OmegaConf.set_struct(content, is_struct)
        OmegaConf.set_struct(config, is_struct)

        # Set the config to readonly if requested
        OmegaConf.set_readonly(content, is_readonly)
        OmegaConf.set_readonly(config, is_readonly)

        if as_container:
            return content
        else:
            config = config.copy()
            return MjCambrianContainerConfig(content, config=config, instantiated=True)

    @classmethod
    def compose(
        cls,
        config_dir: Path | str,
        config_name: str,
        *,
        overrides: List[str] = [],
        return_hydra_config: bool = False,
    ) -> (
        Self
        | DictConfig
        | ListConfig
        | Tuple[Self | DictConfig | ListConfig, DictConfig]
    ):
        """Compose a config using the Hydra compose API. This will return the config as
        a MjCambrianContainerConfig instance."""
        import hydra
        from hydra.core.global_hydra import GlobalHydra
        from hydra.core.hydra_config import HydraConfig

        if GlobalHydra().is_initialized():
            GlobalHydra().clear()

        with hydra.initialize_config_dir(str(config_dir), version_base=None):
            hydra_config = hydra.compose(
                config_name=config_name, overrides=overrides, return_hydra_config=True
            )
            HydraConfig.instance().set_config(hydra_config)
            del hydra_config.hydra

        config = cls.create(hydra_config)
        if return_hydra_config:
            return config, HydraConfig.get()
        return config

    @classmethod
    def load(
        cls, *args, instantiate: bool = True, **instantiate_kwargs
    ) -> Self | DictConfig | ListConfig:
        """Wrapper around OmegaConf.load to instantiate the config."""
        loaded = OmegaConf.load(*args)
        if instantiate:
            return cls.instantiate(loaded, **instantiate_kwargs)
        else:
            return (
                MjCambrianContainerConfig(loaded)
                if OmegaConf.is_config(loaded)
                else loaded
            )

    @classmethod
    def load_pickle(
        cls, path: Path | str, *, overrides: List[str] = []
    ) -> Self | DictConfig | ListConfig:
        """Load a pickled config."""
        import cloudpickle

        with open(path, "rb") as f:
            cfg: Self = cloudpickle.load(f)
        with cfg.set_readonly_temporarily(False), cfg.set_struct_temporarily(False):
            cfg.merge_with_dotlist(overrides)
            cfg.resolve()

        return cfg

    @classmethod
    def create(
        cls, *args, instantiate: bool = True, **instantiate_kwargs
    ) -> Self | DictConfig | ListConfig:
        """Wrapper around OmegaConf.create to instantiate the config."""
        created = OmegaConf.create(*args)
        if instantiate:
            return cls.instantiate(created, **instantiate_kwargs)
        else:
            return (
                MjCambrianContainerConfig(created)
                if OmegaConf.is_config(created)
                else created
            )

    def resolve(self):
        """Wrapper around OmegaConf.resolve to resolve the config."""
        OmegaConf.resolve(self._content)
        if not self._config_is_content:
            OmegaConf.resolve(self._config)

    def merge_with(self, *others: DictConfig | ListConfig):
        """Wrapper around OmegaConf.merge to merge the config with another config."""
        OmegaConf.unsafe_merge(self._content, *others)
        if not self._config_is_content:
            OmegaConf.unsafe_merge(self._config, *others)

    def merge_with_dotlist(self, dotlist: List[str]):
        """Wrapper around DictConfig|ListConfig.merge_with_dotlist to merge the config
        with a dotlist."""
        self._content.merge_with_dotlist(dotlist)
        if not self._config_is_content:
            self._config.merge_with_dotlist(dotlist)

    def set_struct(self, is_struct: bool):
        """Wrapper around OmegaConf.set_struct to set the struct flag."""
        OmegaConf.set_struct(self._content, is_struct)
        if not self._config_is_content:
            OmegaConf.set_struct(self._config, is_struct)

    def set_readonly(self, is_readonly: bool):
        """Wrapper around OmegaConf.set_readonly to set the readonly flag."""
        OmegaConf.set_readonly(self._content, is_readonly)
        if not self._config_is_content:
            OmegaConf.set_readonly(self._config, is_readonly)

    @contextmanager
    def set_struct_temporarily(self, is_struct: bool):
        """Context manager to temporarily set the struct flag."""
        self.set_struct(is_struct)
        yield
        self.set_struct(not is_struct)

    @contextmanager
    def set_readonly_temporarily(self, is_readonly: bool):
        """Context manager to temporarily set the readonly flag."""
        self.set_readonly(is_readonly)
        yield
        self.set_readonly(not is_readonly)

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
        return OmegaConf.to_container(
            self._content if use_instantiated else self._config, **kwargs
        )

    def to_yaml(self, use_instantiated: bool = True, resolve: bool = True) -> str:
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

        def path_representer(dumper: yaml.Dumper, data: Path):
            return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))

        def flag_representer(dumper: yaml.Dumper, data: enum.Flag):
            data = "|".join([m.name for m in type(data) if m in data])
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        dumper = yaml.CDumper
        dumper.add_representer(str, str_representer)
        dumper.add_multi_representer(Path, path_representer)
        dumper.add_multi_representer(enum.Flag, flag_representer)
        return yaml.dump(
            self.to_container(use_instantiated=use_instantiated, resolve=resolve),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            Dumper=dumper,
        )

    def save(
        self,
        path: Path | str,
        *,
        header: str = None,
        use_instantiated: bool = False,
        resolve: bool = True,
        hydra_config: bool = False,
    ):
        """Saves the config to a yaml file."""
        if hydra_config:
            config = OmegaConf.to_yaml(HydraConfig.get(), resolve=True)
        else:
            config = self.to_yaml(use_instantiated=use_instantiated, resolve=resolve)

        with open(path, "w") as f:
            if header:
                f.write(f"{header}\n")
            f.write(config)

    def pickle(self, path: Path | str):
        """Pickle the config to a file."""
        import cloudpickle

        with open(path, "wb") as f:
            cloudpickle.dump(self, f)

    def glob(
        self, key: str, *, flatten: bool = False, assume_one: bool = False
    ) -> Dict[str, Any] | Any:
        """This is effectively select, but allows `*` to be used as a wildcard.

        This method works by finding all `*` in the key and then iterating over all
        subsequent keys that match the globbed pattern.

        NOTE: yaml files aren't necessarily built to support globbing (like xml), so
        this method is fairly slow and should be used sparingly.

        Args:
            key (str): The key to glob. This is a dotlist key, like `a.b.*`. Multiple
                globs can be used, like `a.*.c.*.d.*`. Globs in keys can be used, as
                well, such as `a.ab*.c`

        Keyword Args:
            flatten (bool): If true, the output will be a dict of the leaf keys and
                the accumulated values if there are like leaf keys. If False, the
                output will be a nested dict. Defaults to False.
            assume_one (bool): If True, will assume that there is only one match for
                each glob. If True, will return just the value of the match.
        """
        result = glob(key, flatten, self._config)
        if assume_one:
            assert len(result) == 1, f"Expected one match, got {len(result)}"
            return next(iter(result.values()))[0]
        return result

    def globbed_eval(
        self, src: str, *, key: Optional[str] = None, **patterns: str
    ) -> Any:
        """This method will evaluate a specific `src` given globbed patterns."""
        variables = {}
        for _key, pattern in patterns.items():
            variable = self.glob(pattern, flatten=True)
            variables[_key] = variable

        result: Dict[str, Any] = safe_eval(src, variables)
        if key is not None:
            assert key in result, f"Return key {key} not found in {list(result.keys())}"
            return result[key]
        return result

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
                instantiated = self.__dict__["_instantiated"]
                values.append(
                    MjCambrianContainerConfig(
                        value, config=config, instantiated=instantiated
                    )
                )
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
                instantiated = self.__dict__["_instantiated"]
                items.append(
                    (
                        key,
                        MjCambrianContainerConfig(
                            value, config=config, instantiated=instantiated
                        ),
                    )
                )
            else:
                items.append((key, value))
        return items

    def copy(self) -> Self:
        """Wrapper around the copy method to return a new instance of this class."""
        content = self._content.copy()
        config = self._config.copy() if not self._config_is_content else content
        instantiated = self.__dict__["_instantiated"]
        return MjCambrianContainerConfig(
            content, config=config, instantiated=instantiated
        )

    def clear(self):
        """Wrapper around the clear method to clear the content."""
        self._content.clear()
        if not self._config_is_content:
            self._config.clear()

    def update(self, *args, **kwargs):
        """Wrapper around the OmegaConf.update method to update the content."""
        OmegaConf.update(self._content, *args, **kwargs)
        if not self._config_is_content:
            OmegaConf.update(self._config, *args, **kwargs)

    def __getattr__(self, name: str) -> Self | Any:
        """Get the attribute from the content and return the wrapped instance. If the
        attribute is a DictConfig or ListConfig, we'll wrap it in this class.
        """
        return self._get_impl(name)
        content = getattr(self._content, name)
        if OmegaConf.is_config(content):
            config = self._config
            config = getattr(self._config, name)
            return MjCambrianContainerConfig(content, config=config)
        else:
            return content

    def __getitem__(self, key: Any) -> Self | Any:
        """Get the item from the content and return the wrapped instance. If the item is
        a DictConfig or ListConfig, we'll wrap it in this class."""
        return self._get_impl(key, is_getattr=False)
        content = self._content[key]
        if OmegaConf.is_config(content):
            config = self._config[key]
            return MjCambrianContainerConfig(content, config=config)
        else:
            return content

    def __setattr__(self, name: str, value: Any):
        """Set the attribute in the content. These are fairly slow, but we'll keep them
        using OmegaConf logic since it does validation."""
        if isinstance(value, MjCambrianContainerConfig):
            setattr(self._content, name, value._content)
            if not self._config_is_content:
                setattr(self._config, name, value._config)
        else:
            setattr(self._content, name, value)
            if not self._config_is_content:
                setattr(self._config, name, value)

    def __setitem__(self, key: Any, value: Any):
        """Set the item in the content. These are fairly slow, but we'll keep them using
        OmegaConf logic since it does validation."""
        if isinstance(value, MjCambrianContainerConfig):
            self._content[key] = value._content
            if not self._config_is_content:
                self._config[key] = value._config
        else:
            self._content[key] = value
            if not self._config_is_content:
                self._config[key] = value

    def __delattr__(self, name: str):
        """Delete the attribute from the content."""
        delattr(self._content, name)
        if not self._config_is_content:
            delattr(self._config, name)

    def __delitem__(self, key: Any):
        """Delete the item from the content."""
        del self._content[key]
        if not self._config_is_content:
            del self._config[key]

    def __iter__(self) -> Iterator[Any]:
        """Only supported by ListConfig. Wrapper around the __iter__ method to return
        the iterator of the content. Will convert any DictConfig or ListConfig to this
        class."""
        for content, config in zip(self._content, self._config):
            if OmegaConf.is_config(content):
                instantiated = self.__dict__["_instantiated"]
                yield MjCambrianContainerConfig(
                    content, config=config, instantiated=instantiated
                )
            else:
                yield content

    def __len__(self) -> int:
        """Only supported by ListConfig. Wrapper around the __len__ method to return the
        length of the content."""
        return len(self._content)

    def __getstate__(self) -> Dict[str, Any]:
        """Get the state of the object to pickle it. Will only pickle the config since
        it's not guaranteed the instances in content can be pickled."""
        instantiated = self.__dict__["_instantiated"]
        container = self.to_container(use_instantiated=False, resolve=instantiated)
        container["__state__"] = dict(instantiated=instantiated)
        return container

    def __setstate__(self, state: Dict[str, Any]):
        """Set the state of the object from the pickled state."""
        __state__: Dict[str, Any] = state.pop("__state__")
        config = OmegaConf.create(state)
        if instantiated := __state__.get("instantiated"):
            content = self.instantiate(config, as_container=True)
        else:
            content = config
        self.__init__(content, config=config, instantiated=instantiated)

    def __str__(self) -> str:
        return self.to_yaml(use_instantiated=False)

    # ===========
    # Internal utils

    def _get_impl(
        self,
        key: Any,
        default_value: Any = ...,
        *,
        is_getattr: bool = True,
        as_node: bool = False,
    ):
        """Perform access of the underlying data structures. This is an optimized
        version which uses internal methods of OmegaConf. This is similar to
        DictConfig._get_impl, but optimized for this use case. We know it out configs
        won't be structured, so there are minor optimizations we can do.

        Args:
            key (Any): The key to access.
            default_value (Any): The default value to return if the key is not found.

        Keyword Args:
            is_getattr (bool): Whether the access is an attribute access. If False, will
                treat the key as an item access.
            as_node (bool): Whether to return the node or the value. If True, will
                return the node.
        """
        content = self._content.__dict__["_content"]
        config = self._config.__dict__["_content"]
        if not OmegaConf.is_config(config):
            config = self._config

        if isinstance(self._content, DictConfig):
            # Normalize the key. This will convert the key to allowed dictionary keys,
            # like converts 0, 1 if the key type is bool. Basically validates the key.
            key = self._content._validate_and_normalize_key(key)

            # Check that the key exists
            if key not in content and (not is_getattr or default_value is ...):
                self._content._format_and_raise(
                    key, None, ConfigKeyError(f"Key not found: {key!s}")
                )
        else:
            # ListConfig only accepts integers as keys
            if is_getattr:
                key = self._validate_key_is_int(key)

        config: DictConfig | ListConfig
        content: DictConfig | ListConfig
        is_config: bool
        if is_getattr:
            default_value = None if default_value is ... else default_value
            content = content.get(key, default_value)
            is_config = OmegaConf.is_config(content)
            if is_config and not self._config_is_content:
                config = config.get(key, default_value)
        else:
            content = content[key]
            is_config = OmegaConf.is_config(content)
            if is_config and not self._config_is_content:
                config = config[key]

        # If the content is a config, we'll wrap it in this class
        if is_config and not as_node:
            instantiated = self.__dict__["_instantiated"]
            return MjCambrianContainerConfig(
                content, config=config, instantiated=instantiated
            )
        else:
            get_value = isinstance(content, Node)
            assert not get_value or content is not None, f"Content is None: {key}"
            return content._value() if get_value else content

    def _validate_key_is_int(self, key: Any) -> int:
        """Validate the key for the content."""
        try:
            return int(key)
        except ValueError:
            self._content._format_and_raise(
                key,
                None,
                AttributeError("ListConfig does not support attribute access"),
            )


class MjCambrianDictConfig(MjCambrianContainerConfig, DictConfig):
    """This is a wrapper around the OmegaConf DictConfig class.

    It is intended that this class never actually be instantiated. Config classes
    should inherit from this class (or a base config class should inherit from this)
    such that when duck typing, all the methods of DictConfig and
    MjCambrianContainerConfig are available.
    """

    pass


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
