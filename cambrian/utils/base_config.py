from typing import Dict, Any, Optional, Self, Type
from dataclasses import field
from pathlib import Path
from functools import partial

from omegaconf import OmegaConf, DictConfig, ListConfig, MissingMandatoryValue
import hydra_zen as zen

class MjCambrianContainerConfig:
    _config: DictConfig
    _content: DictConfig

    def __init__(
        self,
        content: DictConfig | ListConfig,
        /,
        structured: Optional["MjCambrianBaseConfig"] = None,
        config: Optional[DictConfig | ListConfig] = None,
        **kwargs,
    ):
        self.__dict__["_config"] = config or content

        if structured:
            content = self.instantiate(content, structured=structured, **kwargs)
        self.__dict__["_content"] = content

    def instantiate(
        self,
        config: DictConfig | ListConfig | Self,
        structured: Type,
        **kwargs,
    ) -> Self:
        config = OmegaConf.merge(structured, zen.instantiate(config, **kwargs))
        if keys := OmegaConf.missing_keys(config):
            config._format_and_raise(
                key=next(iter(keys)),
                value=None,
                cause=MissingMandatoryValue("Missing mandatory value"),
            )
        return config

    @classmethod
    def load(cls, path: Path | str) -> Self:
        return cls.instantiate(OmegaConf.load(path))

    @classmethod
    def load_from_string(cls, string: str) -> Self:
        import tempfile
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(string)
            path = Path(f.name)
        return cls.load(path)

    def __getattr__(self, name: str):
        content = self._content.__getattr__(name)
        config = self._config.__getattr__(name)
        return MjCambrianContainerConfig(content, config=config)

    def get_type(self):
        return OmegaConf.get_type(self._content)

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
            hydrated_cls = zen.hydrated_dataclass(cls, populate_full_signature=True, **dataclass_kwargs)(cls)

        # Add to the hydra store
        # By adding it to the zen store rather than the hydra store directly, we can
        # support partial types (as in types that are not allowed by OmegaConf/hydra).
        # For instance, if we want to type hint a class or function, this would not be
        # allowed by OmegaConf/hydra. But by adding it to the zen store, we can support
        # these types.
        if (None, cls.__name__) not in zen.store:
            zen.store(
                cls,
                name=cls.__name__,
                populate_full_signature=True,
                zen_dataclass=dataclass_kwargs,
                builds_bases=(hydrated_cls,),
            )

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

    custom: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def instantiate(cls, dict_config: DictConfig, **kwargs) -> Self:
        return MjCambrianContainerConfig(dict_config, structured=cls)
