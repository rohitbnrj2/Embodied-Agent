import argparse
import enum
import re
from copy import deepcopy
from dataclasses import dataclass, field, fields, make_dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Concatenate, Dict, List, Optional, Self, Tuple

import hydra_zen as zen
import yaml
from hydra.core.config_store import ConfigStore
from hydra.core.utils import setup_globals
from hydra.utils import get_object
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf
from omegaconf.errors import ConfigKeyError

from cambrian.utils import is_integer, safe_eval

# =============================================================================
# Global stuff

setup_globals()

# =============================================================================
# Config classes and methods


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

    def wrapper(original_cls):
        # Preprocess the fields to convert the types to supported types
        # Only certain primitives are supported by hydra/OmegaConf, so we'll convert
        # these types to supported types using the _sanitized_type method from hydra_zen
        # We'll just include the fields that are defined in this class and not in a base
        # class.
        cls = dataclass(original_cls, **kwargs)

        new_fields = []
        for f in fields(cls):
            # Ignore non-initialized fields
            new_fields.append((f.name, zen.DefaultBuilds._sanitized_type(f.type), f))

        # Create the new dataclass with the sanitized types
        kwargs["bases"] = cls.__bases__
        hydrated_cls = make_dataclass(cls.__name__, new_fields, **kwargs)

        # Copy over custom methods from the original class
        for attr_name in dir(cls):
            if attr_name not in hydrated_cls.__dict__:
                attr = getattr(cls, attr_name)
                if callable(attr):  # Ensure it's a method
                    try:
                        setattr(hydrated_cls, attr_name, attr)
                    except TypeError:
                        pass

        # This is a fix for a bug in the underlying cloudpickle library which is used
        # by hydra/submitit (a hydra plugin) to pickle the configs. Since we're using
        # dataclasses, when pickled, their state doesn't propagate correctly to the new
        # process when it's unpickled. A fix is to define the dataclasses in separate
        # modules, but since we're using make_dataclass all in the same one, we have to
        # explicitly set the module of the class here.
        # See https://github.com/cloudpipe/cloudpickle/issues/386 for a related bug.
        # TODO submit bug report on cloudpickle. #386 is fixed, but _MISSED_TYPE is
        # still an issue.
        hydrated_cls.__module__ = cls.__module__

        # Add to the hydra store
        ConfigStore.instance().store(cls.__name__, hydrated_cls)

        return hydrated_cls

    if cls is None:
        return wrapper
    return wrapper(cls)


@config_wrapper
class MjCambrianContainerConfig:
    """Base dataclass which provides additional methods for working with configs.

    Attributes:
        config (Optional[Dict[str, Any]]): The original, uninstantiated config.
        custom (Optional[Dict[str, Any]]): Custom data to use. This is useful for
            code-specific logic (i.e. not in yaml files) where you want to store
            data that is not necessarily defined in the config. This data is
            temporary, as in it's not accessible from python.
    """

    config: Optional[DictConfig] = field(
        default=None,
        init=False,  # metadata={"omegaconf_ignore": True},
    )
    custom: Optional[Dict[str, Any]] = field(default_factory=dict, init=False)

    @classmethod
    def instantiate(
        cls,
        config: DictConfig | ListConfig,
        **kwargs,
    ) -> Self:
        instance: Self = zen.instantiate(config, _convert_="object", **kwargs)
        OmegaConf.resolve(config)

        # Iteratively set the config attribute for all nested configs
        def set_config_attr(obj: Any, config: DictConfig | ListConfig):
            if isinstance(obj, MjCambrianContainerConfig):
                if obj.config is None:
                    obj.config = config
                if not OmegaConf.is_config(obj.config):
                    obj.config = OmegaConf.create(obj.config)
                for k, v in obj.config.items():
                    if hasattr(obj, k):
                        set_config_attr(getattr(obj, k), v)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if k in config:
                        set_config_attr(v, config[k])
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    set_config_attr(v, config[i])

        # After instantiation, we'll set the config attribute for all nested configs
        # `config` is ignored by omegaconf, so has to come after initialization
        set_config_attr(instance, config)

        return instance

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
        cls,
        *args,
        instantiate: bool = True,
        pattern: Optional[str] = None,
        **instantiate_kwargs,
    ) -> Self | DictConfig | ListConfig:
        """Wrapper around OmegaConf.load to instantiate the config.

        Keyword Args:
            pattern (Optional[str]): The specific pattern to select from the loaded
                config.
        """
        loaded = OmegaConf.load(*args)
        if pattern is not None:
            loaded = OmegaConf.select(loaded, pattern)
        if instantiate:
            return cls.instantiate(loaded, **instantiate_kwargs)
        else:
            return loaded

    @classmethod
    def create(
        cls, *args, instantiate: bool = True, **instantiate_kwargs
    ) -> Self | DictConfig | ListConfig:
        """Wrapper around OmegaConf.create to instantiate the config."""
        created = OmegaConf.create(*args)
        if instantiate:
            return cls.instantiate(created, **instantiate_kwargs)
        else:
            return created

    def merge_with(self, *others: DictConfig | ListConfig | Dict | List) -> Self:
        """Wrapper around OmegaConf.merge to merge the config with another config."""
        # Do an unsafe merge so types aren't checked
        merged = OmegaConf.unsafe_merge(self.config, *others)
        return self.instantiate(merged)

    def copy(self) -> Self:
        """Wrapper around the copy method to return a new instance of this class."""
        return deepcopy(self)

    def save(
        self,
        path: Path | str,
        *,
        header: str = None,
    ):
        """Saves the config to a yaml file."""
        with open(path, "w") as f:
            if header:
                f.write(f"{header}\n")
            f.write(self.to_yaml())

    def to_yaml(self) -> str:
        """Wrapper around OmegaConf.to_yaml to convert the config to a yaml string.
        Adds some custom representers."""
        assert self.config is not None, "Config is None, cannot convert to yaml."

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
        config = OmegaConf.to_container(self.config)
        return yaml.dump(
            config,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            Dumper=dumper,
        )

    def __getstate__(self) -> DictConfig:
        """This is used to pickle the object. We'll return the config as the state."""
        return self.config

    def __setstate__(self, state: DictConfig):
        """This is used to unpickle the object. We'll set the config from the state."""
        instance = self.instantiate(state)
        for field_name in self.__dataclass_fields__.keys():
            setattr(self, field_name, getattr(instance, field_name))

    def __str__(self) -> str:
        if self.config is None:
            return self.__repr__()
        return self.to_yaml()


# =============================================================================


def run_hydra(
    main_fn: Optional[
        Callable[[Concatenate["MjCambrianContainerConfig", ...]], None]
    ] = lambda *_, **__: None,
    /,
    *,
    parser: argparse.ArgumentParser = argparse.ArgumentParser(),
    config_path: Path | str = Path.cwd() / "configs",
    config_name: str = "base",
    instantiate: bool = True,
    **kwargs,
):
    """This function is the main entry point for the hydra application.

    The benefits of using this setup rather than the compose API is that we can
    use the sweeper and launcher APIs, which are not available in the compose API.

    Args:
        main_fn (Callable[[Concatenate[[MjCambrianConfig], ...], None]): The main
            function to be called after the hydra configuration is parsed. It should
            take the config as an argument and kwargs which correspond to the argument
            parser returns. We don't return the config directly because hydra allows
            multi-run sweeps and it doesn't make sense to return multiple configs in
            this case.

            Example:

            .. code-block:: python

                def main(config: MjCambrianConfig, *, verbose: int):
                    pass

                parser = argparse.ArgumentParser()
                parser.add_argument("--verbose", type=int, default=0)

                run_hydra(main_fn=main, parser=parser)

    Keyword Args:
        parser (argparse.ArgumentParser): The parser to use for the hydra
            application. If None, a new parser will be created.
        config_path (Path | str): The path to the config directory. This should be the
            absolute path to the directory containing the config files. By default,
            this is set to the current working directory.
        config_name (str): The name of the config file to use. This should be the
            name of the file without the extension. By default, this is set to
            "base".
        instantiate (bool): Whether to instantiate the config. If False, create
            will be used.
        kwargs: Additional keyword arguments to pass to the instantiate function.
    """
    import hydra
    from omegaconf import DictConfig

    # Import MjCambrianConfig here to have it register with the hydra store
    from cambrian.main import MjCambrianConfig  # noqa: F401

    # Add one default argument for the --hydra-help message
    parser.add_argument(
        "--no-instantiate",
        action="store_false",
        dest="instantiate",
        help="Don't instantiate the config.",
    )
    parser.add_argument(
        "--hydra-help", action="store_true", help="Print the hydra help message."
    )

    def hydra_argparse_override(fn: Callable, /):
        """This function allows us to add custom argparse parameters prior to hydra
        parsing the config.

        We want to set some defaults for the hydra config here. This is a workaround
        in a way such that we don't

        Note:
            Augmented from hydra discussion #2598.
        """
        import sys
        from functools import partial

        parsed_args, unparsed_args = parser.parse_known_args()

        # Move --hydra-help to unparsed_args if it's present
        # Hydra has a weird bug (I think) that doesn't allow overrides when
        # --hydra-help is passed, so remove all unparsed arguments if --hydra-help
        # is passed.
        if parsed_args.hydra_help:
            unparsed_args = ["--hydra-help"]
        del parsed_args.hydra_help

        # By default, argparse uses sys.argv[1:] to search for arguments, so update
        # sys.argv[1:] with the unparsed arguments for hydra to parse (which uses
        # argparse).
        sys.argv[1:] = unparsed_args

        return partial(fn, **vars(parsed_args))

    @hydra.main(
        version_base=None, config_path=str(config_path), config_name=config_name
    )
    @hydra_argparse_override
    def main(cfg: DictConfig, instantiate: bool = instantiate, **fn_kwargs):
        from cambrian.utils.config import MjCambrianContainerConfig

        if instantiate:
            cfg = MjCambrianContainerConfig.instantiate(cfg, **kwargs)

        return main_fn(cfg, **fn_kwargs)

    main()


# =============================================================================


def glob(key: str, flattened: bool, _root_: DictConfig) -> Dict:
    """This resolver will glob a key in the config. This is useful for finding all keys
    that match a pattern. This is particularly useful for finding all keys that match a
    pattern in a nested config. This is effectively select, but allows `*` to be used as
    a wildcard.

    This method works by finding all `*` in the key and then iterating over all
    subsequent keys that match the globbed pattern.

    Note:
        yaml files aren't necessarily built to support globbing (like xml), so
        this method is fairly slow and should be used sparingly.

    Note:
        List indexing is limited in support. To index an element in a list, you
        must use bracket notation, so `a[0].b` is supported, but `a.0.b` is not.

    Args:
        key (str): The key to glob. This is a dotlist key, like `a.b.*`. Multiple
            globs can be used, like `a.*.c.*.d.*`. Globs in keys can be used, as
            well, such as `a.ab*.c`
        flatten (bool): If true, the output will be a dict of the leaf keys and
            the accumulated values if there are like leaf keys. If False, the
            output will be a nested dict. Defaults to False.
        _root_ (DictConfig): The root config.
    """

    def recursive_glob(config: DictConfig | Any, keys: List[str]) -> Dict:
        if not keys or not isinstance(config, DictConfig):
            return config

        # We'll loop over all the keys and find each match with the passed key/pattern
        result = {}
        current_key = keys[0].replace("*", ".*")
        for sub_key, sub_value in config.items():
            if sub_value is None:  # Skip None values, probably optionals
                continue

            if match := re.fullmatch(current_key, sub_key):
                # If it's a match, we'll recursively glob the next key
                matched_key = match.group()
                result[matched_key] = recursive_glob(sub_value, keys[1:])

        # This adds support for direct indexing. This is currently the only supported
        # way to do list accessing for globbing. To check, we'll clean the parentheses
        # and see if the key exists in the config as is.
        # NOTE: this is done after the recursive globbing in case the the key is found
        # earlier
        for cleaned_key in re.sub(r"^\((.*)\)$", r"\1", current_key).split("|"):
            if cleaned_key in result:
                continue

            if sub_value := OmegaConf.select(config, cleaned_key):
                # remove the brackets from the key
                cleaned_key = re.sub(r"^\((.*)\)$", r"\1", cleaned_key)
                result[cleaned_key] = recursive_glob(sub_value, keys[1:])

        return result

    def flatten(
        data: Dict[str, Any], values: Dict[str, List[Any]] = {}
    ) -> Dict[str, Any]:
        """This will flatten the nested dict to a flat dict where each key is a leaf
        key of the nested dict and the value is a list of all the values that were
        accumulated to that leaf key."""
        for k, v in data.items():
            if isinstance(v, dict):
                flatten(v, values)
            else:
                values.setdefault(k, [])
                values[k].append(v)
        return values

    # Glob the key(s)
    globbed = recursive_glob(_root_, key.split("."))

    # Return the flattened or nested dict
    return flatten(globbed) if flattened else globbed


def merge_with_kwargs(
    config: DictConfig,
    *,
    instantiate: bool = True,
    **kwargs,
) -> DictConfig:
    """This method will merge the kwargs into the config. This is useful for merging
    "late", as in after the config has been resolved (not instantiated). By specifying
    the merge to happen at instantiation time rather than at resolution time, it gives
    more freedom in defining overrides within the config. See `base.yaml` for more
    info.

    This is intended to be called from a yaml config file like:

    .. code-block:: yaml

        config_to_merge_late:
            _target_: <path_to>.merge_with_kwargs
            _recursive_: False
            config: ${...} # this is what the kwargs are merged into
            kwarg1: value1
            kwarg2: value2
            ...

    Note:
        You may want _recursive_=False (as above) to avoid instantiating the config
        before merging the kwargs. If you want to override a config attribute in the
        config object which is instantiated (i.e. is a partial), you won't have access
        to the config attribute (only the partial object), so you would want
        _recursive_=False. Simpler cases can just use _recursive_=True.

    Args:
        config (DictConfig): The config to merge the kwargs into.

    Keyword Args:
        kwargs: The kwargs to merge into the config.
    """
    config = OmegaConf.unsafe_merge(config, kwargs)

    if instantiate:
        return MjCambrianContainerConfig.instantiate(config)
    return config


# =============================================================================


def instance_wrapper(
    *,
    instance: Any,
    key: Optional[str] = None,
    locate: bool = False,
    eval: bool = False,
    setitem: bool = False,
    **kwargs,
):
    """Wraps a class instance to allow setting class attributes after initialization.

    This utility is useful when not all attributes are available during class
    instantiation, allowing attributes to be set post-construction using either
    direct assignment, item setting, or attribute modification based on optional flags.

    Args:
        instance (Any): The class instance to wrap.
        key (Optional[str], optional): If provided, fetches the specified attribute
            from the instance to modify. Defaults to None.
        locate (bool, optional): If True, attempts to resolve attribute names
            dynamically (e.g., via object lookup). Defaults to False.
        eval (bool, optional): If True, evaluates attribute values using safe_eval
            before assignment. Defaults to False.
        setitem (bool, optional): If True, uses item assignment (e.g., `instance[key]`)
            instead of `setattr`. Defaults to False.
        **kwargs: Key-value pairs of attributes to set on the instance.

    Returns:
        Any: The modified instance.

    Raises:
        ValueError: If there is an error while setting an attribute.

    Example Usage (via YAML):
        .. code-block:: yaml

            obj_to_instantiate:
                _target_: <path_to>.instance_wrapper
                instance:
                    _target_: <class>
                    _args_: [arg1, arg2]
                    init_arg1: value1
                    init_arg2: value2
                set_arg1: value1
                set_arg2: value2

        For partial instantiation:
        .. code-block:: yaml

            partial_obj_to_instantiate:
                _target_: <path_to>.instance_wrapper
                instance:
                    _target_: <class>
                    _partial_: True
                    _args_: [arg1, arg2]
                    init_arg3: '???' # Set later
                set_arg1: value1
                set_arg2: value2
    """

    def setattrs(instance, **kwargs):
        try:
            for key, value in kwargs.items():
                if callable(value):
                    value = value()

                if locate:
                    key = get_object(key)
                if eval:
                    key = safe_eval(key)

                if isinstance(value, dict):
                    setattrs(getattr(instance, key), **value)
                elif setitem:
                    instance[key] = value
                else:
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
            if key is not None:
                instance = getattr(partial_instance, key)

            instance = partial_instance(*args, **kwargs)
            return setattrs(instance, **config_kwargs)

        return wrapper
    else:
        if key is not None:
            instance = getattr(instance, key)
        return setattrs(instance, **kwargs)


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


# =============================================================================
# Resolvers


def register_new_resolver(name: str, replace: bool = True, **kwargs):
    def decorator(fn):
        OmegaConf.register_new_resolver(name, fn, replace=replace, **kwargs)
        return fn

    return decorator


@register_new_resolver("search")
def search_resolver(
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

    Note:
        This technically uses hidden attributes (i.e. _parent).

    Args:
        key (str | None): The key to search for. Could be none (like when mode is
            "parent_key").
        mode (Optional[str]): The mode to use. Defaults to "value". Available modes:
            - "value": Will return the value of the found key. Key must be set.
            - "parent_key": Will return the parent's key. If key is None, won't do
            any recursion and will return the parent's key.
            - "path": Will return the path to the key.
        depth (Optional[int]): The depth of the search. Used internally
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
            return search_resolver(
                key, mode=mode, depth=depth + 1, _parent_=_parent_._parent
            )
    elif mode == "parent_key":
        if key is None:
            # If the key is None, we'll return the parent's key
            assert _parent_._key() is not None, "Parent key is None."
            return _parent_._key()
        elif _parent_._key() == key:
            assert _parent_._parent._key() is not None, "Parent key is None."
            return _parent_._parent._key()

        if depth != 0 and isinstance(_parent_, DictConfig) and key in _parent_:
            # If we're at a key that's not the parent and the parent has the key we're
            # looking for, we'll return the parent
            return search_resolver(None, mode=mode, depth=depth + 1, _parent_=_parent_)
        else:
            # Otherwise, we'll keep searching up the parent chain
            return search_resolver(
                key, mode=mode, depth=depth + 1, _parent_=_parent_._parent
            )
    elif mode == "path":
        if key in _parent_:
            # If the key is in the parent, we'll return the path
            return _parent_._get_full_key(key)
        else:
            # Otherwise, we'll keep searching up the parent chain
            return search_resolver(
                key, mode=mode, depth=depth + 1, _parent_=_parent_._parent
            )


@register_new_resolver("parent")
def parent_resolver(key: str | None = None, *, _parent_: DictConfig) -> Any:
    return search_resolver(key, mode="parent_key", _parent_=_parent_)


@register_new_resolver("eval")
def eval_resolver(key: str, /, *, _root_: DictConfig) -> Any:
    try:
        return safe_eval(key)
    except Exception as e:
        _root_._format_and_raise(
            key=key,
            value=key,
            msg=f"Error evaluating expression '{key}': {e}",
            cause=e,
        )


@register_new_resolver("glob")
def glob_resolver(
    pattern: str,
    config: Optional[DictConfig | ListConfig | str] = None,
    /,
    *,
    _root_: DictConfig,
) -> ListConfig | DictConfig:
    if config is None:
        config = _root_

    if isinstance(config, str):
        config = OmegaConf.select(_root_, config)
    if isinstance(config, DictConfig):
        return {k: v for k, v in config.items() if re.match(pattern, k)}
    if isinstance(config, ListConfig):
        return [v for v in config if re.match(pattern, v)]


@register_new_resolver("hydra_select")
def hydra_select(
    key: str, default: Optional[Any] = None, /, *, _root_: DictConfig
) -> Any | None:
    """This is similar to the regular hydra resolver, but this won't through an error
    if the global hydra config is unset. Instead, it will return another interpolation
    using dotpath notation directly. As in, ${hydra_select:runtime.choices.test}, if
    HydraConfig is unset, will return ${hydra.runtime.choices.test}."""
    from hydra.core.hydra_config import HydraConfig

    try:
        return OmegaConf.select(HydraConfig.get(), key, default=default)
    except ValueError:
        return OmegaConf.select(
            _root_, f"hydra.{key}", default=default, throw_on_missing=True
        )


@register_new_resolver("path")
def path_resolver(*parts: str) -> Path:
    return Path(*parts)


@register_new_resolver("read")
def read_resolver(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


@register_new_resolver("custom")
def custom_resolver(target: str, default: Optional[Any] = None, /):
    return f"${{oc.select:${{search:custom,'path'}}.{target}, {default}}}"


@register_new_resolver("float_to_str")
def float_to_str_resolver(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "n")


@register_new_resolver("locate")
def locate_resolver(fn: str) -> Any:
    from hydra.utils import get_object

    return get_object(fn)


@register_new_resolver("getitem")
def get_resolver(obj: ListConfig, key: int) -> Any:
    return obj[key]


@register_new_resolver("target")
def target_resolver(target: str, /, *args) -> Dict[str, Any]:
    """This is a resolver which serves as a proxy for the _target_ attribute used
    in hydra. Basically `target` will be defined as `_target_` and the rest of the
    attributes will be passed as arguments to the target. You should always default to
    using `_target_` directly in your config, but because interpolations _may_ be
    resolved prior to or instead of instantiate, it may be desired to resolve
    interpolations before instantiations."""
    return {"_target_": target, "_args_": args}


@register_new_resolver("instantiate")
def instantiate_resolver(target: str | DictConfig, /, *args, _root_: DictConfig) -> Any:
    try:
        if isinstance(target, str):
            target = target_resolver(target, *args)
        return MjCambrianContainerConfig.instantiate(target)
    except Exception as e:
        _root_._format_and_raise(
            key=target,
            value=target,
            msg=f"Error instantiating target '{target}': {e}",
            cause=e,
        )


@register_new_resolver("clean_overrides")
def clean_overrides_resolver(
    overrides: List[str],
    ignore_after_override: Optional[str] = "",
    use_seed_as_subfolder: bool = True,
) -> str:
    cleaned_overrides: List[str] = []

    seed: Optional[Any] = None
    for override in overrides:
        if "=" not in override or override.count("=") > 1:
            continue
        if ignore_after_override and override == ignore_after_override:
            break

        key, value = override.split("=", 1)
        if key == "exp":
            continue
        if key == "seed" and use_seed_as_subfolder:
            seed = value
            continue

        # Special key cases that we want the second key rather than the first
        if (
            key.startswith("env.reward_fn")
            or key.startswith("env.truncation_fn")
            or key.startswith("env.termination_fn")
            or key.startswith("env.step_fn")
            or is_integer(key.split(".")[-1])
        ):
            key = "_".join(key.split(".")[-2:])
        else:
            key = key.split("/")[-1].split(".")[-1]

        # Clean the key and value
        key = (
            key.replace("+", "")
            .replace("[", "")
            .replace("]", "")
            .replace(",", "_")
            .replace(" ", "")
        )
        value = (
            value.replace(".", "p")
            .replace("-", "n")
            .replace("[", "")
            .replace("]", "")
            .replace(",", "_")
            .replace(" ", "")
        )

        cleaned_overrides.append(f"{key}_{value}")

    return "_".join(cleaned_overrides) + (f"/seed_{seed}" if seed is not None else "")
