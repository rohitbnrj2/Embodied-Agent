from typing import Any, Optional, Type, Callable, Concatenate, List, Dict, TYPE_CHECKING
from dataclasses import dataclass, fields, make_dataclass
import enum
from functools import partial
import argparse
import os
import re

from hydra.core.config_store import ConfigStore
import hydra_zen as zen
from omegaconf import OmegaConf, DictConfig

if TYPE_CHECKING:
    from cambrian.utils.config import MjCambrianBaseConfig


def run_hydra(
    main_fn: Optional[
        Callable[[Concatenate["MjCambrianBaseConfig", ...]], None]
    ] = lambda *_, **__: None,
    /,
    *,
    parser: argparse.ArgumentParser = argparse.ArgumentParser(),
    config_path: str = f"{os.getcwd()}/configs",
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

            ```python
            def main(config: MjCambrianConfig, *, verbose: int):
                print(config, verbose)

            parser = argparse.ArgumentParser()
            parser.add_argument("--verbose", type=int, default=0)

            run_hydra(main_fn=main, parser=parser)
            ```

    Keyword Args:
        parser (argparse.ArgumentParser): The parser to use for the hydra
            application. If None, a new parser will be created.
        config_path (str): The path to the config directory. This should be the
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
    from cambrian.utils.config.config import MjCambrianConfig  # noqa: F401

    # Add one default argument for the --hydra-help message
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

    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    @hydra_argparse_override
    def main(cfg: DictConfig, **kwargs):
        from cambrian.utils.config import MjCambrianBaseConfig

        if instantiate:
            cfg = MjCambrianBaseConfig.instantiate(cfg, **kwargs)
        return main_fn(cfg, **kwargs)

    main()


# ===========


def glob(key: str, flattened: bool, _root_: DictConfig) -> Dict:
    """This resolver will glob a key in the config. This is useful for finding all keys
    that match a pattern. This is particularly useful for finding all keys that match a
    pattern in a nested config. This is effectively select, but allows `*` to be used as
    a wildcard.

    This method works by finding all `*` in the key and then iterating over all
    subsequent keys that match the globbed pattern.

    NOTE: yaml files aren't necessarily built to support globbing (like xml), so
    this method is fairly slow and should be used sparingly.

    Args:
        key (str): The key to glob. This is a dotlist key, like `a.b.*`. Multiple
            globs can be used, like `a.*.c.*.d.*`. Globs in keys can be used, as
            well, such as `a.ab*.c`
        flatten (bool): If true, the output will be a dict of the leaf keys and
            the accumulated values if there are like leaf keys. If False, the
            output will be a nested dict. Defaults to False.
        _root_ (DictConfig): The root config.
    """
    # Early exit if no globs
    if "*" not in key and "|" not in key:
        return OmegaConf.select(_root_, key)

    def recursive_glob(config: Dict | Any, keys: List[str]) -> Dict:
        if not keys or not isinstance(config, dict):
            return config

        # Loop over all the keys and find each match with the passed key/pattern
        result = {}
        current_key = keys[0].replace("*", ".*")
        for sub_key, sub_value in config.items():
            if sub_value is None:  # Skip None values, probably optionals
                continue

            if match := re.fullmatch(current_key, sub_key):
                # If it's a match, we'll recursively glob the next key
                matched_key = match.group()
                result[matched_key] = recursive_glob(sub_value, keys[1:])
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
    config = OmegaConf.to_container(_root_)
    globbed = recursive_glob(config, key.split("."))

    # Return the flattened or nested dict
    return flatten(globbed) if flattened else globbed

def build_pattern(patterns: List[str]) -> str:
    """Build a glob pattern from the passed patterns.

    The underlying method for globbing (`MjCambrianConfig.glob`) uses a regex pattern
    which is parses the dot-separated keys independently.

    Example:
        >>> build_pattern(
        ...     "training_config.seed",
        ...     "env_config.animal_configs.*.eye_configs.*.resolution",
        ...     "env_config.animal_configs.*.eye_configs.*.fov",
        ... )
        '(training_config|env_config).(seed|animal_configs).*.eye_configs.*.(resolution|fov)'
    """
    depth_based_keys: List[List[str]] = []  # list of keys at depths in the patterns
    for pattern in patterns:
        # For each key in the pattern, add at the same depth as the other patterns
        for i, key in enumerate(pattern.split(".")):
            if i < len(depth_based_keys):
                if key not in depth_based_keys[i]:
                    depth_based_keys[i].extend([key])
            else:
                depth_based_keys.append([key])

    # Now build the pattern
    pattern = ""
    for keys in depth_based_keys:
        pattern += "(" + "|".join(keys) + ")."
    pattern = pattern[:-1]  # remove the last dot
    return pattern


# ===========


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
        ConfigStore().store(cls.__name__, hydrated_cls)

        return hydrated_cls

    if cls is None:
        return wrapper
    return wrapper(cls)


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
