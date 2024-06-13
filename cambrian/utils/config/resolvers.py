from typing import Any, Dict, List, Optional
from pathlib import Path

from omegaconf import OmegaConf, DictConfig, Node
from omegaconf.errors import ConfigKeyError

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

    NOTE: This technically uses hidden attributes (i.e. _parent).

    Args:
        key (str | None): The key to search for. Could be none (like when mode is
            "parent_key").
        mode (Optional[str]): The mode to use. Defaults to "value". Available modes:
            - "value": Will return the value of the found key. Key must be set.
            - "parent_key": Will return the parent's key. If key is None, won't do
                any recursion and will return the parent's key.
            - "path": Will return the path to the key.
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


@register_new_resolver("clear")
def clear_resolver(key: str | None = None, /, *, _node_: Node) -> Dict | List:
    if _node_ is None:
        # Parent will be None if we're at the top level
        raise ConfigKeyError(f"Key {key} not found in parent chain.")

    key = key or _node_._key()
    if key is not None and _node_._key() == key:
        return {} if isinstance(_node_._parent, DictConfig) else []
    else:
        # Otherwise, we'll keep searching up the parent chain
        return clear_resolver(key, _node_=_node_._parent)


@register_new_resolver("eval")
def safe_eval(key: str, /, *, _root_: DictConfig) -> Any:
    from cambrian.utils import safe_eval

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
def glob(key: str, flattened: bool = False, /, *, _root_: DictConfig) -> List[str]:
    from cambrian.utils.config.utils import glob

    try:
        globbed: Dict[str, Any] = glob(key, flattened, _root_)
        return next(iter(globbed.values()))
    except Exception as e:
        _root_._format_and_raise(
            key=key,
            value=key,
            msg=f"Error evaluating expression '{key}': {e}",
            cause=e,
        )


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

@register_new_resolver("load")
def load_resolver(path: str) -> Any:
    return OmegaConf.load(path)


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
def instantiate_resolver(target: str, /, *args, _root_: DictConfig) -> Any:
    from hydra.utils import instantiate

    try:
        return instantiate(target_resolver(target, *args))
    except Exception as e:
        _root_._format_and_raise(
            key=target,
            value=target,
            msg=f"Error instantiating target '{target}': {e}",
            cause=e,
        )


@register_new_resolver("pattern")
def pattern_resolver(*pattern: str) -> str:
    from cambrian.utils.config.utils import build_pattern

    return build_pattern(*pattern)


@register_new_resolver("custom")
def custom_resolver(target: str, default: Optional[Any] = None, /):
    return f"${{oc.select:${{search:custom,'path'}}.{target}, {default}}}"