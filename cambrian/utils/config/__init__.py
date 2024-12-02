"""This module defines the Cambrian config utilities."""

from typing import TYPE_CHECKING

import cambrian.utils.config.resolvers  # noqa: F401
from cambrian.utils.config.base import MjCambrianBaseConfig, MjCambrianContainerConfig
from cambrian.utils.config.utils import (
    MjCambrianFlagWrapperMeta,
    build_pattern,
    config_wrapper,
    instance_flag_wrapper,
    instance_wrapper,
    run_hydra,
)

if TYPE_CHECKING:
    from cambrian.utils.config.config import MjCambrianConfig
else:

    def lazy_load_class(package, class_name):
        from importlib import import_module

        class_ = None

        def load():
            nonlocal class_
            if class_ is None:
                module = import_module(package)
                class_ = getattr(module, class_name)
            return class_

        class ClassProxyMeta(type):
            def __getattr__(cls, item):
                actual_class = load()
                return getattr(actual_class, item)

        class ClassProxy(metaclass=ClassProxyMeta):
            def __new__(cls, *args, **kwargs):
                actual_class = load()
                return actual_class(*args, **kwargs)

        return ClassProxy

    MjCambrianConfig = lazy_load_class(
        "cambrian.utils.config.config", "MjCambrianConfig"
    )

__all__ = [
    "MjCambrianBaseConfig",
    "MjCambrianContainerConfig",
    "config_wrapper",
    "instance_flag_wrapper",
    "instance_wrapper",
    "MjCambrianFlagWrapperMeta",
    "run_hydra",
    "MjCambrianConfig",
    "build_pattern",
]
