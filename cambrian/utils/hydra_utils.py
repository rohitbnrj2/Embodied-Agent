from typing import Callable, TYPE_CHECKING
from dataclasses import dataclass
import os

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore

if TYPE_CHECKING:
    from cambrian.utils.config import MjCambrianConfig

OmegaConf.register_new_resolver("eval", eval, replace=True)

def config_wrapper(
    protocol_cls=None,
    /,
    *args,
    repr=False,
    slots=True,
    eq=False,
    match_args=False,
    kw_only=True,
    **kwargs,
):
    """Wrapper around a protocol class which will dynamically create a dataclass from it
    and will call ConfigStore.instance().store(name=cls.__name__, node=cfg)"""

    def wrapper(protocol_cls):
        cls = dataclass(
            protocol_cls,
            *args,
            repr=repr,
            slots=slots,
            eq=eq,
            match_args=match_args,
            unsafe_hash=not eq,
            kw_only=kw_only,
            **kwargs,
        )
        ConfigStore.instance().store(name=cls.__name__, node=cls)
        return protocol_cls

    if protocol_cls is None:
        return wrapper
    return wrapper(protocol_cls)

def setup_hydra(main_fn: Callable[["MjCambrianConfig"], None], /):
    """This function is the main entry point for the hydra application.

    Args:
        main_fn (Callable[["MjCambrianConfig"], None]): The main function to be called
            after the hydra configuration is parsed.
    """

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
    @hydra.main(version_base=None, config_path=f"{os.getcwd()}/configs", config_name="base")
    def main(cfg: DictConfig):
        cfg = hydra.utils.instantiate(cfg, _convert_='object')
        print(OmegaConf.get_type(cfg), type(cfg))

        main_fn(cfg)

    main()