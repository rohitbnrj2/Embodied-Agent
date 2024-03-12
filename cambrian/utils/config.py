from typing import Dict, Any, Optional, Callable
import os

import hydra_zen as zen

from cambrian.ml.trainer import MjCambrianTrainerConfig
from cambrian.envs.env import MjCambrianEnvConfig
from cambrian.ml.evo import MjCambrianEvoConfig
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig


@config_wrapper
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

    trainer: MjCambrianTrainerConfig
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

    main()


if __name__ == "__main__":
    import time

    t0 = time.time()

    def main(config: MjCambrianConfig):
        import time
        t0 = time.time()
        for _ in range(100000):
            config.env
        t1 = time.time()
        print(f"Time: {t1 - t0:.2f}")
        # config.save("config.yaml")

    setup_hydra(main)
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s")
