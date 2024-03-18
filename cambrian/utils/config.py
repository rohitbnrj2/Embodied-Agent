from typing import Dict, Any, Optional, Callable, Concatenate
import argparse

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


# =============


def run_hydra(
    main_fn: Optional[
        Callable[[Concatenate[MjCambrianConfig, ...]], None]
    ] = lambda *_, **__: None,
    /,
    *,
    parser: Optional[argparse.ArgumentParser] = argparse.ArgumentParser(),
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
        parser (Optional[argparse.ArgumentParser]): The parser to use for the hydra
            application. If None, a new parser will be created.
    """
    import hydra
    from omegaconf import DictConfig
    import os

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

        # By default, argparse uses sys.argv[1:] to search for arguments, so update
        # sys.argv[1:] with the unparsed arguments for hydra to parse (which uses
        # argparse).
        sys.argv[1:] = unparsed_args

        return partial(fn, **vars(parsed_args))

    # Define the config path relative to the running script, which is assumed to be
    # at the root of the project.
    config_path = f"{os.getcwd()}/configs"
    config_name = "base"

    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    @hydra_argparse_override
    def main(cfg: DictConfig, **kwargs):
        config = MjCambrianConfig.instantiate(cfg)
        main_fn(config, **kwargs)

    main()


if __name__ == "__main__":

    def main(config: MjCambrianConfig):
        pass

    run_hydra(main)
