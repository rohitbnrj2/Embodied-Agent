from pathlib import Path

from cambrian.ml.trainer import MjCambrianTrainerConfig
from cambrian.envs.env import MjCambrianEnvConfig
from cambrian.utils.config import MjCambrianBaseConfig
from cambrian.utils.config.utils import config_wrapper, run_hydra

# ==================


@config_wrapper
class MjCambrianConfig(MjCambrianBaseConfig):
    """The base config for the mujoco cambrian environment. Used for type hinting.

    Attributes:
        logdir (Path): The primary directory which simulation data is stored in. This is 
            the highest level directory used for the experiment. `expdir` is the
            subdirectory used for a specific experiment. If overridding, it's 
            recommended to just override the logdir and not the expdir.
        expdir (Path): The subdirectory used for a specific experiment. This is the
            directory where the experiment's data is stored. 

        expname (str): The name of the experiment. Used to name the logging
            subdirectory. 

        seed (int): The base seed used when initializing the default thread/process.
            Launched processes should use this seed value to calculate their own seed
            values. This is used to ensure that each process has a unique seed.

        training (MjCambrianTrainingConfig): The config for the training process.
        env (MjCambrianEnvConfig): The config for the environment.
        eval_env (MjCambrianEnvConfig): The config for the evaluation environment.
    """

    logdir: Path
    expdir: Path
    expname: str

    seed: int

    trainer: MjCambrianTrainerConfig
    env: MjCambrianEnvConfig
    eval_env: MjCambrianEnvConfig


# =============


if __name__ == "__main__":

    def main(config: MjCambrianConfig):
        pass

    run_hydra(main)
