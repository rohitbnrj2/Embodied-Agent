from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from cambrian.envs.env import MjCambrianEnvConfig
from cambrian.ml.evo import MjCambrianEvoConfig
from cambrian.utils.config import MjCambrianContainerConfig, config_wrapper, run_hydra

if TYPE_CHECKING:
    from cambrian.ml.trainer import MjCambrianTrainerConfig

# ==================


@config_wrapper
class MjCambrianConfig(MjCambrianContainerConfig):
    """The base config for the mujoco cambrian environment. Used for type hinting.

    Attributes:
        logdir (Path): The primary directory which simulation data is stored in. This is
            the highest level directory used for the experiment. `expdir` is the
            subdirectory used for a specific experiment.
        expdir (Path): The directory used for a specific experiment. This is the
            directory where the experiment's data is stored. Should evaluate to
            `logdir / `expsubdir`
        expsubdir (Path): The subdirectory relative to logdir where the experiment's
            data is stored. This is the directory where the experiment's data is stored.
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
    expsubdir: Path
    expname: Any

    seed: int

    trainer: "MjCambrianTrainerConfig"
    evo: Optional[MjCambrianEvoConfig] = None
    env: MjCambrianEnvConfig
    eval_env: MjCambrianEnvConfig | Any


# =============


if __name__ == "__main__":

    def main(config: MjCambrianConfig):
        pass

    run_hydra(main)
