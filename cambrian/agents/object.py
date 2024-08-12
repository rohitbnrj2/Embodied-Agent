from typing import List, TYPE_CHECKING

from cambrian.agents import MjCambrianAgent2D, MjCambrianAgentConfig

if TYPE_CHECKING:
    from cambrian.envs import MjCambrianEnv


class MjCambrianAgentObject(MjCambrianAgent2D):
    """This is a class which defines an object agent. An object agent is
    essentially a non-trainable agent. It's simply an object in the environment which
    has no observations or actions.
    """

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
        idx: int,
    ):
        assert not config.trainable, "Object agents cannot be trainable"
        super().__init__(config, name, idx)

    def get_action_privileged(self, env: "MjCambrianEnv") -> List[float]:
        return []
