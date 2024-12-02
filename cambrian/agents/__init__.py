"""This module defines the Cambrian agents."""

from cambrian.agents.agent import (
    MjCambrianAgent,
    MjCambrianAgent2D,
    MjCambrianAgentConfig,
)
from cambrian.agents.object import MjCambrianAgentObject
from cambrian.agents.point import (
    MjCambrianAgentPoint,
    MjCambrianAgentPointMazeOptimal,
    MjCambrianAgentPointMazeRandom,
    MjCambrianAgentPointPrey,
)

__all__ = [
    "MjCambrianAgentConfig",
    "MjCambrianAgent",
    "MjCambrianAgent2D",
    "MjCambrianAgentPoint",
    "MjCambrianAgentPointPrey",
    "MjCambrianAgentPointMazeOptimal",
    "MjCambrianAgentPointMazeRandom",
    "MjCambrianAgentObject",
]
