"""Done fns. These can be used either with termination or truncation."""

from typing import Any, Dict, Optional, List

import numpy as np

from cambrian.envs import MjCambrianEnv
from cambrian.agents import MjCambrianAgent

# ======================
# Generic Done Functions


def done_never(
    env: MjCambrianEnv, agent: MjCambrianAgent, info: Dict[str, Any]
) -> bool:
    """Never done."""
    return False


def done_if_exceeds_max_episode_steps(
    env: MjCambrianEnv, agent: MjCambrianAgent, info: Dict[str, Any]
) -> bool:
    """Done if episode step exceeds max episode steps."""
    return env.episode_step >= (env.max_episode_steps - 1)


def done_if_low_reward(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    info: Dict[str, Any],
    *,
    threshold: float,
) -> bool:
    """Done if agent has low reward."""
    return env.cumulative_reward < threshold


def done_if_has_contacts(
    env: MjCambrianEnv, agent: MjCambrianAgent, info: Dict[str, Any]
) -> bool:
    """Done if agent has contacts."""
    return info["has_contacts"]


def done_if_close_to_agents(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    info: Dict[str, Any],
    *,
    agents: Optional[List[str]] = None,
    for_agents: Optional[List[str]] = None,
    distance_threshold: float,
) -> bool:
    """Done if agent is close to another agent."""
    # Early exit if the agent is not in the for_agents list
    if for_agents is not None and agent.name not in for_agents:
        return False

    for other_agent in env.agents.values():
        if agents is not None and other_agent.name not in agents:
            continue
        if other_agent.name == agent.name:
            continue

        if np.linalg.norm(other_agent.pos - agent.pos) < distance_threshold:
            return True
    return False


def done_if_facing_agents(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    info: Dict[str, Any],
    *,
    from_agents: Optional[List[str]] = None,
    to_agents: Optional[List[str]] = None,
    for_agents: Optional[List[str]] = None,
    angle_threshold: float = 45,
    n_frames: int = 1,
) -> bool:
    """Terminates the episode if an agent is facing another agent.

    Keyword Args:
        from_agents (Optional[List[str]]): List of agent names to check for facing.
            If None, all agents are checked. Defaults to None.
        to_agents (Optional[List[str]]): List of object names to check for facing.
            If None, all agents are checked. Defaults to None.
        for_agents (Optional[List[str]]): List of agent names to check for facing.
            If None, all agents are checked. Defaults to None.
        angle_threshold (float): Angle threshold for facing. Defaults to 45.
        n_frames (int): Number of frames to check for facing. Defaults to 1.

    """
    if for_agents is not None and agent.name not in for_agents:
        return False

    from_agents = from_agents or list(env.agents.keys())
    to_agents = to_agents or list(env.agents.keys())
    for from_agent in [env.agents[name] for name in from_agents]:
        for to_agent in [env.agents[name] for name in to_agents]:
            vec = to_agent.pos - from_agent.pos
            yaw = np.arctan2(agent.mat[1, 0], agent.mat[0, 0])
            relative_yaw = np.abs(np.arctan2(vec[1], vec[0]) - yaw)
            if relative_yaw < np.deg2rad(angle_threshold / 2):
                key = f"facing_obj_{to_agent.name}_{from_agent.name}"
                info.setdefault(key, 0)
                info[key] += 1
                if info[key] > n_frames:
                    return True
    return False


def done_combined(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    info: Dict[str, Any],
    **done_fns,
) -> bool:
    """Combine multiple done functions."""
    return any(done_fn(env, agent, info) for done_fn in done_fns.values())
