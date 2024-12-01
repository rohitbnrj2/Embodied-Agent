"""Reward fns. These can be used to calculate rewards for agents."""

from typing import Any, Dict, List, Optional

import numpy as np

from cambrian.agents import MjCambrianAgent
from cambrian.envs import MjCambrianEnv
from cambrian.utils import agent_selected

# =====================
# Reward functions


def reward_for_termination(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    for_agents: Optional[List[str]] = None,
    scale_by_quickness: bool = False,
) -> float:
    """Terminated indicates that the episode was ended early in a success.
    Returns termination_reward if terminated, else reward."""
    if not agent_selected(agent, for_agents):
        return 0.0
    factor = calc_quickness(env) if scale_by_quickness else 1.0
    return reward * factor if terminated else 0.0


def reward_for_truncation(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    for_agents: Optional[List[str]] = None,
    scale_by_quickness: bool = False,
) -> float:
    """Truncated indicates that the episode was ended early in a failure.
    Returns truncation_reward if truncated, else reward."""
    if not agent_selected(agent, for_agents):
        return 0.0
    factor = calc_quickness(env) if scale_by_quickness else 1.0
    return reward * factor if truncated else 0.0


def euclidean_delta_from_init(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    factor: float = 1.0,
    for_agents: Optional[List[str]] = None,
) -> float:
    """Rewards the change in distance over the previous step scaled by the timestep."""

    if not agent_selected(agent, for_agents):
        return 0.0

    return calc_delta(agent, info, agent.init_pos) * factor


def reward_euclidean_delta_to_agents(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    factor: float,
    to_agents: Optional[List[str]] = None,
    for_agents: Optional[List[str]] = None,
    scale_by_quickness: bool = False,
):
    """
    Rewards the change in distance to any enabled agent over the previous step.
    Convention is that a positive reward indicates getting closer to the agent.
    """
    # Early exit if the agent is not in the for_agents list
    if not agent_selected(agent, for_agents):
        return 0.0

    accumulated_reward = 0.0
    for other_agent in env.agents.values():
        if not agent_selected(other_agent, to_agents):
            continue

        # NOTE: calc_delta returns a positive value if the agent moves away from the
        # agent. We'll multiple by -1 to flip the convention.
        delta = -factor * calc_delta(agent, info, other_agent.pos)
        accumulated_reward = delta

    factor = calc_quickness(env) if scale_by_quickness else 1.0
    return accumulated_reward * factor


def reward_if_agents_respawned(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    for_agents: Optional[List[str]] = None,
    scale_by_quickness: bool = False,
) -> float:
    """This reward function rewards the agent if it has been respawned."""
    # Early exit if the agent is not in the for_agents list
    if not agent_selected(agent, for_agents):
        return 0.0

    scale = calc_quickness(env) if scale_by_quickness else 1.0
    return reward * scale if info.get("respawned", False) else 0.0


def reward_if_close_to_agents(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    distance_threshold: float,
    for_agents: Optional[List[str]] = None,
    from_agents: Optional[List[str]] = None,
    to_agents: Optional[List[str]] = None,
    scale_by_quickness: bool = False,
) -> float:
    """This reward function rewards the agent if it is close to another agent.

    Keyword Args:
        reward (float): The reward to give the agent if it is close to another agent.
            Default is 0.
        distance_threshold (float): The distance threshold to check if the agent is
            close to another agent.
        for_agents (Optional[List[str]]): The names of the agents that the reward
            should be calculated for. If None, the reward will be calculated for all
            agents.
        from_agents (Optional[List[str]]): The names of the agents that the reward
            should be calculated from. If None, the reward will be calculated from all
            agents.
        to_agents (Optional[List[str]]): The names of the agents that the reward
            should be calculated to. If None, the reward will be calculated to all
            agents.
    """
    accumulated_reward = 0
    for agent_name, agent in env.agents.items():
        if for_agents is not None and agent_name not in for_agents:
            continue
        if from_agents is not None and agent_name not in from_agents:
            continue

        for other_agent_name, other_agent in env.agents.items():
            if to_agents is not None and other_agent_name not in to_agents:
                continue
            if agent_name == other_agent_name:
                continue

            if np.linalg.norm(agent.pos - other_agent.pos) < distance_threshold:
                accumulated_reward += reward

    factor = calc_quickness(env) if scale_by_quickness else 1.0
    return accumulated_reward * factor


def penalize_if_has_contacts(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    penalty: float,
    for_agents: Optional[List[str]] = None,
    scale_by_quickness: bool = False,
) -> float:
    """Penalizes the agent if it has contacts with the ground."""
    # Early exit if the agent is not in the for_agents list
    if for_agents is not None and agent.name not in for_agents:
        return 0

    factor = calc_quickness(env) if scale_by_quickness else 1.0
    return penalty * factor if info.get("has_contacts", False) else 0.0


def reward_combined(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    exclusive_fns: List[str] = [],
    **reward_fns,
) -> float:
    """Combines multiple reward functions into one.

    Keyword Args:
        exclusive_fns (Optional[List[str]]): If provided, only the reward functions
            with this name will be used if it's non-zero. As in, in order, the first
            function to return a non-zero reward will be returned.
    """
    accumulated_reward = 0
    for name, fn in reward_fns.items():
        reward = fn(env, agent, terminated, truncated, info)

        if name in exclusive_fns and reward != 0:
            return reward
        accumulated_reward += reward
    return accumulated_reward


# =====================
# Utility functions


def calc_delta(
    agent: MjCambrianAgent, info: Dict[str, Any], point: np.ndarray = np.array([0, 0])
) -> np.ndarray:
    """Calculates the delta position of the agent from a point.

    Returns:
        np.ndarray: The delta position of the agent from the point
            (i.e. current - prev).
    """

    current_distance = np.linalg.norm(agent.pos - point)
    prev_distance = np.linalg.norm(info["prev_pos"] - point)
    return current_distance - prev_distance


def check_if_larger(
    p1: np.ndarray, p2: np.ndarray, point: np.ndarray = np.array([0, 0])
) -> bool:
    """Checks if the distance from point to p1 is larger than the distance from point
    to p2."""
    return np.linalg.norm(p1 - point) > np.linalg.norm(p2 - point)


def calc_quickness(env: MjCambrianEnv) -> float:
    """Calculates the quickness of the agent."""
    return max(env.max_episode_steps - env.episode_step, 0.0) / env.max_episode_steps
