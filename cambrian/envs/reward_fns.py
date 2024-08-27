from typing import Dict, Any, List, Optional

import numpy as np
import mujoco as mj

from cambrian.envs import MjCambrianEnv
from cambrian.agents import MjCambrianAgent

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
) -> float:
    """Terminated indicates that the episode was ended early in a success.
    Returns termination_reward if terminated, else reward."""
    if for_agents is not None and agent.name not in for_agents:
        return 0.0
    return reward if terminated else 0.0


def reward_for_truncation(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    for_agents: Optional[List[str]] = None,
) -> float:
    """Truncated indicates that the episode was ended early in a failure.
    Returns truncation_reward if truncated, else reward."""
    if for_agents is not None and agent.name not in for_agents:
        return 0.0
    return reward if truncated else 0.0


def euclidean_delta_from_init(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    factor: float = 1.0,
    only_best: bool = False,
) -> float:
    """
    Rewards the change in distance over the previous step scaled by the timestep.

    `only_best` will only reward the agent has moved further from the initial position
    than any previous position. This requires us to keep around a state of the best
    position.
    """
    if only_best:
        # Only reward if the current position is further from the initial position than
        # any previous position. This requires us to keep around a state of the best
        # position. If the current position is closer, we'll skip this agent.
        closest_pos = info.setdefault("best_pos", agent.init_pos.copy())
        if check_if_larger(agent.pos, closest_pos, agent.init_pos):
            info["best_pos"] = agent.pos.copy()
        else:
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
    agents: Optional[List[str]] = None,
    for_agents: Optional[List[str]] = None,
    only_best: bool = False,
    min_delta_threshold: Optional[float] = None,
    max_delta_threshold: Optional[float] = None,
):
    """
    Rewards the change in distance to any enabled agent over the previous step.
    Convention is that a positive reward indicates getting closer to the agent.

    `only_best` will only reward the agent if it is closer to the agent than any
    previous position. This requires us to keep around a state of the best position.
    """
    # Early exit if the agent is not in the for_agents list
    if for_agents is not None and agent.name not in for_agents:
        return 0

    accumulated_reward = 0.0
    for other_agent in env.agents.values():
        if agents is not None and other_agent.name not in agents:
            continue

        if only_best:
            # Only reward if the current position is closer to the agent than any
            # previous position. This requires us to keep around a state of the best
            # position. If the current position is further, we'll skip this agent.
            closest_pos = info.setdefault(
                f"best_pos_{other_agent.name}", agent.pos.copy()
            )
            if check_if_larger(other_agent.pos, closest_pos, agent.pos):
                info[f"best_pos_{other_agent.name}"] = agent.pos.copy()
            else:
                continue

        # NOTE: calc_delta returns a positive value if the agent moves away from the
        # agent. We'll multiple by -1 to flip the convention.
        delta = -factor * calc_delta(agent, info, other_agent.pos) / env.extent
        if min_delta_threshold is not None and delta < min_delta_threshold:
            continue
        elif max_delta_threshold is not None and delta > max_delta_threshold:
            continue

        accumulated_reward = delta

    return accumulated_reward


def reward_if_agents_respawned(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    for_agents: Optional[List[str]] = None,
) -> float:
    """This reward function rewards the agent if it has been respawned."""
    # Early exit if the agent is not in the for_agents list
    if for_agents is not None and agent.name not in for_agents:
        return 0

    return reward if info.get("respawned", False) else 0.0


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

    return accumulated_reward


def penalize_if_has_contacts(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    penalty: float,
    for_agents: Optional[List[str]] = None,
) -> float:
    """Penalizes the agent if it has contacts with the ground."""
    # Early exit if the agent is not in the for_agents list
    if for_agents is not None and agent.name not in for_agents:
        return 0

    return penalty if info.get("has_contacts", False) else 0.0


def reward_if_agents_in_view(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward_in_view: float = 0.0,
    reward_not_in_view: float = 0.0,
    from_agents: Optional[List[str]] = None,
    to_agents: Optional[List[str]] = None,
    for_agents: Optional[List[str]] = None,
    hfov: float = 45,
    scale_by_distance: bool = False,
) -> float:
    """This reward function rewards the agent if it is in the view of other agents.

    Keyword Args:
        reward_in_view (float): The reward to give the agent if it is in view of
            another agent. Default is 0.
        reward_not_in_view (float): The reward to give the agent if it is not in view
            of another agent. Default is 0.
        from_agents (Optional[List[str]]): The names of the agents that the reward
            should be calculated from. If None, the reward will be calculated from all
            agents.
        to_agents (Optional[List[str]]): The names of the agents that the reward
            should be calculated to. If None, the reward will be calculated to all
            agents.
        for_agents (Optional[List[str]]): The names of the agents that the reward
            should be calculated for. If None, the reward will be calculated for all
            agents.
        hfov (float): The horizontal fov to check whether the to agent is within view
            of the from agent. Default is 45. This is in degrees.
        scale_by_distance (bool): Whether to scale the reward by the distance between
            the agents. Default is False.
    """
    # Early exit if the agent is not in the for_agents list
    if for_agents is not None and agent.name not in for_agents:
        return 0

    accumulated_reward = 0
    from_agents = from_agents or list(env.agents.keys())
    to_agents = to_agents or list(env.agents.keys())
    for from_agent in [env.agents[name] for name in from_agents]:
        for to_agent in [env.agents[name] for name in to_agents]:
            # Check if the to_agent is in view of the from_agent
            in_view = check_in_view(
                env.model,
                env.data,
                from_agent,
                to_agent.pos,
                to_agent.geom.id,
                hfov=hfov,
            )

            # Add the reward to the accumulated reward. It may be scaled by the distance
            # if scale_by_distance is True, but only if the agent is in view.
            if in_view:
                dist = np.linalg.norm(to_agent.pos - from_agent.pos)
                scale = 1 / max(dist, 1) if scale_by_distance else 1
                reward = reward_in_view * scale
            else:
                reward = reward_not_in_view
            accumulated_reward += reward

    return accumulated_reward


def reward_if_facing_agents(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward_facing: float = 0.0,
    reward_not_facing: float = 0.0,
    from_agents: Optional[List[str]] = None,
    to_agents: Optional[List[str]] = None,
    for_agents: Optional[List[str]] = None,
    angle_threshold: float = 45,
    scale_by_angle: bool = False,
) -> float:
    """This reward function rewards the agent for minimizing the angle between the
    agent's yaw and the yaw of the vector between itself and the agent.

    Keyword Args:
        reward_facing (float): The reward to give the agent if it is facing the agent.
            Default is 0.
        reward_not_facing (float): The reward to give the agent if it is not facing the
            agent. Default is 0.
        from_agents (Optional[List[str]]): The names of the agents that the reward
            should be calculated from. If None, the reward will be calculated from all
            agents.
        to_agents (Optional[List[str]]): The names of the agents that the reward
            should be calculated to. If None, the reward will be calculated to all
            agents.
        for_agents (Optional[List[str]]): The names of the agents that the reward
            should be calculated for. If None, the reward will be calculated for all
            agents.
    """
    # Early exit if the agent is not in the from_agents list
    if for_agents is not None and agent.name not in for_agents:
        return 0

    accumulated_reward = 0
    from_agents = from_agents or list(env.agents.keys())
    to_agents = to_agents or list(env.agents.keys())
    for from_agent in [env.agents[name] for name in from_agents]:
        for to_agent in [env.agents[name] for name in to_agents]:
            vec = to_agent.pos - from_agent.pos
            yaw = np.arctan2(agent.mat[1, 0], agent.mat[0, 0])
            relative_yaw = np.abs(np.arctan2(vec[1], vec[0]) - yaw)

            # Add the reward to the accumulated reward. If the relative yaw is within
            # the fov of the agent, reward the agent. Otherwise, penalize it.
            if relative_yaw < np.deg2rad(angle_threshold / 2):
                reward = reward_facing
            else:
                reward = reward_not_facing
            if scale_by_angle:
                reward *= 1 - relative_yaw / np.pi
            accumulated_reward += reward

    return accumulated_reward


def constant_reward(
    env: MjCambrianEnv,
    agent: MjCambrianAgent,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    *,
    reward: float,
    for_agents: Optional[List[str]] = None,
) -> float:
    """Returns a constant reward."""
    # Early exit if the agent is not in the for_agents list
    if for_agents is not None and agent.name not in for_agents:
        return 0
    return reward


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

    NOTE: returns delta position of current pos from the previous pos to the point
    (i.e. current - prev)
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


def check_in_view(
    model: mj.MjModel,
    data: mj.MjData,
    from_pos: np.ndarray,
    from_yaw: float,
    to_pos: np.ndarray,
    to_geomid: int,
    hfov: float = 45,
    geomgroup_mask: int = 0,
) -> bool:
    """Checks if the to_pos is in the field of view of the agent."""
    vec = to_pos - from_pos
    relative_yaw = np.arctan2(vec[1], vec[0]) - from_yaw - np.pi / 2
    relative_yaw = (relative_yaw + np.pi) % (2 * np.pi) - np.pi

    # Early exit if the to_pos isn't within view of the from_pos
    if np.abs(relative_yaw) > np.deg2rad(hfov) / 2:
        return False

    # Now we'll trace a ray between the two points to check if there are any obstacles
    geomid = np.zeros(1, np.int32)
    mj.mj_ray(
        model,
        data,
        from_pos,
        vec,
        geomgroup_mask,  # can use to mask out agents to avoid self-collision
        1,  # include static geometries
        -1,  # include all bodies
        geomid,
    )

    # If the ray hit the to geom, then the to_pos is in view
    return geomid[0] == to_geomid
