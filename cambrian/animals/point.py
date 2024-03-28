from typing import Dict, List, Any

import numpy as np
from gymnasium import spaces

from cambrian.animals.animal import MjCambrianAnimal, MjCambrianAnimalConfig
from cambrian.utils import setattrs_temporary


class MjCambrianPointAnimal(MjCambrianAnimal):
    """
    This is a hardcoded class which implements the animal as actuated by a forward
    velocity and a rotational position. In mujoco, to the best of my knowledge, all
    translational joints are actuated in reference to the _global_ frame rather than
    the local frame. This means a velocity actuator applied along the x-axis will move
    the agent along the global x-axis rather than the local x-axis. Therefore, the
    agent will have 3 actuators: two for x and y global velocities and one for
    rotational position. From the perspective the calling class (i.e. MjCambrianEnv),
    this animal has two actuators: a forward velocity and a rotational position. We will
    calculate the global velocities and rotational position from these two "actuators".

    TODO: Will create an issue on mujoco and see if it's possible to implement this
    in xml.

    NOTE: The action obs is still the global velocities and rotational position.
    """

    def _get_obs(self) -> Dict[str, Any]:
        """Creates the entire obs dict."""
        obs = super()._get_obs()

        # Update the action obs
        # Calculate the global velocities
        if self.config.use_action_obs:
            vx, vy, theta = self.last_action
            v, theta = np.hypot(vx, vy), np.arctan2(vy, vx) - self.qpos[2]
            obs["action"] = np.array([v, theta], dtype=np.float32)

        return obs

    def apply_action(self, action: List[float]):
        """This differs from the base implementation as action only has two elements,
        but the model has three actuators. Calculate the global velocities here."""
        assert len(action) == 2, f"Action must have two elements, got {len(action)}."

        # map the v action to be between 0 and 1
        v = (action[0] + 1) / 2

        # Calculate the global velocities
        theta = self._data.qpos[self._joint_qposadr + 2]
        new_action = [v * np.cos(theta), v * np.sin(theta), action[1]]

        # Call the base implementation with the new action
        super().apply_action(new_action)

    @property
    def observation_space(self) -> spaces.Space:
        """Overrides the base implementation so the action obs is only two elements."""
        observation_space = super().observation_space
        if "action" in observation_space.spaces:
            observation_space["action"] = spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            )
        return observation_space

    @property
    def action_space(self) -> spaces.Space:
        """Overrides the base implementation to only have two elements."""
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
