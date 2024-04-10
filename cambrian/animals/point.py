from typing import Dict, List, Any, TYPE_CHECKING

import numpy as np
from gymnasium import spaces

from cambrian.animals.animal import MjCambrianAnimal, MjCambrianAnimalConfig

if TYPE_CHECKING:
    from cambrian.envs.env import MjCambrianEnv


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

    def _update_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Creates the entire obs dict."""
        obs = super()._update_obs(obs)

        # Update the action obs
        # Calculate the global velocities
        if self.config.use_action_obs:
            vx, vy, theta = self.last_action
            v = np.hypot(vx, vy)
            theta = np.interp(
                np.arctan2(vy, vx) - self.qpos[2], [-np.pi / 2, np.pi / 2], [-1, 1]
            )
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


class MjCambrianPointAnimalPredator(MjCambrianPointAnimal):
    """This is an animal which is non-trainable and defines a custom policy which
    acts as a "predator" in the environment. This animal will attempt to catch the prey
    by taking actions that minimize the distance between itself and the prey.

    Keyword Arguments:
        preys (List[str]): The names of the preys in the environment. The prey states
            will be determined from this list by querying the env.
        speed (float): The speed of the predator. Default is 1.0. This is constant
            during the simulation. Must be between -1 and 1, where -1 is no movement.
    """

    def __init__(
        self,
        config: MjCambrianAnimalConfig,
        name: str,
        idx: int,
        *,
        preys: List[str],
        speed: float = 1.0,
    ):
        super().__init__(config, name, idx)

        self._preys = preys
        self._speed = speed

    def get_action_privileged(self, env: "MjCambrianEnv") -> List[float]:
        """This is where the predator will calculate its action based on the prey
        states."""

        # Get the prey states
        prey_pos = [env.animals[prey].pos for prey in self._preys]

        # Calculate the distance between the predator and all preys
        distances = [np.linalg.norm(self.pos - pos) for pos in prey_pos]

        # Calculate the vector that minimizes the distance between the predator and all
        # preys
        min_distance_index = np.argmin(distances)
        min_distance_vector = prey_pos[min_distance_index] - self.pos

        # Calculate the delta from the current angle to the angle that minimizes the
        # distance
        min_distance_angle = np.arctan2(min_distance_vector[1], min_distance_vector[0])
        delta = min_distance_angle - self.last_action[-1]

        # Set the action based on the vector calculated above. Add some noise to the
        # angle to make the movement.
        return [self._speed, np.clip(delta + np.random.randn(), -1, 1)]


class MjCambrianPointAnimalPrey(MjCambrianPointAnimal):
    """This is an animal which is non-trainable and defines a custom policy which
    acts as a "prey" in the environment. This animal will attempt to avoid the predator
    by taking actions that maximize the distance between itself and the predator.

    Keyword Arguments:
        predators (List[str]): The names of the predators in the environment. The
            predator states will be determined from this list by querying the env.
        speed (float): The speed of the prey. Default is 1.0. This is constant during
            the simulation. Must be between -1 and 1, where -1 is no movement.
    """

    def __init__(
        self,
        config: MjCambrianAnimalConfig,
        name: str,
        idx: int,
        *,
        predators: List[str],
        speed: float = 1.0,
    ):
        super().__init__(config, name, idx)

        self._predators = predators
        self._speed = speed

    def get_action_privileged(self, env: "MjCambrianEnv") -> List[float]:
        """This is where the prey will calculate its action based on the predator
        states."""

        # Get the predator states
        predator_pos = [env.animals[predator].pos for predator in self._predators]

        # Calculate the distance between the prey and all predators
        distances = [np.linalg.norm(self.pos - pos) for pos in predator_pos]

        # Calculate the vector that maximizes the distance between the prey and all
        # predators
        max_distance_index = np.argmax(distances)
        max_distance_vector = self.pos - predator_pos[max_distance_index]

        # Calculate the delta from the current angle to the angle that maximizes the
        # distance
        max_distance_angle = np.arctan2(max_distance_vector[1], max_distance_vector[0])
        delta = max_distance_angle - self.last_action[-1]

        # Set the action based on the vector calculated above. Add some noise to the
        # angle to make the movement.
        return [self._speed, np.clip(delta + np.random.randn(), -1, 1)]
