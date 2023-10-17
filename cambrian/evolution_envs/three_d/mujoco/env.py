from typing import Dict, Any, Tuple
from pathlib import Path
import tempfile
import numpy as np
import cv2

import mujoco as mj
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3.common.utils import set_random_seed
from pettingzoo.utils.env import ParallelEnv

from animal import MjCambrianAnimal
from maze import MjCambrianMaze
from cambrian_xml import MjCambrianXML
from config import MjCambrianConfig, MjCambrianAnimalConfig


def make_env(rank: int, seed: float, config_path: str | Path) -> "MjCambrianEnv":
    """Utility function for multiprocessed MjCambrianEnv."""

    def _init():
        env = MjCambrianEnv(config_path)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed + rank)
    return _init


class MjCambrianEnv(MujocoEnv, ParallelEnv):
    """A MjCambrianEnv defines a gymnasium environment that's based off mujoco.

    In our context, a MjCambrianEnv contains a maze and at least one animal.

    NOTE: a third person tracking camera is implemented through the `render()` method
    in the `MujocoEnv` class. By default, this is set in `MjCambrianEnvConfig`, which is
    "track" by default.

    NOTE #2: the `action_space` is defined by MujocoEnv as all the controllable joints
    present in the simulation.

    TODO: Should load in the base model, do checks on geometry, etc. and then reload
    the xml

    Args:
        config_path (str | Path | MjCambrianConfig): The path to the config file or the
            config object itself.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, config: str | Path | MjCambrianConfig):
        self.config = MjCambrianConfig.load(config)
        self.env_config = self.config.env_config

        self.animals: Dict[str, MjCambrianAnimal] = {}
        self._create_animals()

        self.xml = self.generate_xml()

        # Write the xml to a tempfile so that MujocoEnv can read it in
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            self.xml.write(f.name)
            self.env_config.model_path = f.name

        # Create the observation_spaces
        observation_spaces: Dict[str, spaces.Space] = {}
        for name, animal in self.animals.items():
            observation_spaces[name] = animal.observation_space

        MujocoEnv.__init__(
            self,
            model_path=self.env_config.model_path,
            frame_skip=self.env_config.frame_skip,
            observation_space=observation_spaces,
            render_mode=self.env_config.render_mode,
            width=self.env_config.width,
            height=self.env_config.height,
            camera_name=self.env_config.camera_name,
        )

        # Set the class variables associated with the ParallelEnv
        # observation_spaces is already set
        # NOTE: possible_agents is assumed to be the same as agents
        self.agents = list(self.animals.keys())
        self.possible_agents = self.agents
        self.action_spaces = {n: a.action_space for n, a in self.animals.items()}

    def _create_animals(self):
        """Helper method to create the animals."""
        default_animal_config = self.config.animal_config
        for i in range(self.env_config.num_animals):
            animal_config = MjCambrianAnimalConfig(**default_animal_config)
            if animal_config.name is None:
                animal_config.name = f"animal_{i}"
            assert animal_config.name not in self.animals
            self.animals[animal_config.name] = MjCambrianAnimal.create(animal_config)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = MjCambrianXML.make_empty()

        # Add the animals to the xml
        for animal in self.animals.values():
            xml += animal.generate_xml()

        # Create the maze and add it to the xml
        self.maze, maze_xml = MjCambrianMaze.make_maze(self.env_config.maze_config)
        xml += maze_xml

        # Create the track camera, if it doesn't exist. camera_name must be set
        # NOTE: tracks the first animal only
        track_cam = self.env_config.camera_name
        if track_cam is not None and xml.find(".//camera", name=track_cam) is None:
            animal = next(iter(self.animals.values()))
            tracked_body_name = animal.config.body_name
            tracked_body = xml.find(".//body", name=tracked_body_name)
            assert tracked_body is not None
            xml.add(
                tracked_body,
                "camera",
                name=track_cam,
                mode="trackcom",
                pos="0 -10 10",
                xyaxes="1 0 0 0 1 1",
            )

        return xml

    def reset_model(self) -> Dict[str, Dict[str, Any]]:
        """Resets the underlying env. Called by `reset` in `MujocoEnv`."""
        self.maze.reset(self.model, self.data)

        obs: Dict[str, Dict[str, Any]] = {}
        for name, animal in self.animals.items():
            init_qpos = self.maze.generate_reset_pos()
            obs[name] = animal.reset(self.model, self.data, init_qpos)

        return obs

    def _get_reset_info(self) -> Dict[str, Dict[str, Any]]:
        """Function that generates the `info` that is returned during a `reset()`.
        Called by `reset` in `MujocoEnv`."""
        return {a: {} for a in self.animals}

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Step the environment.

        THe dynamics is updated through the `do_simulation` method.

        Args:
            action (Dict[str, Any]): The action to take for each animal. The keys
            define the animal name, and the values define the action for that animal.

        Returns:
            Dict[str, Any]: The observations for each animal.
            Dict[str, float]: The reward for each animal.
            Dict[str, bool]: Whether each animal has terminated.
            Dict[str, bool]: Whether each animal has truncated.
            Dict[str, Dict[str, Any]]: The info dict for each animal.
        """
        obs: Dict[str, Any] = {}
        for name, animal in self.animals.items():
            obs[name] = animal.step(action[name])

        reward = self.compute_reward()
        terminated = self.compute_terminated()
        truncated = self.compute_truncated()
        info = {a: {} for a in self.animals}

        self._step_mujoco_simulation(self.frame_skip)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _step_mujoco_simulation(self, n_frames):
        """Overwrites the MujocoEnv method to allow for directly setting the actuators
        on an animal-by-animal basis."""
        mj.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mj.mj_rnePostConstraint(self.model, self.data)

    def compute_reward(self) -> Dict[str, float]:
        """Computes the reward for the environment. Currently just euclidean distance
        to the goal.

        Using dense reward in maze_env.
        """

        reward: Dict[str, float] = {}
        for name, animal in self.animals.items():
            reward[name] = np.exp(-np.linalg.norm(animal.qpos[:2] - self.maze.goal))

        return reward

    def compute_terminated(self) -> Dict[str, bool]:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure."""

        terminated: Dict[str, bool] = {}
        for name, animal in self.animals.items():
            terminated[name] = np.linalg.norm(animal.qpos[:2] - self.maze.goal) < 0.5

        return terminated

    def compute_truncated(self) -> bool:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure. Failure, for now, indicates that the
        animal has touched the wall."""

        truncated: Dict[str, bool] = {}
        for name, animal in self.animals.items():
            truncated[name] = False  # TODO

        return truncated

    def render(self):
        """Override the render method only to add human rendering.
        Doesn't work otherwise for some reason.

        NOTE: This is more of a patch of the non-functional "human" mode. Really should
        figure out why it's not working first.
        """

        if self.render_mode == "human":
            self.render_mode = "rgb_array"
            cv2.imshow("env", self.render()[:, :, ::-1])
            cv2.waitKey(1)
            self.render_mode = "human"
        else:
            return super().render()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("config_path", type=str, help="Path to the config file.")
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=MjCambrianEnv.metadata["render_modes"],
        default="rgb_array",
    )

    args = parser.parse_args()

    config = MjCambrianConfig.load(args.config_path)
    # TODO: human not working for some reason
    config.env_config.render_mode = args.render_mode

    env = MjCambrianEnv(config)
    env.reset()

    import cv2

    for _ in range(1000):
        env.step(env.action_space.sample())
        image = env.render()

        if args.render_mode != "human":
            image = image[:, :, ::-1] if args.render_mode == "rgb_array" else image
            cv2.imshow("render", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
