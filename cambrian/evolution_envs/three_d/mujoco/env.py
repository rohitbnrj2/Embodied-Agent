from typing import Dict, Any, SupportsFloat, Tuple, List
from pathlib import Path
import tempfile
import numpy as np
import cv2

from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.core import ActType, ObsType
from stable_baselines3.common.utils import set_random_seed

from animal import MjCambrianAnimal
from maze import MjCambrianMaze
from cambrian_xml import MjCambrianXML
from config import MjCambrianConfig


def make_env(rank: int, seed: float, config_path: str | Path) -> "MjCambrianEnv":
    """Utility function for multiprocessed MjCambrianEnv."""

    def _init():
        env = MjCambrianEnv(config_path)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed + rank)
    return _init


class MjCambrianEnv(MujocoEnv):
    """A MjCambrianEnv defines a gymnasium environment that's based off mujoco.

    In our context, a MjCambrianEnv contains a maze and at least one animal.

    NOTE: a third person tracking camera is implemented through the `render()` method
    in the `MujocoEnv` class. By default, this is set in `MjCambrianEnvConfig`, which is
    "track" by default.

    NOTE #2: the `action_space` is defined by MujocoEnv as all the controllable joints
    present in the simulation.

    Args:
        config_path (str | Path | MjCambrianConfig): The path to the config file or the
            config object itself.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, config: str | Path | MjCambrianConfig):
        self.config = MjCambrianConfig.load(config)
        self.env_config = self.config.env_config

        self.animals: List[MjCambrianAnimal] = []
        self.xml = self.generate_xml()

        # Write the xml to a tempfile so that MujocoEnv can read it in
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            self.xml.write(f.name)
            self.env_config.model_path = f.name

        # Create a flattened observations space
        self.observation_space = spaces.Dict()
        for animal in self.animals:
            for eye in animal.eyes:
                key = f"{animal.name}_{eye.name}"
                self.observation_space.spaces[key] = eye.observation_space
        self.observation_space.spaces["position"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        super().__init__(
            model_path=self.env_config.model_path,
            frame_skip=self.env_config.frame_skip,
            observation_space=self.observation_space,
            render_mode=self.env_config.render_mode,
            width=self.env_config.width,
            height=self.env_config.height,
            camera_name=self.env_config.camera_name,
            camera_id=self.env_config.camera_id,
        )

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = MjCambrianXML.make_empty()

        # Create the animals and add them to the xml
        animal_config = self.config.animal_config
        for i in range(self.env_config.num_animals):
            animal = MjCambrianAnimal(animal_config)
            self.animals.append(animal)

            xml += animal.generate_xml()

        # Create the maze and add it to the xml
        self.maze, maze_xml = MjCambrianMaze.make_maze(self.env_config.maze_config)
        xml += maze_xml

        # Create the track camera, if it doesn't exist
        # NOTE: tracks the first animal only
        # TODO: currently only supports camera_name, not camera_id
        track_cam = self.env_config.camera_name
        if track_cam is not None and xml.find(".//camera", name=track_cam) is None:
            tracked_body_name = self.animals[0].config.body_name
            tracked_body = xml.find(".//body", name=tracked_body_name)
            assert (
                tracked_body is not None
            ), f"Could not find body with name {tracked_body_name}."
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
        for animal in self.animals:
            init_qpos = self.maze.generate_reset_pos()
            obs[animal.name] = animal.reset(self.model, self.data, init_qpos)

        return self._get_obs(obs)

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.do_simulation(action, self.frame_skip)

        obs: Dict[str, Dict[str, Any]] = {}
        for animal in self.animals:
            obs[animal.name] = animal.step()

        reward = self.compute_reward()
        terminated = self.compute_terminated()
        truncated = self.compute_truncated()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(obs), reward, terminated, truncated, {}

    def compute_reward(self) -> float:
        """Computes the reward for the environment. Currently just euclidean distance
        to the goal.
        
        Using dense reward in maze_env.
        """

        reward = 0
        for animal in self.animals:
            reward += np.exp(-np.linalg.norm(animal.qpos[:2] - self.maze.goal))

        return reward

    def compute_terminated(self) -> bool:
        """Compute whether the env has terminated. Termination indicates success, 
        whereas truncated indicates failure.
        
        TODO: Assumes there is one animal
        """
        animal = self.animals[0]

        return np.linalg.norm(animal.qpos[:2] - self.maze.goal) < 0.5

    def compute_truncated(self) -> bool:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure. Failure, for now, indicates that the
        animal has touched the wall.
        
        TODO: Assumes there is one animal
        """
        animal = self.animals[0]

        # TODO
        return False

    def _get_obs(self, obs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Flattens the observations from the animal's eyes."""
        out_obs: Dict[str, Any] = {}
        for animal_obs_name, animal_obs in obs.items():
            for eye_obs_name, eye_obs in animal_obs.items():
                out_obs[f"{animal_obs_name}_{eye_obs_name}"] = eye_obs
        out_obs["position"] = self.animals[0].qpos[:2]
        return out_obs

    def render(self):
        """Override the render method only to add human rendering. 
        Doesn't work otherwise for some reason."""

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
