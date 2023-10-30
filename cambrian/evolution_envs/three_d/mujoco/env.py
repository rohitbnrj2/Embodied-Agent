from typing import Dict, Any, Tuple
from pathlib import Path
import tempfile
import numpy as np

import mujoco as mj
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3.common.utils import set_random_seed

from animal import MjCambrianAnimal
from maze import MjCambrianMaze
from cambrian_xml import MjCambrianXML
from config import MjCambrianConfig, MjCambrianAnimalConfig
from utils import get_model_path


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

    Initialization progression goes as follows:
    - create each animal and for each
        - load the base xml to MjModel
        - parse the geometry and place eyes at the appropriate locations
        - load the actuators/joints
        - create the action/observation spaces
        - return the a new xml which includes adjustments (e.g. eyes/cameras, etc.)
    - create the environment xml (maze + animals + etc.)
    - create the main MjModel/MjData (through MujocoEnv constructor)

    NOTES:
    - a third person tracking camera is implemented through the `render()` method
    in the `MujocoEnv` class. By default, this is set in `MjCambrianEnvConfig`, which is
    "track" by default.
    - The MujocoEnv uses the standard gym API that only supports single agents in the
    simulation. This environment instead uses the pettingzoo.ParallelEnv API, which
    is similar to gym but inputs/outputs a dictionary for each type that encodes the obj
    on a per-agent basis.

    Args:
        config_path (str | Path | MjCambrianConfig): The path to the config file or the
            config object itself.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, config: str | Path | MjCambrianConfig):
        self.config = MjCambrianConfig.load(config)
        self.env_config = self.config.env_config

        self.env_config.scene_path = get_model_path(self.env_config.scene_path)

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
            observation_space: spaces.Dict = animal.observation_space
            observation_space.spaces["goal"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            )
            observation_space.spaces["pos"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
            observation_spaces[name] = observation_space
        self.observation_spaces = spaces.Dict(observation_spaces)

        MujocoEnv.__init__(
            self,
            model_path=self.env_config.model_path,
            frame_skip=self.env_config.frame_skip,
            observation_space=self.observation_spaces,
            render_mode=self.env_config.render_mode,
        )

        # Set the class variables associated with the ParallelEnv
        # observation_spaces is already set
        # NOTE: possible_agents is assumed to be the same as agents
        self.agents = list(self.animals.keys())
        self.possible_agents = self.agents
        self.action_spaces = spaces.Dict(
            {n: a.action_space for n, a in self.animals.items()}
        )

        self._episode_step = 0
        self._max_episode_steps = int(
            self.config.training_config.n_steps * 0.9)

    def _create_animals(self):
        """Helper method to create the animals.

        Under the hood, the `create` method does the following:
            - load the base xml to MjModel
            - parse the geometry and place eyes at the appropriate locations
            - create the action/observation spaces
        """
        default_animal_config = self.config.animal_config
        for i in range(self.env_config.num_animals):
            animal_config = MjCambrianAnimalConfig(**default_animal_config)
            animal_config.idx = i
            if animal_config.name is None:
                animal_config.name = f"animal_{i}"
            assert animal_config.name not in self.animals
            self.animals[animal_config.name] = MjCambrianAnimal.create(
                animal_config)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = MjCambrianXML(self.env_config.scene_path)

        # Create the directional light, if desired
        if self.env_config.use_directional_light:
            xml.add(
                xml.find(".//worldbody"),
                "light",
                directional="true",
                cutoff="100",
                exponent="1",
                diffuse="1 1 1",
                specular=".1 .1 .1",
                pos="0 0 1.3",
                dir="-0 0 -1.3",
            )
        if self.env_config.maze_config.use_target_light_source is None:
            self.env_config.maze_config.use_target_light_source = (
                not self.env_config.use_directional_light
            )

        # Add the animals to the xml
        for animal in self.animals.values():
            xml += animal.generate_xml()

        # Create the maze and add it to the xml
        self.maze, maze_xml = MjCambrianMaze.make_maze(
            self.env_config.maze_config)
        xml += maze_xml

        # Create the track camera, if it doesn't exist. camera_name must be set
        # NOTE: tracks the first animal only
        track_cam = self.env_config.camera_name
        if track_cam is not None and xml.find(".//camera", name=track_cam) is None:
            animal = next(iter(self.animals.values()))
            tracked_body = xml.find(
                ".//body", name=f"{animal.config.body_name}")
            assert tracked_body is not None
            xml.add(
                tracked_body,
                "camera",
                name=track_cam,
                mode="trackcom",
                pos="0 -10 10",
                xyaxes="1 0 0 0 1 1",
            )

        # Update the assert path to point to the fully resolved path
        compiler = xml.find(".//compiler")
        assert compiler is not None
        if (texturedir := compiler.attrib.get("texturedir")) is not None:
            texturedir = str(get_model_path(xml.base_dir / texturedir))
            compiler.attrib["texturedir"] = texturedir
        if (meshdir := compiler.attrib.get("meshdir")) is not None:
            meshdir = str(get_model_path(xml.base_dir / meshdir))
            compiler.attrib["meshdir"] = meshdir
        if (assetdir := compiler.attrib.get("assetdir")) is not None:
            assetdir = str(get_model_path(xml.base_dir / assetdir))
            compiler.attrib["assetdir"] = assetdir

        return xml

    def reset_model(self) -> Dict[str, Dict[str, Any]]:
        """Resets the underlying env. Called by `reset` in `MujocoEnv`."""
        self.maze.reset(self.model, self.data)

        obs: Dict[str, Dict[str, Any]] = {}
        for name, animal in self.animals.items():
            init_qpos = (
                self.maze.index_to_pos(*animal.config.init_pos)
                if animal.config.init_pos is not None
                else self.maze.generate_reset_pos()
            )
            obs[name] = animal.reset(self.model, self.data, init_qpos)
            obs[name]["goal"] = self.maze.goal.copy()
            obs[name]["pos"] = animal.pos.copy()

        self._episode_step = 0

        self._step_mujoco_simulation(self.frame_skip)

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
        pos: Dict[str, np.ndarray] = {}  # to keep track of previous position
        for name, animal in self.animals.items():
            pos[name] = animal.pos.copy()

            obs[name] = animal.step(action[name])
            obs[name]["goal"] = self.maze.goal.copy()
            obs[name]["pos"] = animal.pos.copy()

        self._step_mujoco_simulation(self.frame_skip)

        terminated = self.compute_terminated()
        truncated = self.compute_truncated()
        reward = self.compute_reward(pos, terminated, truncated)
        info = {a: {} for a in self.animals}

        if self.render_mode == "human":
            self.render()

        self._episode_step += 1

        return obs, reward, terminated, truncated, info

    def _step_mujoco_simulation(self, n_frames):
        """Overrides the MujocoEnv method to allow for directly setting the actuators
        on an animal-by-animal basis."""
        mj.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mj.mj_rnePostConstraint(self.model, self.data)

    def compute_reward(
        self,
        prev_pos: Dict[str, np.ndarray],
        terminated: Dict[str, bool],
        truncated: Dict[str, bool],
    ) -> Dict[str, float]:
        """Computes the reward for the environment.

        Currently calculates the change in distance to the goal from the previous step,
        rewarding the agent for getting closer to the goal.

        Args:
            prev_pos (Dict[str, np.ndarray]): The previous position of each animal.
            terminated (Dict[str, bool]): Whether each animal has terminated.
            Termination indicates success (agent has reached the goal).
            truncated (Dict[str, bool]): Whether each animal has truncated.
            Truncation indicates failure (agent has hit the wall or something).
        """

        reward: Dict[str, float] = {}
        for name, animal in self.animals.items():
            if terminated[name]:
                reward[name] = 5.0
                continue
            elif truncated[name]:
                reward[name] = -5.0
                continue

            # dpos is positive if the animal is getting closer to the goal
            curr_pos = animal.pos
            delta_pos = np.linalg.norm(
                prev_pos[name]) - np.linalg.norm(curr_pos)
            reward[name] = delta_pos if delta_pos > 0.005 else 0.0

        return reward

    def compute_terminated(self) -> Dict[str, bool]:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure."""

        terminated: Dict[str, bool] = {}
        for name, animal in self.animals.items():
            terminated[name] = np.linalg.norm(
                animal.pos[:2] - self.maze.goal) < 0.5

        return terminated

    def compute_truncated(self) -> bool:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure. Failure, for now, indicates that the
        animal has touched the wall."""

        truncated: Dict[str, bool] = {}
        for name in self.animals:
            truncated[name] = self._episode_step >= self._max_episode_steps

        return truncated

    def render(self):
        """Override the render method to fix viewer bug with custom managed viewers."""

        if self.mujoco_renderer.viewer is not None:
            # MujocoEnv only will run make_context_current if it has multiple viewers
            # We have our own managed viewers as eyes, so we need to manually call it
            self.mujoco_renderer.viewer.make_context_current()
        return super().render()


if __name__ == "__main__":
    import argparse
    from cambrian.reinforce.evo.runner import _update_config_with_overrides

    parser = argparse.ArgumentParser()

    parser.add_argument("config_path", type=str,
                        help="Path to the config file.")
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action="append",
        type=lambda v: v.split("="),
        help="Override config values. Do <dot separated yaml config>=<value>",
        default=[],
    )

    args = parser.parse_args()
    args.overrides.insert(0, ("env_config.render_mode", "human"))

    config = MjCambrianConfig.load(args.config_path)
    _update_config_with_overrides(config, args.overrides)

    env = MjCambrianEnv(config)
    env.render_mode = "rgb_array"
    env.reset()
    env.render_mode = config.env_config.render_mode

    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            mj.mj_step(env.model, env.data)
            viewer.sync()
        print("Exiting...")
