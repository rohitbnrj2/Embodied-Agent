"""Augmented from `gymnasium_robotics.envs.maze.maze.Maze` to utilize MjCambrianXML."""

from typing import Tuple
import numpy as np
import tempfile

import mujoco as mj
from gymnasium_robotics.envs.maze.maze import Maze

from cambrian_xml import MjCambrianXML
from config import MjCambrianMazeConfig

RESET = R = "r"  # Initial Reset position of the agent
GOAL = G = "g"
COMBINED = C = "c"  # These cells can be selected as goal or reset locations


EMPTY_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

OPEN = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

OPEN_DIVERSE_G = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, G, 1],
    [1, G, G, G, G, G, 1],
    [1, G, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

OPEN_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, C, C, C, C, C, 1],
    [1, C, C, C, C, C, 1],
    [1, C, C, C, C, C, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

# Maze specifications for dataset generation
U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

MEDIUM_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

MEDIUM_MAZE_DIVERSE_G = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, G, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, G, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, G, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

MEDIUM_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, C, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, C, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, C, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, C, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

LARGE_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

LARGE_MAZE_DIVERSE_G = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 0, 0, 1, G, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, G, 0, 1, 0, 0, G, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, G, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, G, 0, G, 1, 0, G, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

LARGE_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, C, 0, 0, 0, 1, C, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, C, 0, 1, 0, 0, C, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, C, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, C, 0, C, 1, 0, C, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


def make_map(name: str) -> np.ndarray:
    """Returns a map from a name."""
    return np.asarray(globals()[name.upper()])


class MjCambrianMaze(Maze):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._config: MjCambrianMazeConfig = None

    def reset(self, model: mj.MjModel, data: mj.MjData):
        """Resets the maze. Will generate a target goal and update the site/geom in
        mujoco."""
        self._model = model
        self._data = data

        self.goal = self.generate_target_goal()
        self.goal_z = self.maze_height / 2 * self.maze_size_scaling

        self._goal_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "target")
        self._model.site_pos[self._goal_site_id] = [*self.goal, self.goal_z]

    @classmethod
    def make_maze(
        cls,
        config: MjCambrianMazeConfig,
    ) -> Tuple["MjCambrianMaze", MjCambrianXML]:
        """Overrides the Maze.make_maze method to utilize CambrianXML rather than
        write to file.

        See gymnasium_robotics.envs.maze.maze.Maze.make_maze for more information.
        """
        xml = MjCambrianXML.make_empty()

        maze_map = make_map(config.name)

        # call the original make_maze method with a temporary file that will be deleted
        with tempfile.NamedTemporaryFile("w") as f:
            xml.write(f.name)
            maze, tmp_xml_path = super().make_maze(
                f.name,
                maze_map=maze_map,
                maze_size_scaling=config.size_scaling,
                maze_height=config.height,
            )

        maze._config = config

        xml.load(tmp_xml_path)
        return maze, xml

    def generate_target_goal(self) -> np.ndarray:
        """Taken from `MazeEnv`. Generates a random goal position for an env."""
        assert len(self.unique_goal_locations) > 0
        goal_index = np.random.randint(low=0, high=len(self.unique_goal_locations))
        goal = self.unique_goal_locations[goal_index].copy()
        return goal

    def generate_reset_pos(self) -> np.ndarray:
        """Taken from `MazeEnv`. Generates a random reset position for an animal."""

        assert len(self.unique_reset_locations) > 0

        # While reset position is close to goal position
        reset_pos = self.goal.copy()
        while np.linalg.norm(reset_pos - self.goal) <= 0.5 * self.maze_size_scaling:
            reset_index = np.random.randint(
                low=0, high=len(self.unique_reset_locations)
            )
            reset_pos = self.unique_reset_locations[reset_index].copy()

        return reset_pos


if __name__ == "__main__":
    maze, xml = MjCambrianMaze.make_maze(MjCambrianMazeConfig())
    print(xml)
