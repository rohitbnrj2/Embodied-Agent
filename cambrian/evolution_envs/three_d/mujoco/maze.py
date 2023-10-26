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

        self.goal = (
            self.generate_target_goal()
            if self._config.init_goal_pos is None
            else self.index_to_pos(*self._config.init_goal_pos)
        )
        self.goal_z = self.maze_height / 2 * self.maze_size_scaling

        if self._config.use_target_light_source:
            for light_id in range(model.nlight):
                light_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_LIGHT, light_id)
                if "target_light_" not in light_name:
                    continue
                light_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_LIGHT, light_name)
                self._model.light_pos[light_id] = [*self.goal, self.goal_z]
        else:
            self._goal_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "target")
            self._model.site_pos[self._goal_site_id] = [*self.goal, self.goal_z]

        # target_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "target_body")
        # self._model.body_pos[target_body_id] = [*self.goal, self.goal_z]

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

    def index_to_pos(self, i, j) -> np.ndarray:
        """Converts an index in the map to a position."""
        x = (j + 0.5) * self.maze_size_scaling - self.x_map_center
        y = self.y_map_center - (i + 0.5) * self.maze_size_scaling
        return np.array([x, y])

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
        xml.load(tmp_xml_path)

        # Change the target site to be a light, if desired. By default, it's a red
        # sphere
        assert config.use_target_light_source is not None
        if config.use_target_light_source:
            parent = xml.find(".//site[@name='target']/..")

            # Update the target site to be mostly transparent
            site = xml.find(".//site", name="target")
            assert site is not None
            xml.remove(parent, site)

            # Add a light source at the target site
            worldbody = xml.find(".//worldbody")
            assert worldbody is not None
            for i, dir in enumerate(["0 -1 0", "0 1 0", "-1 0 0", "1 0 0"]):
                xml.add(
                    worldbody,
                    "light",
                    name=f"target_light_{i}",
                    pos="0 0 1",
                    dir=dir,
                    cutoff="90",
                    exponent="1",
                )

            # Disable the headlight
            if not config.use_headlight:
                xml.add(xml.add(xml.root, "visual"), "headlight", active="0")

        maze._config = config
        return maze, xml


if __name__ == "__main__":
    maze, xml = MjCambrianMaze.make_maze(MjCambrianMazeConfig())
    print(xml)
