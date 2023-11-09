"""Augmented from `gymnasium_robotics.envs.maze.maze.Maze` to utilize MjCambrianXML."""

from collections import deque
from typing import Tuple
import numpy as np

import mujoco as mj
from gymnasium_robotics.envs.maze.maze import Maze

from cambrian_xml import MjCambrianXML
from config import MjCambrianMazeConfig

RESET = R = "r"  # Initial Reset position of the agent
GOAL = G = "g"
COMBINED = C = "c"  # These cells can be selected as goal or reset locations
WALL = W = "1"
FREE = F = "0"


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

# ================
# CUSTOM MAPS

# U_MAZE but has one reset position and one goal position
U_MAZE_STATIC = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

# Augmented U_MAZE but has one reset position
MANY_GOAL_MAZE = [
    [1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, 1],
    [1, 1, 1, G, G, 1],
    [1, G, G, G, G, 1],
    [1, G, G, G, G, 1],
    [1, G, G, G, G, 1],
    [1, 1, 1, 1, 1, 1],
]

# ================


def make_map(name: str) -> np.ndarray:
    """Returns a map from a name."""
    return np.asarray(globals()[name.upper()], dtype=str)


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

        if self._config.use_target_light_source:
            for light_id in range(model.nlight):
                light_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_LIGHT, light_id)
                if light_name is not None and "target_light_" not in light_name:
                    continue
                self._model.light_pos[light_id][:2] = self.goal

        self._goal_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "target")
        self._model.site_pos[self._goal_site_id][:2] = self.goal

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

    def pos_to_index(self, pos: np.ndarray) -> Tuple[int, int]:
        """Converts a position to an index in the map."""
        x, y = pos
        i = int((self.y_map_center - y) / self.maze_size_scaling - 0.5)
        j = int((x + self.x_map_center) / self.maze_size_scaling - 0.5)
        return i, j

    @classmethod
    def make_maze(
        cls,
        config: MjCambrianMazeConfig,
    ) -> Tuple["MjCambrianMaze", MjCambrianXML]:
        """Overrides the Maze.make_maze method to utilize CambrianXML rather than
        write to file.

        See gymnasium_robotics.envs.maze.maze.Maze.make_maze for more information.

        NOTE:
            The original make_maze method will write to a file in the temporary dir.
            This causes issue because when we're doing parallel environments, one env
            could be writing to the file while another is reading from it. This means it
            might be empty; hence the need to override this method.
        """
        maze_map = make_map(config.name)

        xml = MjCambrianXML.make_empty()
        worldbody = xml.find(".//worldbody")
        assert worldbody is not None

        # Add the material first
        assets = xml.add(xml.root, "asset")
        block_tex = xml.add(
            assets,
            "texture",
            name="block_tex",
            builtin="checker",
            rgb1="0.1 0.1 0.1",
            rgb2="0.9 0.9 0.9",
            width="100",
            height="100",
        )
        xml.add(
            assets,
            "material",
            name="block_mat",
            texture=block_tex.attrib["name"],
            texuniform="true"
        )

        maze = cls(maze_map, config.size_scaling, config.height)
        empty_locations = []
        for i in range(maze.map_length):
            for j in range(maze.map_width):
                struct = maze_map[i][j]
                # Store cell locations in simulation global Cartesian coordinates
                x = (j + 0.5) * config.size_scaling - maze.x_map_center
                y = maze.y_map_center - (i + 0.5) * config.size_scaling
                if struct == WALL:  # Unmovable block.
                    # Offset all coordinates so that maze is centered.
                    # rgba = "0.9 0.9 0.9 1.0" if (i + j) % 2 == 0 else "0.1 0.1 0.1 1.0"
                    size = config.size_scaling
                    xml.add(
                        worldbody,
                        "geom",
                        name=f"block_{i}_{j}",
                        pos=f"{x} {y} {config.height / 2 * config.size_scaling}",
                        size=f"{size / 2} {size / 2} {config.height / 2 * size}",
                        type="box",
                        material="block_mat",
                        contype="1",
                        conaffinity="1",
                    )

                elif struct == RESET:
                    maze._unique_reset_locations.append(np.array([x, y]))
                elif struct == GOAL:
                    maze._unique_goal_locations.append(np.array([x, y]))
                elif struct == COMBINED:
                    maze._combined_locations.append(np.array([x, y]))
                elif struct == 0:
                    empty_locations.append(np.array([x, y]))

        # Add the combined cell locations (goal/reset) to goal and reset
        if (
            not maze._unique_goal_locations
            and not maze._unique_reset_locations
            and not maze._combined_locations
        ):
            # If there are no given "r", "g" or "c" cells in the maze data structure,
            # any empty cell can be a reset or goal location at initialization.
            maze._combined_locations = empty_locations
        maze._unique_goal_locations += maze._combined_locations
        maze._unique_reset_locations += maze._combined_locations

        # Add a floor geometry
        # Going to just be black/grey
        xml.add(
            assets,
            "material",
            name="floor_mat",
            shininess="0.0",
            specular="0.0",
        )
        xml.add(
            xml.add(worldbody, "body", name="floor"),
            "geom",
            name="floor_geom",
            pos="0 0 -0.05",
            size=f"{maze.x_map_center * 2} {maze.y_map_center * 2} 0.1",
            type="plane",
            material="floor_mat",
            rgba="0.1 0.1 0.1 1.0",
            contype="1",
            conaffinity="1",
            condim="1",
        )

        # Change the target site to be a light, if desired. By default, it's a red
        # sphere
        assert config.use_target_light_source is not None
        if config.use_target_light_source:
            # Add the light sources
            for i, dir in enumerate(["0 0 -1"]):
                xml.add(
                    worldbody,
                    "light",
                    name=f"target_light_{i}",
                    pos="0 0 1",
                    dir=dir,
                    cutoff="91", # idk
                    exponent="0.1",
                    attenuation="1 0.14 0.070", # from https://learnopengl.com/Lighting/Light-casters
                    ambient="1 1 1",
                    diffuse="1 1 1",
                    specular="1 1 1",
                    castshadow="false",
                )

        # Visualize the target either way
        # If using light source, the target is emissive
        xml.add(
            assets,
            "material",
            name="site_mat",
            emission="5" if config.use_target_light_source else "0",
        )
        xml.add(
            worldbody,
            "site",
            name="target",
            pos=f"0 0 {config.height / 2 * config.size_scaling}",
            size=f"{0.2 * config.size_scaling}",
            type="sphere",
            rgba="1 1 1 1",
            material="site_mat",
        )

        maze._config = config
        return maze, xml

    def compute_optimal_path(
        self, start: np.ndarray, target: np.ndarray
    ) -> np.ndarray | None:
        """Computes the optimal path from the start position to the target.

        Uses a BFS to find the shortest path.
        """
        start = self.pos_to_index(start)
        target = self.pos_to_index(target)

        rows = len(self.maze_map)
        cols = len(self.maze_map[0])
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        visited[start[0]][start[1]] = True
        queue = deque([([start], 0)])  # (path, distance)

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while queue:
            path, dist = queue.popleft()
            current = path[-1]
            if np.all(current == target):
                # Convert path from indices to positions
                path = [self.index_to_pos(*pos) for pos in path]
                path.append(self.index_to_pos(*target))
                return np.array(path)

            # Check all moves (left, right, up, down, and all diagonals)
            for dr, dc in moves:
                r, c = current[0] + dr, current[1] + dc
                if (
                    0 <= r < rows
                    and 0 <= c < cols
                    and not visited[r][c]
                    and self.maze_map[r][c] != WALL
                ):
                    # If the movement is diagonal, check that the adjacent cells are 
                    # free as well so the path doesn't clip through walls
                    pr, pc = current[0], current[0]
                    if (dr, dc) in moves[4:]:
                        if self.maze_map[r][pc] == WALL or self.maze_map[pr][c] == WALL:
                            continue

                    visited[r][c] = True
                    queue.append((path + [(r, c)], dist + 1))

        return None


if __name__ == "__main__":
    maze, xml = MjCambrianMaze.make_maze(
        MjCambrianMazeConfig(use_target_light_source=True)
    )

    print(xml)
