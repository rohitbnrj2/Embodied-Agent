"""Augmented from `gymnasium_robotics.envs.maze.maze.Maze` to utilize MjCambrianXML."""

from collections import deque
from typing import Tuple, Dict
import numpy as np

import mujoco as mj
from gymnasium_robotics.envs.maze.maze import Maze

from cambrian_xml import MjCambrianXML
from config import MjCambrianMazeConfig
from utils import get_model_path

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
# EVAL MAPS

OPTIC_FLOW_EVAL_MAZE_TUNNEL = [
    [1, R, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, G, 1],
    [1, 1, 1, 1, 1],
]

OPTIC_FLOW_EVAL_MAZE_SNAKE = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, 0, G, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]
# ================

# from https://learnopengl.com/Lighting/Light-casters
ATTENUATION_LOOKUP_TABLE: Dict[int, Tuple[float, float, float]] = {
    0: (1.0, 0.0, 0.0),  # no attenuation
    7: (1.0, 0.7, 1.8),
    13: (1.0, 0.35, 0.44),
    20: (1.0, 0.22, 0.20),
    32: (1.0, 0.14, 0.07),
    50: (1.0, 0.09, 0.032),
    65: (1.0, 0.07, 0.017),
    100: (1.0, 0.045, 0.0075),
    160: (1.0, 0.027, 0.0028),
    200: (1.0, 0.022, 0.0019),
    325: (1.0, 0.014, 0.0007),
    600: (1.0, 0.007, 0.0002),
    3250: (1.0, 0.0014, 0.000007),
}


def get_attenuation(max_distance: float) -> Tuple[float, float, float]:
    """Returns the attenuation for a light source given the max distance."""
    return ATTENUATION_LOOKUP_TABLE[
        max(k for k in ATTENUATION_LOOKUP_TABLE if k <= max_distance)
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
            else self.cell_rowcol_to_xy(self._config.init_goal_pos)
        )

        # Update target site position and size
        self._goal_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "target")
        assert self._goal_site_id is not None, "`target` site not found"
        self._model.site_pos[self._goal_site_id][:2] = self.goal
        self._model.site_size[self._goal_site_id] = np.array(
            [0.2 * self.maze_size_scaling] * 3
        )

        # Update target light position and attenuation
        if self._config.use_target_light_source:
            light_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_LIGHT, "target_light")
            self._model.light_pos[light_id][:2] = self.goal
            self._model.light_attenuation = get_attenuation(max(self.xy_size) * 1.75)

        # Update floor size
        floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor_geom")
        assert floor_id is not None, "`floor_geom` not found"
        self._model.geom_size[floor_id] = np.array(
            [self.x_map_center * 2, self.y_map_center * 2, 0.1]
        )

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

    @property
    def xy_size(self) -> Tuple[float, float]:
        """Calculate the size in xy coordinates."""
        return (
            self.map_width * self.maze_size_scaling,
            self.map_length * self.maze_size_scaling,
        )

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

        NOTE #2: The block has a gap and margin of 0.025 and 0.05 respectively. This
            means that if the animal get's within 0.05 of the block, it will be recorded
            as a contact, but there is no actual contact force applied by mujoco. This
            is helpful so that we can terminate the simulation before the animal 
            actually comes in conatct with the block and the camera/eye starts seeing
            inside of the block.
        """
        maze_map = make_map(config.name)

        xml = MjCambrianXML(get_model_path(config.maze_path))
        worldbody = xml.find(".//worldbody")
        assert worldbody is not None

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
                        gap="0.05",
                        margin="0.1",
                        material="block_mat",
                        contype="1",
                        conaffinity="1",
                        group="1",
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

        # If target_light_source is not requested, remove it from the xml and update
        # the target site to not be emissive
        if not config.use_target_light_source:
            target_light = xml.find(".//light[@name='target_light']")
            assert target_light is not None, "`target_light` not found"
            xml.remove(worldbody, target_light)

            target_mat = xml.find(".//material[@name='target_mat']")
            assert target_mat is not None, "`target_mat` not found"
            target_mat.attrib["emission"] = "0"

        # Update the floor texture to repeat in a way that matches the blocks
        floor_mat = xml.find(".//material[@name='floor_mat']")
        assert floor_mat is not None, "`floor_mat` not found"
        floor_mat.attrib["texrepeat"] = " ".join(map(str, [2 / config.size_scaling] * 2))

        maze._config = config
        return maze, xml

    def compute_optimal_path(
        self, start: np.ndarray, target: np.ndarray
    ) -> np.ndarray | None:
        """Computes the optimal path from the start position to the target.

        Uses a BFS to find the shortest path.
        """
        start = self.cell_xy_to_rowcol(start)
        target = self.cell_xy_to_rowcol(target)

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
                path = [self.cell_rowcol_to_xy(pos) for pos in path]
                path.append(self.cell_rowcol_to_xy(target))
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
