"""Augmented from `gymnasium_robotics.envs.maze.maze.Maze` to utilize MjCambrianXML."""

from collections import deque
from typing import Tuple, Dict, List, Any
import numpy as np

import mujoco as mj

from cambrian.evolution_envs.three_d.mujoco.cambrian_xml import MjCambrianXML
from cambrian.evolution_envs.three_d.mujoco.config import MjCambrianMazeConfig
from cambrian.evolution_envs.three_d.mujoco.utils import get_model_path

# ================

RESET = R = "R"  # Initial Reset position of the agent
TARGET = T = "T"
COMBINED = C = "C"  # These cells can be selected as target or reset locations
WALL = W = "1"
EMPTY = E = "0"

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


class MjCambrianMaze:
    """The maze class. Generates a maze from a given map and provides utility
    functions for working with the maze.

    Terminology:
        target: some object which is interacted with (i.e. not a wall)
        goal: the target which the agent is trying to reach
        adversary: the target which the agent is trying to avoid
    """

    def __init__(self, config: MjCambrianMazeConfig):
        self._config: MjCambrianMazeConfig = config

        self._map: np.ndarray = np.array(config.map, dtype=str)

        self._unique_target_locations: List[np.ndarray] = []
        self._unique_reset_locations: List[np.ndarray] = []
        self._combined_locations: List[np.ndarray] = []
        self._empty_locations: List[np.ndarray] = []
        self._wall_locations: List[np.ndarray] = []
        self._occupied_locations: List[np.ndarray] = []
        self._update_locations()

        self._goal: np.ndarray = None
        self._init_goal_pos: np.ndarray = np.array(self._config.init_goal_pos)
        self._adversary: np.ndarray = None
        self._init_adversary_pos: np.ndarray = np.array(self._config.init_adversary_pos)

    def _update_locations(self):
        for i in range(self.map_length):
            for j in range(self.map_width):
                struct = self.map[i][j]

                # Store cell locations in simulation global Cartesian coordinates
                x = (j + 0.5) * self.size_scaling - self.x_map_center
                y = self.y_map_center - (i + 0.5) * self.size_scaling

                if struct == WALL:
                    self._wall_locations.append(np.array([x, y]))
                elif struct == RESET:
                    self._unique_reset_locations.append(np.array([x, y]))
                elif struct == TARGET:
                    self._unique_target_locations.append(np.array([x, y]))
                elif struct == COMBINED:
                    self._combined_locations.append(np.array([x, y]))
                elif struct == EMPTY:
                    self._empty_locations.append(np.array([x, y]))

        # Add the combined cell locations (goal/reset) to goal and reset
        if (
            not self._unique_target_locations
            and not self._unique_reset_locations
            and not self._combined_locations
        ):
            # If there are no given "r", "g" or "c" cells in the maze data structure,
            # any empty cell can be a reset or goal location at initialization.
            self._combined_locations = self._empty_locations
        self._unique_target_locations += self._combined_locations
        self._unique_reset_locations += self._combined_locations

    def generate_xml(self) -> MjCambrianXML:
        xml = MjCambrianXML(get_model_path(self._config.maze_path))
        worldbody = xml.find(".//worldbody")
        assert worldbody is not None

        # Add the walls. Each wall has it's own geom.
        size_scaling = self.size_scaling
        for wall, (x, y) in enumerate(self._wall_locations):
            scale = size_scaling / 2
            xml.add(
                worldbody,
                "geom",
                name=f"block_{wall}",
                pos=f"{x} {y} {scale * self.map_height}",
                size=f"{scale} {scale} {scale * self.map_height}",
                **{"class": "maze_block"}
            )

        # Add the goal/adversary sites
        # Set their positions as placeholders, we'll update them later
        def add_target(
            name: str, *, site_kw: Dict[str, Any] = {}, mat_kw: Dict[str, Any] = {}, top_mat_kw: Dict[str, Any] = {}
        ):
            # Create a body which we use to change the position of the target
            targetbody = xml.add(
                worldbody,
                "body",
                name=f"{name}_body",
                childclass="maze_target"
            )

            # Each target is represented as a site sphere with a material
            xml.add(
                targetbody,
                "site",
                name=f"{name}_site",
                size=f"{0.2 * size_scaling}",
                material=f"{name}_mat",
                **site_kw,
            )

            assets = xml.find(".//asset")
            assert assets is not None
            mat = xml.add(
                assets,
                "material",
                name=f"{name}_mat",
                rgba="1 1 1 1",
                **mat_kw,
            )

            if self.config.use_target_light_sources:
                xml.add(
                    targetbody,
                    "light",
                    name=f"{name}_light",
                    pos="0 0 0",
                )
                # Set the target material to be emissive
                mat.attrib["emission"] = "5"

            # And each target has a small site on top so we can differentiate between
            # the different targets in the birds-eye view
            xml.add(
                assets,
                "material",
                name=f"{name}_top_mat",
                emission="5",
                **top_mat_kw,
            )

            xml.add(
                targetbody,
                "site",
                name=f"{name}_top_site",
                size=f"{0.05 * size_scaling}",
                material=f"{name}_top_mat",
                pos=f"0 0 {0.2 * size_scaling}",
                group="3", # any group > 2 will be hidden to the agents
            )

        mat_kw = dict(texture="vertical_pattern") if self.config.use_adversary else {}
        add_target("goal", mat_kw=mat_kw, top_mat_kw=dict(rgba="0 1 0 1"))
        if self.config.use_adversary:
            add_target(
                "adversary",
                mat_kw=mat_kw,
                site_kw=dict(
                    euler="0 90 0",
                ),
                top_mat_kw=dict(rgba="1 0 0 1"),
            )

        # # Update the floor texture to repeat in a way that matches the blocks
        floor_mat = xml.find(".//material[@name='floor_mat']")
        assert floor_mat is not None, "`floor_mat` not found"
        floor_mat.attrib["texrepeat"] = f"{2 / size_scaling} {2 / size_scaling}"

        # # Update floor size
        floor = xml.find(".//geom[@name='floor']")
        assert floor is not None, "`floor` not found"
        floor.attrib["size"] = f"{self.map_width_scaled} {self.map_length_scaled} 0.1"

        return xml

    def reset(self, model: mj.MjModel):
        """Resets the maze. Will generate a goal and update the site/geom in mujoco."""
        self._occupied_locations.clear()
        self._goal = (
            self._init_goal_pos if self._init_goal_pos else self.generate_target_pos()
        )
        if self.config.use_adversary:
            assert len(self._unique_target_locations) > 1, (
                "Must have at least 2 unique target locations to use an adversary"
            )
            self._adversary = (
                self._init_adversary_pos
                if self._init_adversary_pos
                else self.generate_target_pos()
            )

        # Update target bodies position and size
        def update_target(name: str, pos: np.ndarray):
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{name}_body")
            assert body_id != -1, f"`{name}_body` body not found"
            model.body_pos[body_id] = [*pos, self.map_height * self.size_scaling // 2]

            if self.config.use_target_light_sources:
                light_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_LIGHT, f"{name}_light")
                model.light_attenuation[light_id] = get_attenuation(self.max_dim * 1.75)

        update_target("goal", self.goal)
        if self.config.use_adversary:
            update_target("adversary", self.adversary)

    def _generate_pos(
        self, locations: List[np.ndarray], *, tries: int = 10
    ) -> np.ndarray:
        """Helper method to generate a position. The generated position must be at a
        unique location from self._occupied_locations."""
        assert len(locations) > 0, "No locations to choose from"

        for _ in range(tries):
            idx = np.random.randint(low=0, high=len(locations))
            pos = locations[idx].copy()

            # Check if the position is already occupied
            for occupied in self._occupied_locations:
                if np.linalg.norm(pos - occupied) <= 0.5 * self.size_scaling:
                    break
            else:
                return pos

    def generate_target_pos(self, *, add_as_occupied: bool = True) -> np.ndarray:
        """Generates a random target position for an env."""
        target_pos = self._generate_pos(self.unique_target_locations)
        if add_as_occupied:
            self._occupied_locations.append(target_pos)
        return target_pos

    def generate_reset_pos(self, *, add_as_occupied: bool = True) -> np.ndarray:
        """Generates a random reset position for an env."""
        reset_pos = self._generate_pos(self.unique_reset_locations)
        if add_as_occupied:
            self._occupied_locations.append(reset_pos)
        return reset_pos

    def compute_optimal_path(
        self, start: np.ndarray, target: np.ndarray
    ) -> np.ndarray | None:
        """Computes the optimal path from the start position to the target.

        Uses a BFS to find the shortest path.
        """
        start = self.cell_xy_to_rowcol(start)
        target = self.cell_xy_to_rowcol(target)

        rows = self.map_length
        cols = self.map_width
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
                    and self.map[r][c] != WALL
                ):
                    # If the movement is diagonal, check that the adjacent cells are
                    # free as well so the path doesn't clip through walls
                    pr, pc = current[0], current[1]
                    if (dr, dc) in moves[4:]:
                        if self.map[r][pc] == WALL or self.map[pr][c] == WALL:
                            continue

                    visited[r][c] = True
                    queue.append((path + [(r, c)], dist + 1))

        return None

    def cell_rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        x = (rowcol_pos[1] + 0.5) * self.size_scaling - self.x_map_center
        y = self.y_map_center - (rowcol_pos[0] + 0.5) * self.size_scaling

        return np.array([x, y])

    def cell_xy_to_rowcol(self, xy_pos: np.ndarray) -> np.ndarray:
        i = np.floor((self.y_map_center - xy_pos[1]) / self.size_scaling)
        j = np.floor((xy_pos[0] + self.x_map_center) / self.size_scaling)
        return np.array([i, j], dtype=int)

    # ==================

    @property
    def config(self) -> MjCambrianMazeConfig:
        """Returns the config."""
        return self._config

    @property
    def goal(self) -> np.ndarray:
        """Returns the goal."""
        return self._goal

    @property
    def adversary(self) -> np.ndarray:
        """Returns the adversary."""
        return self._adversary

    @property
    def map(self) -> np.ndarray:
        """Returns the maze map."""
        return self._map

    @property
    def map_length(self) -> int:
        """Returns the map length."""
        return len(self._map)

    @property
    def map_width(self) -> int:
        """Returns the map width."""
        return len(self._map[0])

    @property
    def map_height(self) -> int:
        """Returns the map height."""
        return self._config.height

    @property
    def size_scaling(self) -> float:
        """Returns the size scaling."""
        return self._config.size_scaling

    @property
    def map_length_scaled(self) -> float:
        """Returns the map length scaled."""
        return self.map_length * self.size_scaling

    @property
    def map_width_scaled(self) -> float:
        """Returns the map width scaled."""
        return self.map_width * self.size_scaling

    @property
    def max_dim(self) -> float:
        """Returns the max dimension."""
        return max(self.map_length_scaled, self.map_width_scaled)

    @property
    def x_map_center(self) -> float:
        """Returns the x map center."""
        return self.map_width_scaled / 2

    @property
    def y_map_center(self) -> float:
        """Returns the y map center."""
        return self.map_length_scaled / 2

    @property
    def wall_locations(self) -> List[np.ndarray]:
        """Returns the wall locations."""
        return self._wall_locations

    @property
    def unique_reset_locations(self) -> List[np.ndarray]:
        """Returns the unique reset locations."""
        return self._unique_reset_locations

    @property
    def unique_target_locations(self) -> List[np.ndarray]:
        """Returns the unique target locations."""
        return self._unique_target_locations

    @property
    def combined_locations(self) -> List[np.ndarray]:
        """Returns the combined locations."""
        return self._combined_locations

    @property
    def empty_locations(self) -> List[np.ndarray]:
        """Returns the empty locations."""
        return self._empty_locations

    @property
    def occupied_locations(self) -> List[np.ndarray]:
        """Returns the occupied locations."""
        return self._occupied_locations


if __name__ == "__main__":
    import time

    config = MjCambrianMazeConfig(
        use_target_light_sources=True,
        use_adversary=True,
        name="MANY_GOAL_MAZE",
        maze_path="models/maze.xml",
        size_scaling=4.0,
        height=0.5,
    )
    t0 = time.time()
    maze = MjCambrianMaze(config)
    t1 = time.time()

    xml = maze.generate_xml()
    t2 = time.time()
    print(xml)

    t3 = time.time()
    start = maze.generate_reset_pos()
    goal = maze.generate_target_pos()
    adversary = maze.generate_target_pos()
    t4 = time.time()

    t5 = time.time()
    path = maze.compute_optimal_path(start, goal)
    t6 = time.time()

    print(f"Time to initialize maze: {(t1 - t0) * 1000:.5f}ms")
    print(f"Time to generate xml: {(t2 - t1) * 1000:.5f}ms")
    print(f"Time to generate pos: {(t4 - t3) * 1000:.5f}ms")
    print(f"Time to compute optimal path: {(t6 - t5) * 1000:.5f}ms")
