"""Augmented from `gymnasium_robotics.envs.maze.maze.Maze` to utilize MjCambrianXML."""

from typing import Tuple, Dict, List, Any, Optional, Type, TYPE_CHECKING
from collections import deque
import numpy as np

import mujoco as mj

from cambrian.utils import (
    get_geom_id,
    get_body_id,
    get_site_id,
    get_light_id,
)
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.config import MjCambrianMazeConfig, MjCambrianMazeSelectionFn

if TYPE_CHECKING:
    from cambrian.env import MjCambrianEnv

# ================

RESET = R = "R"  # Initial Reset position of the agent
TARGET = T = "T"
COMBINED = C = "C"  # These cells can be selected as target or reset locations
WALL = W = "1"
EMPTY = E = "0"
ADVERSARY = A = (
    "A"  # Optional adversary location - if not in maze, a target will be selected
)

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

# [center_x, width]
MAP_EXTENTS: List[Tuple[int, int]] = [(0, 0)]


class MjCambrianMaze:
    """The maze class. Generates a maze from a given map and provides utility
    functions for working with the maze.

    Terminology:
        target: some object which is interacted with (i.e. not a wall)
        goal: the target which the agent is trying to reach
        adversary: the target which the agent is trying to avoid
    """

    def __init__(self, config: MjCambrianMazeConfig, name: str, *, ref: "MjCambrianMaze" = None):
        self._config = config
        self._name = name
        self._ref = ref

        self._map: np.ndarray = np.array(config.map, dtype=str)
        if config.flip:
            self._map = np.flip(self._map)

        if ref is not None:
            # If we're using a reference maze, copy the map center from the reference
            self._x_center = ref._x_center
        else:
            # Otherwise, move the map origin such that the map doesn't overlap with
            # existing maps. We'll place the map next to the previous one
            prev_center, prev_width = MAP_EXTENTS[-1]
            center = prev_center + prev_width / 2 + self.map_width_scaled / 2
            MAP_EXTENTS.append((center, self.map_width_scaled))
            self._x_center = center

        self._wall_textures: List[str] = []
        self._unique_target_locations: List[np.ndarray] = []
        self._unique_adv_locations: List[np.ndarray] = []
        self._unique_reset_locations: List[np.ndarray] = []
        self._combined_locations: List[np.ndarray] = []
        self._empty_locations: List[np.ndarray] = []
        self._wall_locations: List[np.ndarray] = []
        self._occupied_locations: List[np.ndarray] = []
        self._update_locations()

        self._init_goal_pos: np.ndarray = np.array(self._config.init_goal_pos)
        self._goal: np.ndarray = self._reset_target(self._init_goal_pos)
        if self.config.use_adversary:
            self._init_adversary_pos: np.ndarray = np.array(
                self._config.init_adversary_pos
            )
            self._adversary: np.ndarray = self._reset_adversary(
                self._init_adversary_pos
            )

        # If we're using ref, verify the wall locations are the same and copy
        # block names
        if ref is not None:
            assert np.array_equal(
                self._wall_locations, ref._wall_locations
            ), f"Wall locations for {self.name} must be the same with reference {ref.name}."

    def _update_locations(self):
        def _is_wall(struct: str) -> bool:
            if struct == WALL:
                return True
            elif len(struct) > 1 and struct[0] == W:
                assert (
                    struct[1] == ":" and len(struct) > 2
                ), f"Invalid wall format: {struct}"
                return True
            else:
                return False

        for i in range(self.map_length):
            for j in range(self.map_width):
                struct = self.map[i][j]

                # Store cell locations in simulation global Cartesian coordinates
                x = (j + 0.5) * self.size_scaling - self.x_map_center
                y = self.y_map_center - (i + 0.5) * self.size_scaling

                if _is_wall(struct):
                    self._wall_locations.append(np.array([x, y]))

                    wall_texture = struct[2:] if len(struct) > 2 else "default"
                    assert (
                        wall_texture in self.config.wall_texture_map
                    ), f"Invalid texture: {wall_texture}"
                    self._wall_textures.append(wall_texture)
                elif struct == RESET:
                    self._unique_reset_locations.append(np.array([x, y]))
                elif struct == TARGET:
                    self._unique_target_locations.append(np.array([x, y]))
                elif struct == COMBINED:
                    self._combined_locations.append(np.array([x, y]))
                elif struct == EMPTY:
                    self._empty_locations.append(np.array([x, y]))
                elif struct == ADVERSARY:
                    self._unique_adv_locations.append(np.array([x, y]))

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
        # If we're using a reference maze, return an empty xml
        if self._ref is not None:
            return MjCambrianXML.make_empty()

        xml = MjCambrianXML.from_string(self._config.xml)
        worldbody = xml.find(".//worldbody")
        assert worldbody is not None
        assets = xml.find(".//asset")
        assert assets is not None

        # Create the wall textures
        for name, textures in self.config.wall_texture_map.items():
            for tex in textures:
                xml.add(
                    assets,
                    "texture",
                    name=f"wall_{self._name}_{name}_{tex}_tex",
                    file=f"maze_textures/{tex}.png",
                    gridsize="3 4",
                    gridlayout=".U..LFRB.D..",
                )
                xml.add(
                    assets,
                    "material",
                    name=f"wall_{self._name}_{name}_{tex}_mat",
                    texture=f"wall_{self._name}_{name}_{tex}_tex",
                )

        # Add the walls. Each wall has it's own geom.
        size_scaling = self.size_scaling
        for i, ((x, y), t) in enumerate(zip(self._wall_locations, self._wall_textures)):
            scale = size_scaling / 2
            name = f"block_{self._name}_{i}"
            xml.add(
                worldbody,
                "geom",
                name=name,
                pos=f"{x} {y} {scale * self.map_height}",
                size=f"{scale} {scale} {scale * self.map_height}",
                **{"class": f"maze_block_{self._name}"},
            )

        # Add the goal/adversary sites
        # Set their positions as placeholders, we'll update them later
        target_tex = "maze_textures/vertical_square_20.png"  # TODO add to config
        tex_kw = dict(file=target_tex) if self.config.use_adversary else None
        self._add_target(
            xml, self.goal_name, tex_kw=tex_kw, top_mat_kw=dict(rgba="0 1 0 1")
        )
        if self.config.use_adversary:
            self._add_target(
                xml,
                self.adversary_name,
                tex_kw=tex_kw,
                site_kw=dict(
                    euler="0 90 0",
                ),
                top_mat_kw=dict(rgba="1 0 0 1"),
            )

        # Update the floor texture to repeat in a way that matches the blocks
        floor_mat_name = f"floor_mat_{self._name}"
        floor_mat = xml.find(f".//material[@name='{floor_mat_name}']")
        assert floor_mat is not None, f"`{floor_mat_name}` not found"
        floor_mat.attrib["texrepeat"] = f"{2 / size_scaling} {2 / size_scaling}"

        # Update floor size
        floor_name = f"floor_{self._name}"
        floor = xml.find(f".//geom[@name='{floor_name}']")
        assert floor is not None, f"`{floor_name}` not found"
        floor.attrib["size"] = f"{self.map_width_scaled} {self.map_length_scaled} 0.1"
        floor.attrib["pos"] = f"{self.x_map_center} {self.y_map_center} -0.05"

        return xml

    def _add_target(
        self,
        xml: MjCambrianXML,
        name: str,
        *,
        site_kw: Dict[str, Any] = {},
        mat_kw: Dict[str, Any] = {},
        tex_kw: Dict[str, Any] | None = {},
        top_mat_kw: Dict[str, Any] = {},
    ):
        worldbody = xml.find(".//worldbody")
        assert worldbody is not None
        assets = xml.find(".//asset")
        assert assets is not None

        # Create a body which we use to change the position of the target
        targetbody = xml.add(
            worldbody,
            "body",
            name=f"{name}_body",
            childclass=f"maze_target_{self._name}",
        )

        # Each target is represented as a site sphere with a material
        xml.add(
            targetbody,
            "site",
            name=f"{name}_site",
            size=f"{0.2 * self.size_scaling}",
            material=f"{name}_mat",
            **site_kw,
        )

        mat = xml.add(
            assets,
            "material",
            name=f"{name}_mat",
            rgba="1 1 1 1",
            **mat_kw,
        )

        if tex_kw is not None:
            tex_kw.setdefault("type", "2d")
            xml.add(
                assets,
                "texture",
                name=f"{name}_tex",
                **tex_kw,
            )
            mat.attrib.setdefault("texture", f"{name}_tex")

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
        top_mat = xml.add(
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
            size=f"{0.05 * self.size_scaling}",
            material=f"{name}_top_mat",
            pos=f"0 0 {0.2 * self.size_scaling}",
            group="3",  # any group > 2 will be hidden to the agents
        )

        if self.config.hide_targets:
            # If hide targets is true, set the target and top site's mat to be
            # transparent
            mat.attrib["rgba"] = "0 0 0 0"
            top_mat.attrib["rgba"] = "0 0 0 0"

    def reset(self, model: mj.MjModel, *, active: bool = True):
        """Resets the maze. Will generate a goal and update the site/geom in mujoco."""
        if active:
            self._occupied_locations.clear()

            self._reset_wall_textures(model)

            self._goal = self._reset_target(self._init_goal_pos)
            if self.config.use_adversary:
                assert (
                    len(self._unique_target_locations) > 1
                    or len(self._unique_adv_locations) > 0
                ), "Must have at least 2 unique target locations to use an adversary"
                self._adversary = self._reset_adversary(self._init_adversary_pos)

        self._update_target(model, self.goal_name, self.goal, active)
        if self.config.use_adversary:
            self._update_target(model, self.adversary_name, self.adversary, active)

    def _reset_wall_textures(self, model: mj.MjModel):
        """Helper method to reset the wall textures."""
        if self._ref is not None:
            self._ref._reset_wall_textures(model)
            return

        # create a dict from wall to texture
        _wall_to_dict: Dict = {}
        for i, t in zip(range(len(self._wall_locations)), self._wall_textures):
            wall_name = f"block_{self._name}_{i}"
            geom_id = get_geom_id(model, wall_name)
            assert geom_id != -1, f"`{wall_name}` geom not found"

            # Randomly select a texture for the wall
            if t not in _wall_to_dict:
                _wall_to_dict[t] = np.random.choice(self.config.wall_texture_map[t])
            texture = _wall_to_dict[t]
            material_name = f"wall_{self._name}_{t}_{texture}_mat"
            material_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MATERIAL, material_name)
            assert material_id != -1, f"`{material_name}` material not found"

            model.geom_matid[geom_id] = material_id

    def _reset_target(self, init_pos: Optional[np.ndarray]) -> np.ndarray:
        """Helper method to reset the target."""
        return init_pos if init_pos else self.generate_target_pos()

    def _reset_adversary(self, init_pos: Optional[np.ndarray]) -> np.ndarray:
        """Helper method to reset the adversary."""
        if len(self._unique_adv_locations):
            return init_pos if init_pos else self.generate_adv_pos()
        return init_pos if init_pos else self.generate_target_pos()

    def _update_target(
        self, model: mj.MjModel, name: str, pos: np.ndarray, active: bool
    ):
        body_id = get_body_id(model, f"{name}_body")
        assert body_id != -1, f"`{name}_body` body not found"
        model.body_pos[body_id] = [*pos, self.map_height * self.size_scaling // 2]

        site_id = get_site_id(model, f"{name}_site")
        assert site_id != -1, f"`{name}_site` site not found"

        model.body_conaffinity[body_id] = 1 if active else 0
        model.site_group[site_id] = 1 if active else 3

        if self.config.use_target_light_sources:
            light_id = get_light_id(model, f"{name}_light")
            assert light_id != -1, f"`{name}_light` light not found"
            model.light_attenuation[light_id] = get_attenuation(self.max_dim * 1.5)

            # Set the light to be active or not
            model.light_active[light_id] = active

    def _generate_pos(
        self, locations: List[np.ndarray], *, tries: int = 20
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
        raise ValueError(
            f"Could not generate a unique position. {tries} tries failed. "
            f"Occupied locations: {self._occupied_locations}. "
            f"Available locations: {locations}."
        )

    def generate_target_pos(self, *, add_as_occupied: bool = True) -> np.ndarray:
        """Generates a random target position for an env."""
        target_pos = self._generate_pos(self._unique_target_locations)
        if add_as_occupied:
            self._occupied_locations.append(target_pos)
        return target_pos

    def generate_adv_pos(self, *, add_as_occupied: bool = True) -> np.ndarray:
        """Generates a random adversary position for an env."""
        adv_pos = self._generate_pos(self._unique_adv_locations)
        if add_as_occupied:
            self._occupied_locations.append(adv_pos)
        return adv_pos

    def generate_reset_pos(self, *, add_as_occupied: bool = True) -> np.ndarray:
        """Generates a random reset position for an env."""
        reset_pos = self._generate_pos(self._unique_reset_locations)
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

        raise ValueError("No path found")

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
    def name(self) -> str:
        """Returns the name."""
        return self._name

    @property
    def ref(self) -> Type["MjCambrianMaze"] | None:
        """Returns the reference maze."""
        return self._ref

    @property
    def ref_name(self) -> str | None:
        """Returns the reference maze name."""
        return self._ref.name if self._ref else None

    @property
    def goal(self) -> np.ndarray:
        """Returns the goal."""
        return self._goal

    @property
    def goal_name(self) -> str:
        """Returns the goal name."""
        return self._ref.goal_name if self._ref else f"goal_{self._name}"

    @property
    def adversary(self) -> np.ndarray:
        """Returns the adversary."""
        return self._adversary

    @property
    def adversary_name(self) -> str:
        """Returns the adversary name."""
        return self._ref.adversary_name if self._ref else f"adversary_{self._name}"

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
    def min_dim(self) -> float:
        """Returns the min dimension."""
        return min(self.map_length_scaled, self.map_width_scaled)

    @property
    def ratio(self) -> float:
        """Returns the ratio."""
        return self.map_length_scaled / self.map_width_scaled

    @property
    def x_map_center(self) -> float:
        """Returns the x map center."""
        return self.map_width_scaled / 2 + self._x_center

    @property
    def y_map_center(self) -> float:
        """Returns the y map center."""
        return self.map_length_scaled / 2

    @property
    def lookat(self) -> np.ndarray:
        """Returns the lookat."""
        x = -self.x_map_center + self.map_width_scaled // 2 + self.map_width
        y = self.y_map_center - self.map_length_scaled // 2
        return np.array([x, y, 0])

    # ==================

    def __repr__(self) -> str:
        return f"MjCambrianMaze(name={self._name})"

    def __str__(self) -> str:
        return self.__repr__()


# ================================


class MjCambrianMazeStore:
    """This is a simple class to store a collection of mazes."""

    def __init__(self, maze_configs: Dict[str, MjCambrianMazeConfig], maze_selection_fn: MjCambrianMazeSelectionFn):
        self._mazes: Dict[str, MjCambrianMaze] = {}
        self._ref_mazes: Dict[str, MjCambrianMaze] = {}
        self._current_maze: MjCambrianMaze = None

        self._maze_selection_fn = maze_selection_fn

        self._create_mazes(maze_configs)

    def _create_mazes(self, maze_configs: Dict[str, MjCambrianMazeConfig]):
        for name, config in maze_configs.items():
            if name in self._mazes:
                # If the maze already exists, skip it
                continue

            if (ref := config.ref) and ref not in self._mazes:
                # If the reference maze is not in the store, create it first
                assert ref in maze_configs, (
                    f"Unrecognized reference maze {ref=} for maze {name}. "
                    f"Must be one of the following: {list(maze_configs.keys())}"
                )
                self._ref_mazes[ref] = MjCambrianMaze(maze_configs[ref])

            # Now create the maze
            self._mazes[name] = MjCambrianMaze(config, name, ref=self._ref_mazes.get(ref))

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the current maze."""
        xml = MjCambrianXML.make_empty()

        for maze in self._mazes.values():
            xml += maze.generate_xml()
            if (ref := maze.ref) and ref not in self._mazes:
                xml += ref.generate_xml()

        return xml

    def reset(self, model: mj.MjModel, *, all_active: bool = False):
        """Resets the mazes."""
        for maze in self._mazes.values():
            maze.reset(model, active=maze == self._current_maze or all_active)
            if (ref := maze.ref) and ref not in self._mazes:
                ref.reset(model, active=maze == self._current_maze or all_active)

        # Explicitly reset the current maze to ensure it's active
        self._current_maze.reset(model, active=True)

    @property
    def current_maze(self) -> MjCambrianMaze:
        """Returns the current maze."""
        return self._current_maze

    @property
    def maze_list(self) -> List[MjCambrianMaze]:
        """Returns the list of mazes."""
        return list(self._mazes.values())

    # ======================
    # Maze selection methods

    def select_maze(self, env: "MjCambrianEnv") -> MjCambrianMaze:
        """This should be called by the environment to select a maze."""
        maze = self._maze_selection_fn(self, env)
        self._current_maze = maze
        return maze

    def select_maze_random(self, _: "MjCambrianEnv") -> MjCambrianMaze:
        """Selects a maze at random."""
        return np.random.choice(self.maze_list)

    def select_maze_difficulty(
        self, env: "MjCambrianEnv", *, schedule: Optional[str] = None, **kwargs
    ) -> MjCambrianMaze:
        """Selects a maze based on a difficulty schedule.

        Keyword Args:
            schedule (Optional[str]): The schedule to use. One of "linear",
                "exponential", or "logistic". If None, the selection is proportional
                to the difficulty.

            lam_0 (Optional[float]): The lambda value at the start of the schedule.
                Unused if schedule is None.
            lam_n (Optional[float]): The lambda value at the end of the schedule.
                Unused if schedule is None.
        """

        def calc_scheduled_lambda(
            *,
            schedule: str,
            lam_0: Optional[float] = -2.0,
            lam_n: Optional[float] = 2.0,
        ):
            """Selects a maze based on a schedule."""
            assert lam_0 < lam_n, "lam_0 must be less than lam_n"

            steps_per_env = (
                env.config.training.total_timesteps // env.config.training.n_envs
            )

            # Compute the current step
            step = env.num_timesteps / steps_per_env

            # Compute the lambda value
            if schedule == "linear":
                lam = lam_0 + (lam_n - lam_0) * step
            elif schedule == "exponential":
                lam = lam_0 * (lam_n / lam_0) ** (step / env.config.training.n_envs)
            elif schedule == "logistic":
                lam = lam_0 + (lam_n - lam_0) / (
                    1 + np.exp(-2 * step / env.config.training.n_envs)
                )
            else:
                raise ValueError(f"Invalid schedule: {schedule}")
            return lam

        # Sort the mazes by difficulty
        sorted_mazes = sorted(self.maze_list, key=lambda maze: maze.config.difficulty)
        sorted_difficulty = np.array([maze.config.difficulty for maze in sorted_mazes])

        if schedule is not None:
            lam = calc_scheduled_lambda(schedule=schedule, **kwargs)
            p = np.exp(lam * np.array(sorted_difficulty) / sorted_difficulty.max())
        else:
            # If no schedule, the selection is proportional to the difficulty
            p = sorted_difficulty

        return np.random.choice(sorted_mazes, p=p / p.sum())

    def select_maze_cycle(self, env: "MjCambrianEnv") -> MjCambrianMaze:
        """Selects a maze based on a cycle."""
        pass


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("map", type=str)

    args = parser.parse_args()

    config = MjCambrianMazeConfig.load(args.map)
    t0 = time.time()
    maze = MjCambrianMaze(config)
    t1 = time.time()

    xml = maze.generate_xml()
    t2 = time.time()

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
