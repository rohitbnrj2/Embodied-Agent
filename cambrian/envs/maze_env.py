from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
    List,
    TypeAlias,
    Callable,
    Concatenate,
)
from enum import Enum
import itertools

import mujoco as mj
import numpy as np

from cambrian.envs.env import MjCambrianEnv
from cambrian.envs.object_env import (
    MjCambrianObjectEnv,
    MjCambrianObjectEnvConfig,
    MjCambrianObject,
)
from cambrian.animals.animal import MjCambrianAnimal
from cambrian.utils import get_geom_id, safe_index
from cambrian.utils.config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.cambrian_xml import MjCambrianXML


class MjCambrianMapEntity(Enum):
    """
    Enum representing different states in a grid.

    Attributes:
        RESET (str): Initial reset position of the agent.
        OBJECT (str): Possible object locations.
        WALL (str): Represents a wall in the grid. Can include texture IDs in the
            format "1:<texture id>".
        EMPTY (str): Represents an empty space in the grid.
    """

    RESET = "R"
    OBJECT = "X"
    WALL = "1"
    EMPTY = "0"

    @staticmethod
    def parse(value: str) -> Tuple[Enum, str]:
        """
        Parse a value to handle special formats like "1:<texture id>".

        Args:
            value (str): The value to parse.

        Returns:
            Tuple[Enum, str]: The parsed entity and the texture id if applicable.
        """
        if value.startswith("1:"):
            return MjCambrianMapEntity.WALL, value[2:]
        for entity in MjCambrianMapEntity:
            if value == entity.value:
                return entity, "default"
        raise ValueError(f"Unknown MjCambrianMapEntity: {value}")


@config_wrapper
class MjCambrianMazeConfig(MjCambrianBaseConfig):
    """Defines a map config. Used for type hinting.

    Attributes:
        xml (MjCambrianXML): The xml for the maze. This is the xml that will be
            used to create the maze.
        map (str): The map to use for the maze. It's a 2D array where
            each element is a string and corresponds to a "pixel" in the map. See
            `maze.py` for info on what different strings mean. This is actually a
            List[List[str]], but we keep it as a string for readability when dumping
            the config to a file. Will convert to list when creating the maze.

        scale (float): The maze scaling for the continuous coordinates in the
            MuJoCo simulation.
        height (float): The height of the walls in the MuJoCo simulation.
        flip (bool): Whether to flip the maze or not. If True, the maze will be
            flipped along the x-axis.

        wall_texture_map (Dict[str, List[str]]): The mapping from texture id to
            texture names. Textures in the list are chosen at random. If the list is of
            length 1, only one texture will be used. A length >= 1 is required.
            The keyword "default" is required for walls denoted simply as 1 or W.
            Other walls are specified as 1/W:<texture id>.

        enabled_objects (Optional[List[str]]): The objects that are enabled in the
            maze. If None, all objects are enabled.
    """

    xml: MjCambrianXML
    map: str

    scale: float
    height: float
    flip: bool

    wall_texture_map: Dict[str, List[str]]

    enabled_objects: Optional[List[str]] = None


MjCambrianMazeSelectionFn: TypeAlias = Callable[
    Concatenate[MjCambrianAnimal, Dict[str, Any], ...], float
]


@config_wrapper
class MjCambrianMazeEnvConfig(MjCambrianObjectEnvConfig):
    """
    mazes (Dict[str, MjCambrianMazeConfig]): The configs for the mazes. Each
        maze will be loaded into the scene and the animal will be placed in a maze
        at each reset.
    maze_selection_fn (MjCambrianMazeSelectionFn): The function to use to select
        the maze. The function will be called at each reset to select the maze
        to use. See `MjCambrianMazeSelectionFn` and `maze.py` for more info.
    """

    mazes: Dict[str, MjCambrianMazeConfig]
    maze_selection_fn: MjCambrianMazeSelectionFn


class MjCambrianMazeEnv(MjCambrianObjectEnv):
    def __init__(self, config: MjCambrianMazeEnvConfig, **kwargs):
        self._config = config

        # Have to initialize the mazes first since generate_xml is called from the
        # MjCambrianEnv constructor
        self._maze: MjCambrianMaze = None
        self._maze_store = MjCambrianMazeStore(config.mazes, config.maze_selection_fn)

        super().__init__(config, **kwargs)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = MjCambrianXML.make_empty()

        # Add the mazes to the xml
        # Do this first so overrides defined in the env xml are applied
        xml += self._maze_store.generate_xml()

        # Add the rest of the xml
        xml += super().generate_xml()

        return xml

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[Any, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        # Set the random seed first
        if seed is not None:
            self.set_random_seed(seed)

        # Choose the maze
        self._maze = self._maze_store.select_maze(self)
        self._maze_store.reset(self.model)

        # For each animal, generate an initial position
        for animal in self.animals.values():
            animal.init_pos = self._maze.generate_reset_pos()

        # For each object, generate an initial position
        enabled_objects = self._maze.config.enabled_objects
        for obj in self.objects.values():
            if enabled_objects is None or obj.name in enabled_objects:
                if init_pos := obj.config.pos:
                    # If the initial pos is set in the config, we'll interpret it as
                    # a position in the maze and convert it to global coordinates
                    obj.pos[:2] = self._maze.rowcol_to_xy(list(init_pos)[:2])
                    obj.pos[2] = init_pos[2]
                else:
                    # If the initial object pos is not set, generate a new one
                    obj.pos[:2] = self._maze.generate_object_pos()
                    obj.pos[2] = self._maze.config.scale / 4.0

        # Now reset the environment
        obs, info = super().reset(seed=seed, options=options)

        if (renderer := self.renderer) and (viewer := renderer.viewer):
            # Update the camera positioning to match the current maze
            # Only update if the camera lookat is not set in the config file
            if viewer.config.select("camera.lookat") is None:
                viewer.camera.lookat = self._maze.lookat

            # Update the camera distance to match the current maze's extent
            viewer.camera.distance = viewer.config.select(
                "camera.distance", default=1.25
            )
            if self._maze.ratio < 2:
                viewer.camera.distance *= renderer.ratio * self._maze.min_dim
            else:
                viewer.camera.distance *= self._maze.max_dim / renderer.ratio

        return obs, info

    # ==================

    @property
    def maze(self) -> "MjCambrianMaze":
        """Returns the current maze."""
        return self._maze

    @maze.setter
    def maze(self, maze: "MjCambrianMaze"):
        """Sets the current maze."""
        self._maze = maze

    @property
    def maze_store(self) -> "MjCambrianMazeStore":
        """Returns the maze store."""
        return self._maze_store


# ================


class MjCambrianMaze:
    """The maze class. Generates a maze from a given map and provides utility
    functions for working with the maze."""

    def __init__(self, config: MjCambrianMazeConfig, name: str):
        self._config = config
        self._name = name
        self._starting_x = None

        self._map: np.ndarray = None
        self._load_map()

        self._wall_textures: List[str] = []
        self._wall_locations: List[np.ndarray] = []
        self._reset_locations: List[np.ndarray] = []
        self._object_locations: List[np.ndarray] = []
        self._occupied_locations: List[np.ndarray] = []

    def initialize(self, starting_x: float):
        self._starting_x = starting_x
        self._update_locations()

    def _load_map(self):
        """Parses the map (which is a str) as a yaml str and converts it to an
        np array."""
        import yaml

        self._map = np.array(yaml.safe_load(self._config.map), dtype=str)
        if self._config.flip:
            self._map = np.flip(self._map)

    def _update_locations(self):
        """This helper method will update the initially place the wall and reset
        locations. These are known at construction time. It will also parse wall
        textures."""

        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                struct = self._map[i][j]

                # Calculate the cell location in global coords
                x = (j + 0.5) * self._config.scale - self.x_map_center
                y = self.y_map_center - (i + 0.5) * self._config.scale
                loc = np.array([x, y])

                entity, texture_id = MjCambrianMapEntity.parse(struct)
                if entity == MjCambrianMapEntity.WALL:
                    self._wall_locations.append(loc)

                    # Do a check for the texture
                    assert texture_id in self._config.wall_texture_map, (
                        f"Invalid texture: {texture_id}. "
                        f"Available textures: {self._config.wall_texture_map.keys()}"
                    )
                    self._wall_textures.append(texture_id)
                elif entity == MjCambrianMapEntity.RESET:
                    self._reset_locations.append(loc)
                elif entity == MjCambrianMapEntity.OBJECT:
                    self._object_locations.append(loc)

    def generate_xml(self) -> MjCambrianXML:
        xml = MjCambrianXML.from_string(self._config.xml)

        worldbody = xml.find(".//worldbody")
        assert worldbody is not None, "xml must have a worldbody tag"
        assets = xml.find(".//asset")
        assert assets is not None, "xml must have an asset tag"

        # Add the wall textures
        for t, textures in self._config.wall_texture_map.items():
            for texture in textures:
                name_prefix = f"wall_{self._name}_{t}_{texture}"
                xml.add(
                    assets,
                    "material",
                    name=f"{name_prefix}_mat",
                    texture=f"{name_prefix}_tex",
                )
                xml.add(
                    assets,
                    "texture",
                    name=f"{name_prefix}_tex",
                    file=f"maze_textures/{texture}.png",
                    gridsize="3 4",
                    gridlayout=".U..LFRB.D..",
                )

        # Add the walls. Each wall has it's own geom.
        scale = self._config.scale / 2
        height = self._config.height
        for i, (x, y) in enumerate(self._wall_locations):
            name = f"wall_{self._name}_{i}"
            # Set the contype != conaffinity so walls don't collide with each other
            xml.add(
                worldbody,
                "geom",
                name=name,
                pos=f"{x} {y} {scale * height}",
                size=f"{scale} {scale} {scale * height}",
                contype="1",
                conaffinity="2",
                **{"class": f"maze_wall_{self._name}"},
            )

        # Update floor size based on the map extent
        # Only done if the size is explicitly set to 0 0 0
        floor_name = f"floor_{self._name}"
        floor = xml.find(f".//geom[@name='{floor_name}']")
        assert floor is not None, f"`{floor_name}` not found"
        if floor.attrib.get("size", "0 0 0"):
            size = f"{self.map_width_scaled // 2} {self.map_length_scaled // 2} 0.1"
            floor.attrib["size"] = size
        floor.attrib["pos"] = " ".join(map(str, [-self._starting_x, 0, -0.05]))

        return xml

    def reset(self, model: mj.MjModel, *, reset_occupied: bool = True):
        """Resets the maze. Will reset the wall textures and reset the occupied
        locations, if desired."""
        if reset_occupied:
            self._occupied_locations.clear()

        self._reset_wall_textures(model)

    def _reset_wall_textures(self, model: mj.MjModel):
        """Helper method to reset the wall textures.

        All like-labelled walls will have the same texture. Their textures will be
        randomly selected from their respective texture lists.
        """

        # First, generate the texture_id -> texture_name mapping
        texture_map: Dict[str, str] = {}
        for t in self._wall_textures:
            if t not in texture_map:
                texture_map[t] = np.random.choice(
                    list(self._config.wall_texture_map[t])
                )

        # Now, update the wall textures
        for i, t in zip(range(len(self._wall_locations)), self._wall_textures):
            wall_name = f"wall_{self._name}_{i}"
            geom_id = get_geom_id(model, wall_name)
            assert geom_id != -1, f"`{wall_name}` geom not found"

            # Randomly select a texture for the wall
            material_name = f"wall_{self._name}_{t}_{texture_map[t]}_mat"
            material_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_MATERIAL, material_name)
            assert material_id != -1, f"`{material_name}` material not found"

            # Update the geom material
            model.geom_matid[geom_id] = material_id

    # ==================

    def rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        x = (rowcol_pos[1] + 0.5) * self._config.scale - self.x_map_center
        y = self.y_map_center - (rowcol_pos[0] + 0.5) * self._config.scale

        return np.array([x, y])

    def xy_to_rowcol(self, xy_pos: np.ndarray) -> np.ndarray:
        i = np.floor((self.y_map_center - xy_pos[1]) / self._config.scale)
        j = np.floor((xy_pos[0] + self.x_map_center) / self._config.scale)
        return np.array([i, j], dtype=int)

    def compute_optimal_path(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Computes the optimal path from the start position to the target.

        Uses a BFS to find the shortest path.
        """
        from typing import Deque

        start = self.xy_to_rowcol(start)
        target = self.xy_to_rowcol(target)

        rows = self._map.shape[0]
        cols = self._map.shape[1]
        visited = [[False for _ in range(cols)] for _ in range(rows)]
        visited[start[0]][start[1]] = True
        queue = Deque([([start], 0)])  # (path, distance)

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while queue:
            path, dist = queue.popleft()
            current = path[-1]
            if np.all(current == target):
                # Convert path from indices to positions
                path = [self.rowcol_to_xy(pos) for pos in path]
                path.append(self.rowcol_to_xy(target))
                return np.array(path)

            # Check all moves (left, right, up, down, and all diagonals)
            for dr, dc in moves:
                r, c = current[0] + dr, current[1] + dc
                map_entity = MjCambrianMapEntity.parse(self._map[r][c])[0]
                if (
                    0 <= r < rows
                    and 0 <= c < cols
                    and not visited[r][c]
                    and map_entity != MjCambrianMapEntity.WALL
                ):
                    # If the movement is diagonal, check that the adjacent cells are
                    # free as well so the path doesn't clip through walls
                    pr, pc = current[0], current[1]
                    if (dr, dc) in moves[4:]:
                        pc_map_entity = MjCambrianMapEntity.parse(self._map[r][pc])[0]
                        pr_map_entity = MjCambrianMapEntity.parse(self._map[pr][c])[0]
                        if (
                            pc_map_entity == MjCambrianMapEntity.WALL
                            or pr_map_entity == MjCambrianMapEntity.WALL
                        ):
                            continue

                    visited[r][c] = True
                    queue.append((path + [(r, c)], dist + 1))

        raise ValueError("No path found")

    # ==================

    def _generate_pos(
        self,
        locations: List[np.ndarray],
        add_as_occupied: bool = True,
        tries: int = 20,
    ) -> np.ndarray:
        """Helper method to generate a position. The generated position must be at a
        unique location from self._occupied_locations.

        Args:
            locations (List[np.ndarray]): The locations to choose from.
            add_as_occupied (bool): Whether to add the chosen location to the
                occupied locations. Defaults to True.
            tries (int): The number of tries to attempt to find a unique position.
                Defaults to 20.

        Returns:
            np.ndarray: The chosen position. Is of size (2,).
        """
        assert len(locations) > 0, "Not enough locations to choose from"

        for _ in range(tries):
            idx = np.random.randint(low=0, high=len(locations))
            pos = locations[idx].copy()

            # Check if the position is already occupied
            for occupied in self._occupied_locations:
                if np.linalg.norm(pos - occupied) <= 0.5 * self._config.scale:
                    break
            else:
                if add_as_occupied:
                    self._occupied_locations.append(pos)
                return pos
        raise ValueError(
            f"Could not generate a unique position. {tries} tries failed. "
            f"Occupied locations: {self._occupied_locations}. "
            f"Available locations: {locations}."
        )

    def generate_reset_pos(self, *, add_as_occupied: bool = True) -> np.ndarray:
        """Generates a random reset position for an agent.

        Returns:
            np.ndarray: The chosen position. Is of size (2,).
        """
        return self._generate_pos(self._reset_locations, add_as_occupied)

    def generate_object_pos(self, *, add_as_occupied: bool = True) -> np.ndarray:
        """Generates a random object position.

        Returns:
            np.ndarray: The chosen position. Is of size (2,).
        """
        return self._generate_pos(self._object_locations, add_as_occupied)

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
    def map(self) -> np.ndarray:
        """Returns the map."""
        return self._map

    @property
    def map_length_scaled(self) -> float:
        """Returns the map length scaled."""
        return self._map.shape[0] * self._config.scale

    @property
    def map_width_scaled(self) -> float:
        """Returns the map width scaled."""
        return self._map.shape[1] * self._config.scale

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
        """Returns the ratio of the length over width."""
        return self.map_length_scaled / self.map_width_scaled

    @property
    def x_map_center(self) -> float:
        """Returns the x map center."""
        assert self._starting_x is not None, "Maze has not been initialized"
        return self.map_width_scaled // 2 + self._starting_x

    @property
    def y_map_center(self) -> float:
        """Returns the y map center."""
        return self.map_length_scaled / 2

    @property
    def lookat(self) -> np.ndarray:
        """Returns a point which aids in placement of a camera to visualize this maze."""
        # NOTE: Negative because of convention based on BEV camera
        assert self._starting_x is not None, "Maze has not been initialized"
        return np.array([-self._starting_x + len(self._map[0]) / 2, 0, 0])

    @property
    def object_locations(self) -> List[np.ndarray]:
        """Returns the object locations."""
        return self._object_locations

    @property
    def reset_locations(self) -> List[np.ndarray]:
        """Returns the reset locations."""
        return self._reset_locations


# ================================


class MjCambrianMazeStore:
    """This is a simple class to store a collection of mazes."""

    def __init__(
        self,
        maze_configs: Dict[str, MjCambrianMazeConfig],
        maze_selection_fn: MjCambrianMazeSelectionFn,
    ):
        self._mazes: Dict[str, MjCambrianMaze] = {}
        self._create_mazes(maze_configs)

        self._current_maze: MjCambrianMaze = None
        self._maze_selection_fn = maze_selection_fn

    def _create_mazes(self, maze_configs: Dict[str, MjCambrianMazeConfig]):
        prev_x, prev_width = 0, 0
        for name, config in maze_configs.items():
            if name in self._mazes:
                # If the maze already exists, skip it
                continue

            # First create the maze
            maze = MjCambrianMaze(config, name)
            self._mazes[name] = maze

            # Calculate the starting x of the maze
            # We'll place the maze such that it doesn't overlap with existing mazes
            # It'll be placed next to the previous one
            # The positions of the maze is calculated from one corner (defined as x
            # in this case)
            x = prev_x + prev_width / 2 + maze.map_width_scaled / 2
            maze.initialize(x)

            # Update the prev_center and prev_width
            prev_x, prev_width = x, maze.map_width_scaled

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the current maze."""
        xml = MjCambrianXML.make_empty()

        for maze in self._mazes.values():
            xml += maze.generate_xml()

        return xml

    def reset(self, model: mj.MjModel):
        """Resets all mazes."""
        for maze in self._mazes.values():
            maze.reset(model)

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

    def select_maze_schedule(
        self,
        env: "MjCambrianEnv",
        *,
        schedule: Optional[str] = "linear",
        total_timesteps: int,
        n_envs: int,
        lam_0: Optional[float] = -2.0,
        lam_n: Optional[float] = 2.0,
    ) -> MjCambrianMaze:
        """Selects a maze based on a schedule. The scheduled selections are based on
        the order of the mazes in the list.

        Keyword Args:
            schedule (Optional[str]): The schedule to use. One of "linear",
                "exponential", or "logistic". Defaults to "linear".

            total_timesteps (int): The total number of timesteps in the training
                schedule. Unused if schedule is None. Required otherwise.
            n_envs (int): The number of environments. Unused if schedule is None.
                Required otherwise.
            lam_0 (Optional[float]): The lambda value at the start of the schedule.
                Unused if schedule is None.
            lam_n (Optional[float]): The lambda value at the end of the schedule.
                Unused if schedule is None.
        """

        assert lam_0 < lam_n, "lam_0 must be less than lam_n"

        # Compute the current step
        steps_per_env = total_timesteps // n_envs
        step = env.num_timesteps / steps_per_env

        # Compute the lambda value
        if schedule == "linear":
            lam = lam_0 + (lam_n - lam_0) * step
        elif schedule == "exponential":
            lam = lam_0 * (lam_n / lam_0) ** (step / n_envs)
        elif schedule == "logistic":
            lam = lam_0 + (lam_n - lam_0) / (1 + np.exp(-2 * step / n_envs))
        else:
            raise ValueError(f"Invalid schedule: {schedule}")

        p = np.exp(lam * np.arange(len(self.maze_list)))
        return np.random.choice(self.maze_list, p=p / p.sum())

    def select_maze_cycle(self, env: "MjCambrianEnv") -> MjCambrianMaze:
        """Selects a maze based on a cycle."""
        idx = safe_index(self.maze_list, self._current_maze, default=-1)
        return self.maze_list[(idx + 1) % len(self.maze_list)]

    def select_maze_cycle_objects(
        self,
        env: MjCambrianMazeEnv,
        *,
        max_permutations_per_maze: Optional[int] = None,
        _return_permutations: bool = False,
    ) -> MjCambrianMaze:
        """This selection function will cycle through all combinations of enabled
        objects in each maze.

        Basically, it will cycle through each maze. And for each maze, it will set the
        initial position of each enabled objects such that all permutations of the
        placement of the objects are covered.
        """
        # Current maze index
        maze_idx: int = safe_index(self.maze_list, self._current_maze, default=0)
        maze = self.maze_list[maze_idx]

        # Get all the enabled objects and position locations
        objects: List[MjCambrianObject] = []
        positions: List[np.ndarray] = []
        for obj in env.objects.values():
            # Only consider enabled objects
            if maze.config.enabled_objects is not None:
                if obj.name not in maze.config.enabled_objects:
                    continue

            # Calculate the object's position. The first iteration, it may be None or
            # have None values.
            pos = None
            if obj.config.pos is not None and any(p for p in obj.config.pos[:2]):
                # TODO: Does this support pos with some, but not all, Nones?
                pos = obj.config.pos[:2]

            objects.append(obj)
            positions.append(pos)

        # Calculate all permutations
        object_locations = [
            maze.xy_to_rowcol(p).tolist() for p in maze.object_locations
        ]
        permutations = list(itertools.permutations(object_locations, len(positions)))
        if _return_permutations:
            return permutations

        # Get the current permutation
        idx = safe_index(permutations, tuple(positions), default=-1) + 1
        max_num_permutations = max_permutations_per_maze or len(positions)
        if idx == max_num_permutations:
            # Reset the object positions
            for obj in objects:
                with obj.config.set_readonly_temporarily(False):
                    obj.config.pos = None

            # If we've reached the end of the permutations, move to the next maze
            self._current_maze = self.maze_list[(maze_idx + 1) % len(self.maze_list)]
            return self.select_maze_cycle_objects(
                env, max_permutations_per_maze=max_permutations_per_maze
            )

        # Otherwise, update the object positions
        permutation = permutations[idx]
        for obj, pos in zip(objects, permutation):
            with obj.config.set_readonly_temporarily(False):
                if obj.config.pos is not None:
                    obj.config.pos[:2] = pos
                else:
                    z = maze.config.scale / 4.0
                    obj.config.pos = [*pos, z]

        return maze

    def select_maze_cycle_resets(
        self,
        env: MjCambrianMazeEnv,
        *,
        max_permutations_per_maze: Optional[int] = None,
        _return_permutations: bool = False,
    ) -> MjCambrianMaze:
        """This selection function will cycle through all combinations of reset
        positions in each maze.

        Basically, it will cycle through each maze. And for each maze, it will set the
        initial position of the agent such that all permutations of the placement of the
        agent are covered.
        """
        # Current maze index
        maze_idx: int = safe_index(self.maze_list, self._current_maze, default=0)
        maze = self.maze_list[maze_idx]

        # Get the current reset positions
        animals: List[MjCambrianAnimal] = []
        resets: List[np.ndarray] = []
        for animal in env.animals.values():
            pos = None
            if animal.init_pos is not None and any(p for p in animal.init_pos[:2]):
                # TODO: Does this support pos with some, but not all, Nones?
                pos = maze.xy_to_rowcol(animal.init_pos[:2]).tolist()

            animals.append(animal)
            resets.append(pos)

        # Calculate all permutations
        reset_locations = [maze.xy_to_rowcol(p).tolist() for p in maze.reset_locations]
        permutations = list(itertools.permutations(reset_locations, len(resets)))
        if _return_permutations:
            return permutations

        # Get the current permutation
        idx = safe_index(permutations, tuple(resets), default=-1) + 1
        max_num_permutations = max_permutations_per_maze or len(resets)
        if idx >= max_num_permutations:
            # Reset the agent positions
            for animal in animals:
                animal.init_pos = None

            # If we've reached the end of the permutations, move to the next maze
            self._current_maze = self.maze_list[(maze_idx + 1) % len(self.maze_list)]
            return self.select_maze_cycle_resets(
                env, max_permutations_per_maze=max_permutations_per_maze
            )

        # Otherwise, update the agent position
        permutation = permutations[idx]
        for animal, pos in zip(animals, permutation):
            if animal.init_pos is not None:
                animal.init_pos[:2] = pos
            else:
                x, y = maze.rowcol_to_xy(pos)
                animal.init_pos = [x, y]

        return maze
