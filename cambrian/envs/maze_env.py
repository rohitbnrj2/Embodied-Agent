from typing import Dict, Any, Tuple, Optional, List

from cambrian.maze import MjCambrianMaze, MjCambrianMazeStore
from cambrian.envs.env import MjCambrianEnv
from cambrian.utils.cambrian_xml import MjCambrianXML, MjCambrianXMLConfig
from cambrian.utils.config import MjCambrianConfig
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig

@config_wrapper
class MjCambrianMazeConfig(MjCambrianBaseConfig):
    """Defines a map config. Used for type hinting.

    Attributes:
        ref (Optional[str]): Reference to a named maze config. Used to share walls and
            other geometries/assets. A check will be done to ensure the walls are
            identical between configs.

        map (List[List[str]]): The map to use for the maze. It's a 2D array where
            each element is a string and corresponds to a "pixel" in the map. See
            `maze.py` for info on what different strings mean.
        xml (str): The xml for the maze. This is the xml that will be used to
            create the maze.

        difficulty (float): The difficulty of the maze. This is used to determine
            the selection probability of the maze when the mode is set to "DIFFICULTY".
            The value should be set between 0 and 100, where 0 is the easiest and 100
            is the hardest.

        size_scaling (float): The maze scaling for the continuous coordinates in the
            MuJoCo simulation.
        height (float): The height of the walls in the MuJoCo simulation.
        flip (bool): Whether to flip the maze or not. If True, the maze will be
            flipped along the x-axis.
        smooth_walls (bool): Whether to smooth the walls such that they are continuous
            appearing. This is an approximated as a spline fit to the walls.

        hide_targets (bool): Whether to hide the target or not. If True, the target
            will be hidden.
        use_target_light_sources (bool): Whether to use a target light sources or not.
            If False, the colored target sites will be used (e.g. a red sphere).
            Otherwise, a light source will be used. The light source is simply a spot
            light facing down.

        wall_texture_map (Dict[str, List[str]]): The mapping from texture id to
            texture names. Textures in the list are chosen at random. If the list is of
            length 1, only one texture will be used. A length >= 1 is required.
            The keyword "default" is required for walls denoted simply as 1 or W.
            Other walls are specified as 1/W:<texture id>.

        init_goal_pos (Optional[Tuple[float, float]]): The initial position of the
            goal in the maze. If unset, will be randomly generated.
        eval_goal_pos (Optional[Tuple[float, float]]): The evaluation position of the
            goal in the maze. If unset, will be randomly generated.

        use_adversary (bool): Whether to use an adversarial target or not. If
            True, a second target will be created which is deemed adversarial. Also,
            the target's will be given high frequency textures which correspond to
            whether a target is adversarial or the true goal. This is done in hopes of
            having the animal learn to see high frequency input.
        init_adversary_pos (Optional[Tuple[float, float]]): The initial position
            of the adversary target in the maze. If unset, will be randomly generated.
        eval_adversary_pos (Optional[Tuple[float, float]]): The evaluation
            position of the adversary target in the maze. If unset, will be randomly
            generated.
    """

    ref: Optional[str] = None

    map: Optional[str] = None
    xml: MjCambrianXMLConfig

    difficulty: float

    size_scaling: float
    height: float
    flip: bool
    smooth_walls: bool

    hide_targets: bool
    use_target_light_sources: bool

    wall_texture_map: Dict[str, List[str]]

    init_goal_pos: Optional[Tuple[float, float]] = None
    eval_goal_pos: Optional[Tuple[float, float]] = None

    use_adversary: bool
    init_adversary_pos: Optional[Tuple[float, float]] = None
    eval_adversary_pos: Optional[Tuple[float, float]] = None


class MjCambrianMazeEnv(MjCambrianEnv):
    def __init__(self, config: MjCambrianConfig):
        self.maze: MjCambrianMaze = None
        self.maze_store = MjCambrianMazeStore(
            self.config.env.mazes, self.config.env.maze_selection_fn
        )

        super().__init__(config)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = super().generate_xml()

        # Add the mazes to the xml
        xml += self.maze_store.generate_xml()

        return xml

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[Any, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        # If this is the first reset, reset all the mazes and set them all as active
        if self._num_resets == 0:
            self.maze_store.reset(self.model, all_active=True)

        # Choose the maze
        self.maze = self.maze_store.select_maze(self)
        self.maze_store.reset(self.model)

        # For each animal, generate an initial position
        for animal in self.animals.values():
            animal.config.initial_state = self.maze.generate_reset_pos()

        # Now reset the environment
        obs, info = super().reset(seed=seed, options=options)

        # Add the maze info to the info dict
        info["maze"] = {}
        info["maze"]["goal"] = self.maze.goal

        if self.renderer is not None:
            self.renderer.set_option("sitegroup", True, slice(None))
            self.renderer.set_option("geomgroup", True, slice(None))

            self.renderer.config.camera
            if self.renderer.config.camera.lookat is None:
                self.renderer.viewer.camera.lookat[:] = self.maze.lookat
            if self.renderer.config.camera_config.distance is None:
                if self.maze.ratio < 2:
                    distance = self.renderer.ratio * self.maze.min_dim
                else:
                    distance = self.maze.max_dim / self.renderer.ratio
                self.renderer.viewer.camera.distance = distance

