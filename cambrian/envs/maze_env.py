from typing import Dict, Any, Tuple, Optional

from cambrian.maze import MjCambrianMaze, MjCambrianMazeStore
from cambrian.envs.env import MjCambrianEnv
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.config import MjCambrianConfig

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

