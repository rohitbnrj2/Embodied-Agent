from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import pickle

import numpy as np
import gymnasium as gym
import mujoco as mj
from gymnasium import spaces
from stable_baselines3.common.utils import set_random_seed

from cambrian.animal import MjCambrianAnimal, MjCambrianPointAnimal
from cambrian.maze import MjCambrianMaze
from cambrian.utils import get_include_path
from cambrian.utils.config import MjCambrianConfig, MjCambrianEnvConfig
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.logger import get_logger
from cambrian.renderer import (
    MjCambrianRenderer,
    MjCambrianViewerOverlay,
    MjCambrianTextViewerOverlay,
    MjCambrianImageViewerOverlay,
    MjCambrianSiteViewerOverlay,
    MjCambrianCursor,
    resize_with_aspect_fill,
    TEXT_HEIGHT,
    TEXT_MARGIN,
)


class MjCambrianEnv(gym.Env):
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
    - This is an overridden version of the MujocoEnv class. The two main differences is
    that we allow for /resetmultiple agents and use our own custom renderer. It also reduces
    the need to create temporary xml files which MujocoEnv had to load. It's essentially
    a copy of MujocoEnv with the two aforementioned major changes.

    Args:
        config_path (str | Path | MjCambrianConfig): The path to the config file or the
            config object itself.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: str | Path | MjCambrianConfig):
        self._setup_config(config)
        self.logger = get_logger(self.config)

        self.animals: Dict[str, MjCambrianAnimal] = {}
        self._create_animals()

        self.maze: MjCambrianMaze = None  # the chosen maze that's updated at each reset
        self._training_mazes: Dict[str, MjCambrianMaze] = {}
        self._eval_mazes: Dict[str, MjCambrianMaze] = {}
        self._create_mazes()

        self.xml = self.generate_xml()

        self.model = mj.MjModel.from_xml_string(self.xml.to_string())
        self.data = mj.MjData(self.model)

        self.renderer: MjCambrianRenderer = None
        self.render_mode = (
            "human" if "human" in self.renderer_config.render_modes else "rgb_array"
        )
        if self.env_config.use_renderer:
            self.renderer = MjCambrianRenderer(self.renderer_config)

        # Reset all the mazes and set as active once
        maze_names = self.maze_names
        for name, maze in self._maze_store.items():
            if name in maze_names:
                maze.reset(self.model, active=True)
            if (ref := maze._ref) and ref.name not in maze_names:
                ref.reset(self.model, active=True)

        self._episode_step = 0
        self._max_episode_steps = self.config.training_config.max_episode_steps
        self._num_timesteps = 0
        self._num_resets = 0
        self._cumulative_reward = 0

        self._record: bool = False
        self._rollout: Dict[str, Any] = {}
        self._overlays: Dict[str, Any] = {}

        self._reward_fn = self._get_reward_fn(self.env_config.reward_fn_type)

        # Used to store the optimal path for each animal in the maze
        # Each animal has a different start position so optimal path is different
        # Tuple[List, List] = [path, accumulated_path_lengths]
        self._optimal_animal_paths: Dict[str, Tuple[List, List]] = {}

    def _setup_config(self, config: str | Path | MjCambrianConfig):
        """Helper method to setup the config. This is called by the constructor."""
        self.config = (
            MjCambrianConfig.load(config) if isinstance(config, str) else config
        )
        self.env_config = self.config.env_config
        self.renderer_config = self.env_config.renderer_config

        assert "mode" in self.env_config.maze_selection_criteria

    def _create_animals(self):
        """Helper method to create the animals.

        Under the hood, the `create` method does the following:
            - load the base xml to MjModel
            - parse the geometry and place eyes at the appropriate locations
            - create the action/observation spaces

        TODO: Hardcoded to use MjCambrianPointAnimal for now!!
        """
        for animal_config in self.env_config.animal_configs.values():
            assert animal_config.name not in self.animals
            self.animals[animal_config.name] = MjCambrianPointAnimal(animal_config)

    def _create_mazes(self):
        """Helper method to create the mazes.

        NOTE: The keys for self._training_mazes and self._eval_mazes are f"{name}_{i}",
        not the actual name of the maze. This is to ensure that the keys are unique.
        """
        self._maze_store: Dict[str, MjCambrianMaze] = {}
        mazes_to_create = self.env_config.maze_configs
        if (eval_maze_configs := self.env_config.eval_maze_configs) is not None:
            mazes_to_create += eval_maze_configs
        for maze_name in mazes_to_create:
            if maze_name in self._maze_store:
                continue

            assert maze_name in self.env_config.maze_configs_store, (
                f"Unrecognized maze name {maze_name}. "
                "Must be one of the following: "
                f"{[m for m in self.env_config.maze_configs_store]}"
            )

            maze_config = self.env_config.maze_configs_store[maze_name]
            if ref := maze_config.ref:
                if ref not in self._maze_store:
                    assert ref in self.env_config.maze_configs_store, (
                        f"Unrecognized ref maze name {ref} for {maze_config.name}. "
                        "Must be one of the following: "
                        f"{[m for m in self.env_config.maze_configs_store]}"
                    )
                    self._maze_store[ref] = MjCambrianMaze(
                        self.env_config.maze_configs_store[ref]
                    )
                ref = self._maze_store[ref]

            self._maze_store[maze_name] = MjCambrianMaze(maze_config, ref=ref)

        self._training_mazes: Dict[str, MjCambrianMaze] = {
            f"{n}_{i}": self._maze_store[n]
            for i, n in enumerate(self.env_config.maze_configs)
        }

        self._eval_mazes: Dict[str, MjCambrianMaze] = {}
        if (eval_maze_configs := self.env_config.eval_maze_configs) is not None:
            self._eval_mazes: Dict[str, MjCambrianMaze] = {
                f"{n}_{i}": self._maze_store[n] for i, n in enumerate(eval_maze_configs)
            }

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = MjCambrianXML.from_string(self.env_config.xml)

        # Create the ambient light, if desired
        if self.env_config.use_ambient_light:
            assert self.env_config.ambient_light_intensity is not None
            xml.add(
                xml.find(".//worldbody"),
                "light",
                ambient=" ".join(map(str, self.env_config.ambient_light_intensity)),
                diffuse="0 0 0",
                specular="0 0 0",
                cutoff="180",
                castshadow="false",
            )

        # Disable the headlight
        if not self.env_config.use_headlight:
            xml.add(xml.add(xml.root, "visual"), "headlight", active="0")

        # Add the animals to the xml
        for idx, animal in enumerate(self.animals.values()):
            xml += animal.generate_xml(idx)

        # Add the mazes to the xml
        # We'll add only the mazes in the eval and training list, as well as the refs
        # if they're not already in the list
        maze_names = self.maze_names
        for name, maze in self._maze_store.items():
            if name in maze_names:
                xml += maze.generate_xml()
            if (ref := maze._ref) and ref.name not in maze_names:
                xml += ref.generate_xml()

        # Update the assert path to point to the fully resolved path
        compiler = xml.find(".//compiler")
        assert compiler is not None
        model_dir = Path("models")  # TODO: make config attr
        if (texturedir := compiler.attrib.get("texturedir")) is not None:
            texturedir = str(get_include_path(model_dir / texturedir))
            compiler.attrib["texturedir"] = texturedir
        if (meshdir := compiler.attrib.get("meshdir")) is not None:
            meshdir = str(get_include_path(model_dir / meshdir))
            compiler.attrib["meshdir"] = meshdir
        if (assetdir := compiler.attrib.get("assetdir")) is not None:
            assetdir = str(get_include_path(model_dir / assetdir))
            compiler.attrib["assetdir"] = assetdir

        return xml

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[Any, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Reset the environment.

        Will reset all underlying components (the maze, the animals, etc.). The
        simulation will then be stepped once to ensure that the observations are
        up-to-date.

        Returns:
            Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]: The observations for each
                animal and the info dict for each animal.
        """
        super().reset(seed=seed, options=options)
        if seed is not None:
            set_random_seed(seed)

        mj.mj_resetData(self.model, self.data)

        self.maze = self._choose_maze(**self.env_config.maze_selection_criteria)
        for maze in self.mazes:
            if maze.name != self.maze.name:
                if (ref := maze._ref) and ref.name == self.maze.name:
                    continue
                maze.reset(self.model, active=False)
        # reset explicitly such that the active maze is reset last to ensure that the
        # setup is done correctly
        self.maze.reset(self.model, active=True)

        info: Dict[str, Any] = {a: {} for a in self.animals}
        obs: Dict[str, Dict[str, Any]] = {}
        for name, animal in self.animals.items():
            init_qpos = (
                self.maze.cell_rowcol_to_xy(*animal.config.init_pos)
                if animal.config.init_pos
                else self.maze.generate_reset_pos()
            )

            obs[name] = animal.reset(self.model, self.data, init_qpos)
            if self.env_config.use_goal_obs:
                obs[name]["goal"] = self.maze.goal.copy()

            if self.env_config.compute_optimal_path:
                path = self.maze.compute_optimal_path(animal.pos, self.maze.goal)
                accum_path_len = np.cumsum(
                    np.linalg.norm(np.diff(path, axis=0), axis=1)
                )
                self._optimal_animal_paths[name] = (path, accum_path_len)

            info[name]["pos"] = animal.pos

        info["maze"] = {}
        info["maze"]["goal"] = self.maze.goal

        self._step_mujoco_simulation(1)

        if self.renderer is not None:
            if self._num_resets == 0:
                # If this is the first reset, then we need to reset the renderer
                # to populate the width and height fields to properly set the lookat
                self.renderer.reset(self.model, self.data)

            self.renderer.set_option("sitegroup", True, slice(None))
            self.renderer.set_option("geomgroup", True, slice(None))

            if self.renderer.config.camera_config.lookat is None:
                self.renderer.viewer.camera.lookat[:] = self.maze.lookat
            if self.renderer.config.camera_config.distance is None:
                if self.maze.ratio < 2:
                    distance = self.renderer.ratio * self.maze.min_dim
                else:
                    distance = self.maze.max_dim / self.renderer.ratio
                self.renderer.viewer.camera.distance = distance

            self.renderer.reset(self.model, self.data)

        self._episode_step = 0
        self._cumulative_reward = 0
        self._num_resets += 1
        if not self.record:
            self._overlays.clear()
            self._rollout.clear()

        if self.env_config.add_overlays:
            self._overlays["Exp"] = self.config.training_config.exp_name

        return obs, info

    def _choose_maze(
        self, mode: MjCambrianEnvConfig.MazeSelectionMode | str, **kwargs
    ) -> MjCambrianMaze:
        """Chooses a maze from the list of mazes base on the selection mode."""
        mode = MjCambrianEnvConfig.MazeSelectionMode[mode]
        training_mazes = list(self._training_mazes.values())
        training_maze_names = [m.name for m in training_mazes]

        if mode == MjCambrianEnvConfig.MazeSelectionMode.RANDOM:
            return np.random.choice(training_mazes)
        elif mode == MjCambrianEnvConfig.MazeSelectionMode.DIFFICULTY:
            # Sort the mazes by difficulty
            sorted_mazes = sorted(training_mazes, key=lambda m: m.config.difficulty)
            sorted_difficulty = np.array([m.config.difficulty for m in sorted_mazes])

            if schedule := kwargs.get("schedule", None):
                training_config = self.config.training_config
                steps_per_env = training_config.total_timesteps / training_config.n_envs
                t = self._num_timesteps / steps_per_env
                lam_0 = kwargs.get("lam_0", -2.0)  # initial lambda
                lam_n = kwargs.get("lam_n", 2.0)  # final lambda
                lam_range = lam_n - lam_0
                if schedule == "exponential":
                    lam_factor = kwargs.get("lam_factor", 5.0)
                    lam = lam_0 + lam_range * (1 - np.exp(-t * lam_factor))
                elif schedule == "logistic":
                    lam_factor = kwargs.get("lam_factor", 10.0)
                    ti = kwargs.get("ti", 0.3)  # inflection
                    lam = lam_0 + lam_range / (1 + np.exp(-lam_factor * (t - ti)))
                else:
                    # Linear
                    lam = lam_0 + (lam_n - lam_0) * t
                p = np.exp(lam * sorted_difficulty / sorted_difficulty.max())
            else:
                # Prob is proportional to the difficulty of the maze
                p = sorted_difficulty
            return np.random.choice(sorted_mazes, p=p / p.sum())
        elif mode == MjCambrianEnvConfig.MazeSelectionMode.CURRICULUM:
            # Sort mazes based on current reward (higher reward = harder maze)
            # We'll assume the max cumulative reward is max_episode_steps
            normalized_reward = min(
                self._cumulative_reward / self._max_episode_steps, 1
            )

            # calc probs for each maze
            difficulties = np.array([m.config.difficulty for m in training_mazes])
            p = np.exp(-np.abs(difficulties / 100 - normalized_reward))

            return np.random.choice(training_mazes, p=p / p.sum())
        elif mode == MjCambrianEnvConfig.MazeSelectionMode.NAMED:
            assert "name" in kwargs, "Must specify `name` if using NAMED selection mode"
            assert kwargs["name"] in training_maze_names, (
                f"Unrecognized maze name {kwargs['name']}. "
                "Must be one of the following: "
                f"{training_maze_names}"
            )
            return training_mazes[training_maze_names.index(kwargs["name"])]
        elif mode == MjCambrianEnvConfig.MazeSelectionMode.CYCLE:
            # Will cycle through the mazes in the order they were defined in the config
            # If maze is unset, will choose the first maze
            idx = -1 if self.maze is None else training_mazes.index(self.maze)
            return training_mazes[(idx + 1) % len(training_mazes)]
        elif mode == MjCambrianEnvConfig.MazeSelectionMode.EVAL:
            eval_mazes = list(self._eval_mazes.values())
            idx = self.env_config.maze_selection_criteria.setdefault("eval_idx", 0)
            eval_maze = eval_mazes[idx]
            self.env_config.maze_selection_criteria["eval_idx"] = (idx + 1) % len(
                eval_mazes
            )
            return eval_maze
        else:
            raise ValueError(f"Unrecognized maze selection mode {mode}")

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

        The dynamics is updated through the `do_simulation` method.

        Args:
            action (Dict[str, Any]): The action to take for each animal. The keys
                define the animal name, and the values define the action for that
                animal.

        Returns:
            Dict[str, Any]: The observations for each animal.
            Dict[str, float]: The reward for each animal.
            Dict[str, bool]: Whether each animal has terminated.
            Dict[str, bool]: Whether each animal has truncated.
            Dict[str, Dict[str, Any]]: The info dict for each animal.
        """
        info: Dict[str, Any] = {a: {} for a in self.animals}

        # First, apply the actions to the animals and step the simulation
        for name, animal in self.animals.items():
            animal.apply_action(action[name])
            info[name]["prev_pos"] = animal.pos

        # Then, step the mujoco simulation
        self._step_mujoco_simulation(self.env_config.frame_skip)

        # We'll then step each animal to render it's current state and get the obs
        obs: Dict[str, Any] = {}
        for name, animal in self.animals.items():
            obs[name] = animal.step()
            if self.env_config.use_goal_obs:
                obs[name]["goal"] = self.maze.goal.copy()

            if not animal.config.disable_intensity_sensor:
                info[name]["intensity"] = animal.intensity_sensor.last_obs
            info[name]["action"] = action[name]

        info["maze"] = {}
        info["maze"]["goal"] = self.maze.goal

        # Compute the reward, terminated, and truncated
        terminated = self._compute_terminated()
        truncated = self._compute_truncated()
        reward = self._compute_reward(terminated, truncated, info)

        self._episode_step += 1
        self._num_timesteps += 1
        self._cumulative_reward += sum(reward.values())

        if self.env_config.add_overlays:
            self._overlays["Step"] = self._episode_step
            self._overlays["Cumulative Reward"] = round(self._cumulative_reward, 2)

            # Add the position of each animal to the overlays
            for animal in self.animals.values():
                self._overlays[
                    f"{animal.name}_pos_{self._episode_step}"
                ] = MjCambrianSiteViewerOverlay(
                    self.renderer.viewer.scene, [*animal.pos, 0.1]
                )

        if self.record:
            self._rollout.setdefault("actions", [])
            self._rollout["actions"].append(list(action.values()))
            self._rollout.setdefault("positions", [])
            self._rollout["positions"].append([a.pos for a in self.animals.values()])

        return obs, reward, terminated, truncated, info

    def _step_mujoco_simulation(self, n_frames):
        """Sets the mujoco simulation. Will step the simulation `n_frames` times, each
        time checking if the animal has contacts. If so, will break early (if the
        environment is configured to truncate on contact)."""
        # Check contacts at _every_ step.
        # NOTE: Doesn't process whether hits are terminal or not
        for _ in range(n_frames):
            mj.mj_step(self.model, self.data)

            # TODO: don't break here since it will effect the other animals. Instead,
            # have a `should_terminate` flag or something.
            if self.env_config.truncate_on_contact and self.data.ncon > 0:
                if any(animal.has_contacts for animal in self.animals.values()):
                    break

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mj.mj_rnePostConstraint(self.model, self.data)

    def _compute_reward(
        self,
        terminated: Dict[str, bool],
        truncated: Dict[str, bool],
        info: Dict[str, bool],
    ) -> Dict[str, float]:
        """Computes the reward for the environment.

        Args:
            terminated (Dict[str, bool]): Whether each animal has terminated.
                Termination indicates success (agent has reached the goal).
            truncated (Dict[str, bool]): Whether each animal has truncated.
                Truncation indicates failure (agent has hit the wall or something).
            info (Dict[str, bool]): The info dict for each animal.
        """

        rewards: Dict[str, float] = {}
        for name, animal in self.animals.items():
            # Early exits
            if terminated[name]:
                rewards[name] = 1
                continue
            elif truncated[name]:
                rewards[name] = -1
                continue

            # Call reward_fn
            rewards[name] = self._reward_fn(animal, info[name])

            # Add a penalty for each action taken
            rewards[name] += self.env_config.action_penalty * np.sum(
                np.square(info[name]["action"])
            )

            # Add a penalty to the reward if the animal has contacts and
            # truncate_on_contact is False. We'll assume that the truncation value
            # corresponds correctly to whether contacts have been recorded, so no
            # need to check the truncate
            if animal.has_contacts:
                if self.env_config.force_exclusive_contact_penalty:
                    rewards[name] = self.env_config.contact_penalty
                else:
                    rewards[name] += self.env_config.contact_penalty

            # If we're using an adversarial target and we're at the adversary, then
            # give a reward of -1. We'll assume that the truncation value corresponds
            # correctly to whether we've reached the adversary, so no need to check
            # we need to truncate on contact
            # TODO: Should we terminate when at adversary? Above comment is incorrect
            if self.maze.config.use_adversary:
                if self._is_at_target(animal, self.maze.adversary):
                    rewards[name] += self.env_config.adversary_penalty

        return rewards

    def _compute_terminated(self) -> Dict[str, bool]:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure."""

        terminated: Dict[str, bool] = {}
        for name, animal in self.animals.items():
            if self.env_config.terminate_at_goal:
                terminated[name] = bool(self._is_at_goal(animal))
            else:
                terminated[name] = False

        return terminated

    def _compute_truncated(self) -> bool:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure. Failure, for now, indicates that the
        animal has touched the wall."""

        truncated: Dict[str, bool] = {}
        for name, animal in self.animals.items():
            over_max_steps = self._episode_step >= (self._max_episode_steps - 1)
            if self.env_config.truncate_on_contact:
                truncated[name] = animal.has_contacts or over_max_steps
            else:
                truncated[name] = over_max_steps

        return truncated

    def render(self) -> Dict[str, np.ndarray]:
        """Renders the environment.

        Returns:
            Dict[str, np.ndarray]: The rendered image for each render mode mapped to
                its str.

        TODO:
            - Make the cursor stuff clearer
        """

        assert self.renderer is not None, "Renderer has not been initialized! "
        "Ensure `use_renderer` is set to True in the constructor."

        renderer = self.renderer
        renderer_width = renderer.width
        renderer_height = renderer.height

        overlays: List[MjCambrianViewerOverlay] = []
        overlay_width = int(renderer_width * self.env_config.overlay_width)
        overlay_height = int(renderer_height * self.env_config.overlay_height)
        overlay_size = (overlay_width, overlay_height)

        cursor = MjCambrianCursor(x=0, y=renderer_height - TEXT_MARGIN * 2)
        for key, value in self._overlays.items():
            if issubclass(type(value), MjCambrianViewerOverlay):
                overlays.append(value)
            else:
                cursor.y -= TEXT_HEIGHT + TEXT_MARGIN
                overlays.append(MjCambrianTextViewerOverlay(f"{key}: {value}", cursor))

        if not self.env_config.add_overlays:
            return self.renderer.render(overlays=overlays)

        cursor = MjCambrianCursor(0, 0)
        for i, (name, animal) in enumerate(self.animals.items()):
            cursor.x += 2 * i * overlay_width
            cursor.y = 0
            if cursor.x + overlay_width * 2 > renderer_width:
                self.logger.warning("Renderer width is too small!!")
                continue

            composite = animal.create_composite_image()
            if composite is None:
                # Make the composite image black so we can still render other overlays
                composite = np.zeros((*overlay_size, 3), dtype=np.float32)
            if animal.config.disable_intensity_sensor:
                # Make the intensity image black so we can still render other overlays
                intensity = np.zeros(composite.shape, dtype=np.float32)
            else:
                intensity = animal.intensity_sensor.last_obs

            # NOTE: flipud here since we always flipud when copying buffer from gpu,
            # and when reading the buffer again after drawing the overlay, it will be
            # flipped again. Flipping here means it will be the right side up.
            new_composite = np.flipud(resize_with_aspect_fill(composite, *overlay_size))
            new_intensity = np.flipud(resize_with_aspect_fill(intensity, *overlay_size))

            overlays.append(MjCambrianImageViewerOverlay(new_composite * 255.0, cursor))

            cursor.x -= TEXT_MARGIN
            cursor.y -= TEXT_MARGIN
            overlay_text = f"Num Eyes: {len(animal.eyes)}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            cursor.y += TEXT_HEIGHT
            eye0 = next(iter(animal.eyes.values()))
            overlay_text = f"Res: {tuple(eye0.resolution)}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            cursor.y += TEXT_HEIGHT
            overlay_text = f"FOV: {tuple(eye0.fov)}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            cursor.y = overlay_height - TEXT_HEIGHT * 2 + TEXT_MARGIN * 2
            overlay_text = f"Animal: {name}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            overlay_text = f"Action: {[f'{a:0.3f}' for a in animal.last_action]}"
            cursor.y -= TEXT_HEIGHT
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))

            cursor.x += overlay_width
            cursor.y = 0

            overlays.append(MjCambrianImageViewerOverlay(new_intensity * 255.0, cursor))
            if not animal.config.disable_intensity_sensor:
                overlay_text = str(intensity.shape[:2])
                overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
                cursor.y = overlay_height - TEXT_HEIGHT * 2 + TEXT_MARGIN * 2
                overlay_text = animal.intensity_sensor.name
                overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))

        return renderer.render(overlays=overlays)

    @property
    def episode_step(self) -> int:
        """Returns the current episode step."""
        return self._episode_step

    @property
    def num_resets(self) -> int:
        """Returns the number of resets."""
        return self._num_resets

    @property
    def max_episode_steps(self) -> int:
        """Returns the max episode steps."""
        return self._max_episode_steps

    @property
    def overlays(self) -> Dict[str, Any]:
        """Returns the overlays."""
        return self._overlays

    @property
    def agents(self) -> List[str]:
        """Returns the agents in the environment.

        This is part of the PettingZoo API.
        """
        return list(self.agents.keys())

    @property
    def possible_agents(self) -> List[str]:
        """Returns the possible agents in the environment.

        This is part of the PettingZoo API.

        Assumes that the possible agents are the same as the agents.
        """
        return self.agents

    @property
    def observation_spaces(self) -> spaces.Dict:
        """Creates the observation spaces.

        This is part of the PettingZoo API.

        By default, this environment will support multi-animal
        observationsa/actions/etc. This method will create _all_ the obeservation
        spaces for the environment. But note that stable baselines3 only suppots single
        agent environments (i.e. non-nested spaces.Dict), so ensure you wrap this env
        with a `wrappers.MjCambrianSingleAnimalEnvWrapper` if you want to use stable
        baselines3.
        """

        # Create the observation_spaces
        observation_spaces: Dict[str, spaces.Space] = {}
        for name, animal in self.animals.items():
            observation_space: spaces.Dict = animal.observation_space
            if self.env_config.use_goal_obs:
                observation_space.spaces["goal"] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                )
            observation_spaces[name] = observation_space
        return spaces.Dict(observation_spaces)

    @property
    def action_spaces(self) -> spaces.Dict:
        """Creates the action spaces.

        This is part of the PettingZoo API.

        By default, this environment will support multi-animal
        observationsa/actions/etc. This method will create _all_ the action
        spaces for the environment. But note that stable baselines3 only suppots single
        agent environments (i.e. non-nested spaces.Dict), so ensure you wrap this env
        with a `wrappers.MjCambrianSingleAnimalEnvWrapper` if you want to use stable
        baselines3.
        """

        # Create the action_spaces
        action_spaces: Dict[str, spaces.Space] = {}
        for name, animal in self.animals.items():
            action_spaces[name] = animal.action_space
        return spaces.Dict(action_spaces)

    @property
    def record(self):
        """Returns whether the environment is recording."""
        return self._record

    @record.setter
    def record(self, value: bool):
        """Sets whether the environment is recording."""
        self._record = value
        self.renderer.record = value

        if not self.record:
            self._rollout.clear()

    @property
    def mazes(self) -> List[MjCambrianMaze]:
        """Returns the mazes as a list."""
        return [*self._training_mazes.values(), *self._eval_mazes.values()]

    @property
    def maze_names(self) -> List[str]:
        """Returns the maze names as a list."""
        training_maze_names = [m.name for m in self._training_mazes.values()]
        eval_maze_names = [m.name for m in self._eval_mazes.values()]
        return training_maze_names + eval_maze_names

    def save(self, path: str | Path):
        """Saves the simulation output to the given path."""
        self.renderer.save(path)

        self.logger.info(f"Saving rollout to {path.with_suffix('.pkl')}")
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self._rollout, f)
        self.logger.debug(f"Saved rollout to {path.with_suffix('.pkl')}")

    def _is_at_target(self, animal: MjCambrianAnimal, target: np.ndarray) -> bool:
        """Returns whether the animal is at the target."""
        return (
            np.linalg.norm(animal.pos - target)
            < self.env_config.distance_to_target_threshold
        )

    def _is_at_goal(self, animal: MjCambrianAnimal) -> bool:
        """Alias to _is_at_target(animal, self.maze.goal)"""
        return self._is_at_target(animal, self.maze.goal)

    # ================
    # Reward Functions

    def _get_reward_fn(self, reward_fn_type: str):
        assert reward_fn_type is not None, "reward_fn_type must be set"
        fn_name = f"_reward_fn_{reward_fn_type}"
        assert hasattr(self, fn_name), f"Unrecognized reward_fn_type {reward_fn_type}"
        return getattr(self, fn_name)

    def _reward_fn_euclidean(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """Rewards the euclidean distance to the goal."""
        current_distance_to_goal = np.linalg.norm(animal.pos - self.maze.goal)
        initial_distance_to_goal = np.linalg.norm(animal.init_pos - self.maze.goal)
        return 1 - current_distance_to_goal / initial_distance_to_goal

    def _reward_fn_euclidean_and_at_goal(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """This reward combines `reward_fn_euclidean` and `reward_fn_sparse`."""
        euclidean_reward = self._reward_fn_euclidean(animal, info)
        return 1 if self._is_at_goal(animal) else euclidean_reward

    def _reward_fn_delta_euclidean(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """Rewards the change in distance to the goal from the previous step."""
        return -self._calc_delta_pos(animal, info, self.maze.goal)

    def _reward_fn_delta_euclidean_and_at_goal(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """This reward combines `reward_fn_delta_euclidean` and `reward_fn_sparse`."""
        delta_euclidean_reward = self._reward_fn_delta_euclidean(animal, info)
        return 1 if self._is_at_goal(animal) else delta_euclidean_reward

    def _reward_fn_euclidean_delta_from_init(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """
        Rewards the change in distance over the previous step scaled by the timestep.
        """
        return self._calc_delta_pos(animal, info, animal.init_pos)

    def _reward_fn_delta_euclidean_w_movement(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """Same as delta_euclidean, but also rewards movement away from the initial
        position"""
        delta_distance_to_goal = self._calc_delta_pos(animal, info, self.maze.goal)
        delta_distance_from_init = np.linalg.norm(animal.init_pos - animal.pos)
        return np.clip(delta_distance_to_goal + delta_distance_from_init, 0, 1)

    def _reward_fn_distance_along_path(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """Rewards the distance along the optimal path to the goal."""
        path, accum_path_len = self._optimal_animal_paths[animal.name]
        idx = np.argmin(np.linalg.norm(path[:-1] - animal.pos, axis=1))
        return accum_path_len[idx] / accum_path_len[-1]

    def _reward_fn_delta_distance_along_path(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """Rewards the distance along the optimal path to the goal."""
        path, accum_path_len = self._optimal_animal_paths[animal.name]
        idx = np.argmin(np.linalg.norm(path[:-1] - animal.pos, axis=1))
        prev_idx = np.argmin(np.linalg.norm(path[:-1] - info["prev_pos"], axis=1))
        return (accum_path_len[idx] - accum_path_len[prev_idx]) / accum_path_len[-1]

    def _reward_fn_intensity_sensor(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """The reward is the grayscaled intensity of the a intensity sensor taken to
        the power of some gamma value multiplied by a
        scale factor (1 / max_episode_steps).
        """
        if self._num_resets == 1:
            # Do some checks
            assert "intensity" in info
            assert "gamma" in self.env_config.reward_options
            assert isinstance(self.env_config.reward_options["gamma"], (int, float))

        intensity = info["intensity"] / 255.0
        if ambient_light_intensity := self.env_config.ambient_light_intensity:
            intensity = np.clip(intensity - ambient_light_intensity, 0.0, 1.0)

        return np.mean(intensity) ** self.env_config.reward_options["gamma"]

    def _reward_fn_intensity_and_velocity(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """This reward combines `reward_fn_intensity_sensor` and
        `reward_fn_delta_euclidean`."""
        intensity_reward = self._reward_fn_intensity_sensor(animal, info)
        velocity_reward = self._reward_fn_delta_euclidean(animal, info)
        return (intensity_reward + velocity_reward) / 2

    def _reward_fn_intensity_euclidean_and_at_goal(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """This reward combines `reward_fn_intensity_sensor`,
        `reward_fn_euclidean`, and `reward_fn_sparse`."""
        intensity_reward = self._reward_fn_intensity_sensor(animal, info)
        euclidean_reward = self._reward_fn_delta_euclidean(animal, info)
        reward = (intensity_reward + euclidean_reward) / 2
        return 1 if self._is_at_goal(animal) else reward

    def _reward_fn_energy_per_step_and_at_goal(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """This reward combines `reward_fn_energy_per_step` and `reward_fn_intensity_sensor`."""
        if self._num_resets == 1:
            assert "energy_per_step" in self.env_config.reward_options

        energy_per_step = self.env_config.reward_options["energy_per_step"]
        energy_per_step = np.clip(energy_per_step * (animal.num_pixels), -1.0, 0)
        intensity_reward = self._reward_fn_intensity_sensor(animal, info)
        return (
            self.env_config.reward_at_goal + energy_per_step + intensity_reward
            if self._is_at_goal(animal)
            else energy_per_step + intensity_reward
        )

    def _reward_fn_intensity_and_at_goal(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """The reward is the intensity whenever the animal is outside some threshold
        in terms of euclidean distance to the goal. But if it's within this threshold,
        then the reward is 1."""
        intensity_reward = self._reward_fn_intensity_sensor(animal, info)
        return 1 if self._is_at_goal(animal) else intensity_reward

    def _reward_fn_intensity_and_euclidean(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """This reward combines `reward_fn_intensity_sensor` and
        `reward_fn_euclidean`."""
        intensity_reward = self._reward_fn_intensity_sensor(animal, info)
        euclidean_reward = self._reward_fn_euclidean(animal, info)
        return (intensity_reward + euclidean_reward) / 2

    def _reward_fn_sparse(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """This reward is 1 if the animal is at the goal, -0.1 otherwise."""
        return 1 if self._is_at_goal(animal) else -0.1

    # Helpers

    def _calc_delta_pos(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
        point: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculates the delta position of the animal.

        NOTE: returns delta position of current pos from prev pos (i.e. current - prev)
        """
        if point is None:
            point = np.array([0, 0])
        current_distance = np.linalg.norm(animal.pos - point)
        prev_distance = np.linalg.norm(info["prev_pos"] - point)
        return current_distance - prev_distance


def make_single_env(
    config: Path | str | MjCambrianConfig, seed: int, **kwargs
) -> MjCambrianEnv:
    """Utility function for multiprocessed MjCambrianEnv."""

    def _init():
        env = MjCambrianEnv(config, **kwargs)
        env.reset(seed=seed)
        return env

    return _init


if __name__ == "__main__":
    import time
    from cambrian.utils.utils import MjCambrianArgumentParser

    parser = MjCambrianArgumentParser()

    parser.add_argument(
        "--mj-viewer",
        action="store_true",
        help="Whether to use the mujoco viewer.",
        default=False,
    )

    parser.add_argument(
        "-t",
        "--total-timesteps",
        type=int,
        help="The number of timesteps to run the environment for.",
        default=np.inf,
    )
    parser.add_argument(
        "--record-path",
        type=str,
        help="The path to save the video to. It will save a gif and mp4. "
        "Don't specify an extension. If not specified, will not record.",
        default=None,
    )
    parser.add_argument(
        "--record-composites",
        action="store_true",
        help="Whether to record the composite image in addition to the full rendered "
        "image. Only used if `--record-path` is specified.",
    )

    parser.add_argument(
        "--speed-test",
        action="store_true",
        help="Whether to run a speed test.",
        default=False,
    )

    args = parser.parse_args()

    config = MjCambrianConfig.load(args.config, overrides=args.overrides)
    if args.mj_viewer:
        config.env_config.use_renderer = False
    env = MjCambrianEnv(config)
    env.reset(seed=config.training_config.seed)
    # env.xml.write("test.xml")

    action = {
        name: np.zeros_like(animal.action_space.sample())
        for name, animal in env.animals.items()
    }

    print("Running...")
    if args.mj_viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            env.model, env.data  # , show_left_ui=False, show_right_ui=False
        ) as viewer:
            while viewer.is_running():
                env.step(action)
                viewer.sync()
    else:
        record_composites = False
        if args.record_path is not None:
            assert (
                args.total_timesteps < np.inf
            ), "Must specify `-t\--total-timesteps` if recording."
            env.renderer.record = True
            if args.record_composites:
                record_composites = True
                composites = {k: [] for k in env.animals}

        action_map = {0: np.array([0, -0.5]), 10: np.array([1, -0.5])}

        t0 = time.time()
        step = 0
        while step < args.total_timesteps:
            if step in action_map:
                action = {
                    name: action_map[step] for name, animal in env.animals.items()
                }

            _, reward, _, _, _ = env.step(action)
            env.overlays["Step Reward"] = f"{reward['animal_0']:.2f}"

            if env.config.env_config.use_renderer:
                if not env.renderer.is_running():
                    break
                env.render()
            if record_composites:
                for name, animal in env.animals.items():
                    composite = animal.create_composite_image()
                    resized_composite = resize_with_aspect_fill(
                        composite, composite.shape[0] * 20, composite.shape[1] * 20
                    )
                    composites[name].append(resized_composite)

            if args.speed_test and step % 100 == 0:
                fps = step / (time.time() - t0)
                print(f"FPS: {fps}")

            step += 1
        t1 = time.time()
        if args.speed_test:
            print(f"Total time: {t1 - t0}")
            print(f"FPS: {env._episode_step / (t1 - t0)}")

        env.close()

        if args.record_path is not None:
            env.renderer.save(args.record_path)
            print(f"Saved video to {args.record_path}")
            if record_composites:
                import imageio

                for name, composite in composites.items():
                    path = f"{args.record_path}_{name}_composites"
                    imageio.mimwrite(
                        f"{path}.gif",
                        composite,
                        duration=1000 * 1 / 30,
                    )
                    imageio.imwrite(f"{path}.png", composite[-1])

    print("Exiting...")
