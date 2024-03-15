from typing import Dict, Any, Tuple, List, Optional, Callable, Concatenate, TYPE_CHECKING
from pathlib import Path
import pickle

import numpy as np
import gymnasium as gym
import mujoco as mj
from gymnasium import spaces
from stable_baselines3.common.utils import set_random_seed

from cambrian.animal import MjCambrianAnimal, MjCambrianPointAnimal, MjCambrianAnimalConfig
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.cambrian_xml import MjCambrianXML, MjCambrianXMLConfig
from cambrian.utils.logger import get_logger
from cambrian.renderer import (
    MjCambrianRenderer,
    MjCambrianRendererConfig,
    MjCambrianViewerOverlay,
    MjCambrianTextViewerOverlay,
    MjCambrianImageViewerOverlay,
    MjCambrianSiteViewerOverlay,
    MjCambrianCursor,
    resize_with_aspect_fill,
    TEXT_HEIGHT,
    TEXT_MARGIN,
)
if TYPE_CHECKING:
    from cambrian.utils.config import MjCambrianConfig

@config_wrapper
class MjCambrianEnvConfig(MjCambrianBaseConfig):
    """Defines a config for the cambrian environment.

    Attributes:
        xml (MjCambrianXMLConfig): The xml for the scene. This is the xml that will be
            used to create the environment. See `MjCambrianXMLConfig` for more info.

        reward_fn (MjCambrianRewardFn): The reward function type to use. See the
            `MjCambrianRewardFn` for more info.

        use_goal_obs (bool): Whether to use the goal observation or not.
        terminate_at_goal (bool): Whether to terminate the episode when the animal
            reaches the goal or not.
        truncate_on_contact (bool): Whether to truncate the episode when the animal
            makes contact with an object or not.
        distance_to_target_threshold (float): The distance to the target at which the
            animal is assumed to be "at the target".
        action_penalty (float): The action penalty when it moves.
        adversary_penalty (float): The adversary penalty when it goes to the wrong target.
        contact_penalty (float): The contact penalty when it contacts the wall.
        force_exclusive_contact_penalty (bool): Whether to force exclusive contact
            penalty or not. If True, the contact penalty will be used exclusively for
            the reward. If False, the contact penalty will be used in addition to the
            calculated reward.

        frame_skip (int): The number of mujoco simulation steps per `gym.step()` call.

        add_overlays (bool): Whether to add overlays or not.
        clear_overlays_on_reset (bool): Whether to clear the overlays on reset or not.
            Consequence of setting to False is that if `add_position_tracking_overlay`
            is True and mazes change between evaluations, the sites will be drawn on top
            of each other which may not be desired. When record is False, the overlays
            are always cleared.
        renderer (Optional[MjCambrianViewerConfig]): The default viewer config to
            use for the mujoco viewer. If unset, no renderer will be used. Should
            set to None if `render` will never be called. This may be useful to
            reduce the amount of vram consumed by non-rendering environments.

        eval_overrides (Optional[Dict[str, Any]]): Key/values to override the default
            env during evaluation. Applied during evaluation only. Merged directly
            with the env. The actual datatype is Self/MjCambrianEnvConfig but all
            attributes are optional. NOTE: This dict is only applied at reset,
            meaning mujoco xml changes will not be reflected in the eval episode.

        animals (List[MjCambrianAnimalConfig]): The configs for the animals.
            The key will be used as the default name for the animal, unless explicitly
            set in the animal config.
    """

    xml: MjCambrianXMLConfig

    reward_fn: Callable[Concatenate[MjCambrianAnimal, Dict[str, Any], ...], float]

    use_goal_obs: bool
    terminate_at_goal: bool
    truncate_on_contact: bool
    distance_to_target_threshold: float
    action_penalty: float
    adversary_penalty: float
    contact_penalty: float
    force_exclusive_contact_penalty: bool

    frame_skip: int

    add_overlays: bool
    clear_overlays_on_reset: bool
    renderer: Optional[MjCambrianRendererConfig] = None

    eval_overrides: Optional[Dict[str, Any]] = None

    animals: Dict[str, MjCambrianAnimalConfig]

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
    that we allow for /reset multiple agents and use our own custom renderer. It also
    reduces the need to create temporary xml files which MujocoEnv had to load. It's
    essentially a copy of MujocoEnv with the two aforementioned major changes.

    Args:
        config_path (str | Path | MjCambrianConfig): The path to the config file or the
            config object itself.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: "MjCambrianConfig"):
        self.config = config
        self.logger = get_logger(self.config)

        self.animals: Dict[str, MjCambrianAnimal] = {}
        self._create_animals()

        self.xml = self.generate_xml()

        self.model = mj.MjModel.from_xml_string(self.xml.to_string())
        self.data = mj.MjData(self.model)

        self.renderer: MjCambrianRenderer = None
        self.render_mode = (
            "human" if "human" in self.config.env.renderer.render_modes else "rgb_array"
        )
        if renderer_config := self.config.env.renderer:
            self.renderer = MjCambrianRenderer(renderer_config)

        self._episode_step = 0
        self._max_episode_steps = self.config.training.max_episode_steps
        self._num_resets = 0
        self._stashed_cumulative_reward = 0
        self._cumulative_reward = 0

        self._record: bool = False
        self._rollout: Dict[str, Any] = {}
        self._overlays: Dict[str, Any] = {}

    def _create_animals(self):
        """Helper method to create the animals.

        Under the hood, the `create` method does the following:
            - load the base xml to MjModel
            - parse the geometry and place eyes at the appropriate locations
            - create the action/observation spaces

        TODO: Hardcoded to use MjCambrianPointAnimal for now!!
        """
        for name, animal_config in self.config.env.animals.items():
            assert name not in self.animals
            self.animals[name] = MjCambrianPointAnimal(animal_config, name)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = MjCambrianXML.make_empty()

        # Add the animals to the xml
        for idx, animal in enumerate(self.animals.values()):
            xml += animal.generate_xml(idx)

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
        if seed is not None and self._num_resets == 0:
            self.logger.info(f"Setting random seed to {seed}")
            set_random_seed(seed)

        mj.mj_resetData(self.model, self.data)

        # Reset the animals
        info: Dict[str, Any] = {a: {} for a in self.animals}
        obs: Dict[str, Dict[str, Any]] = {}
        for name, animal in self.animals.items():
            obs[name] = animal.reset(self.model, self.data)

            info[name]["pos"] = animal.pos

        self._step_mujoco_simulation(1)

        if self.renderer is not None:
            # The env renderer can see all sites and geoms
            # This is done by setting the all sitegroups and geomgroups to True
            self.renderer.set_option("sitegroup", True, slice(None))
            self.renderer.set_option("geomgroup", True, slice(None))

            self.renderer.reset(self.model, self.data)

        self._episode_step = 0
        self._stashed_cumulative_reward = self._cumulative_reward
        self._cumulative_reward = 0
        self._num_resets += 1
        if not self.record:
            self._rollout.clear()
            self._overlays.clear()
        elif self.env_config.clear_overlays_on_reset:
            self._overlays.clear()

        self._overlays["Exp"] = self.config.training_config.exp_name

        if self.record:
            self._rollout.setdefault("actions", [])
            self._rollout["actions"].append(
                [np.zeros_like(a.action_space.sample()) for a in self.animals.values()]
            )
            self._rollout.setdefault("positions", [])
            self._rollout["positions"].append([a.pos for a in self.animals.values()])

        return obs, info

    def step(self, action: Dict[str, Any]) -> Tuple[
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
        self._cumulative_reward += sum(reward.values())
        self._stashed_cumulative_reward = self._cumulative_reward

        self._overlays["Step"] = self._episode_step
        self._overlays["Cumulative Reward"] = round(self._cumulative_reward, 2)

        # Add the position of each animal to the overlays. Only add if we're recording
        if self.record:
            i = self._num_resets * self._max_episode_steps + self._episode_step
            for animal in self.animals.values():
                key = f"{animal.name}_pos_{i}"
                size = self.maze.max_dim * 5e-3
                self._overlays[key] = MjCambrianSiteViewerOverlay(
                    animal.xpos.copy(),
                    [1, 0, 0, 1],  # red
                    size,
                )

        if self.record:
            self._rollout["actions"].append(list(action.values()))
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
            rewards[name] = self.reward_fn(animal, info[name])

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

        if not self.env_config.add_overlays:
            return self.renderer.render()

        renderer = self.renderer
        renderer_width = renderer.width
        renderer_height = renderer.height

        overlays: List[MjCambrianViewerOverlay] = []

        cursor = MjCambrianCursor(x=0, y=renderer_height - TEXT_MARGIN * 2)
        for key, value in self._overlays.items():
            if issubclass(type(value), MjCambrianViewerOverlay):
                overlays.append(value)
            else:
                cursor.y -= TEXT_HEIGHT + TEXT_MARGIN
                overlays.append(MjCambrianTextViewerOverlay(f"{key}: {value}", cursor))

        # Set the overlay size to be a fraction of the renderer size relative to
        # the animal count. The overlay height will be set to 35% of the renderer from
        # the bottom
        num_animals = len(self.animals)
        overlay_width = int(renderer_width // num_animals) if num_animals > 0 else 0
        overlay_height = int(renderer_height * 0.35)
        overlay_size = (overlay_width, overlay_height)

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

            # NOTE: flipud here since we always flipud when copying buffer from gpu,
            # and when reading the buffer again after drawing the overlay, it will be
            # flipped again. Flipping here means it will be the right side up.
            new_composite = np.flipud(resize_with_aspect_fill(composite, *overlay_size))

            overlays.append(MjCambrianImageViewerOverlay(new_composite * 255.0, cursor))

            cursor.x -= TEXT_MARGIN
            cursor.y -= TEXT_MARGIN
            overlay_text = f"Num Eyes: {len(animal.eyes)}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            cursor.y += TEXT_HEIGHT
            if animal.num_eyes > 0:
                eye0 = next(iter(animal.eyes.values()))
                overlay_text = f"Res: {tuple(eye0.resolution)}"
                overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
                cursor.y += TEXT_HEIGHT
                overlay_text = f"FOV: {tuple(eye0.fov)}"
                overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
                cursor.y = overlay_height - TEXT_HEIGHT * 2 + TEXT_MARGIN * 2
            overlay_text = f"Animal: {name}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            overlay_text = f"Action: {[f'{a: 0.3f}' for a in animal.last_action]}"
            cursor.y -= TEXT_HEIGHT
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))

            cursor.x += overlay_width
            cursor.y = 0

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
    def num_timesteps(self) -> int:
        """Returns the number of timesteps."""
        return self.max_episode_steps * self.num_resets + self.episode_step

    @property
    def max_episode_steps(self) -> int:
        """Returns the max episode steps."""
        return self._max_episode_steps

    @property
    def overlays(self) -> Dict[str, Any]:
        """Returns the overlays."""
        return self._overlays

    @property
    def cumulative_reward(self) -> float:
        """Returns the cumulative reward."""
        return self._stashed_cumulative_reward

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
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
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
    def n_eval_mazes(self) -> int:
        """Returns the number of evaluation mazes. Will run eval on all mazes."""
        n_eval_mazes = len(self.env_config.maze_configs)
        eval_overrides = self.config.env_config.eval_overrides
        if eval_overrides and (eval_maze_configs := eval_overrides.get("maze_configs")):
            n_eval_mazes = len(eval_maze_configs)
        return n_eval_mazes

    def save(self, path: str | Path, *, save_pkl: bool = True, **kwargs):
        """Saves the simulation output to the given path."""
        self.renderer.save(path, **kwargs)

        if save_pkl:
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
        path, accum_path_len = info["optimal_path"][animal.name]
        idx = np.argmin(np.linalg.norm(path[:-1] - animal.pos, axis=1))
        return accum_path_len[idx] / accum_path_len[-1]

    def _reward_fn_delta_distance_along_path(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """Rewards the distance along the optimal path to the goal."""
        path, accum_path_len = info["optimal_path"][animal.name]
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
        """This reward combines `reward_fn_energy_per_step` and
        `reward_fn_intensity_sensor`."""
        if self._num_resets == 1:
            assert "energy_per_step" in self.env_config.reward_options
            assert "reward_at_goal" in self.env_config.reward_options

        reward_at_goal = self.env_config.reward_options["reward_at_goal"]
        energy_per_step = self.env_config.reward_options["energy_per_step"]
        energy_per_step = np.clip(energy_per_step * animal.num_pixels, -1.0, 0)
        intensity_reward = self._reward_fn_intensity_sensor(animal, info)
        return (
            reward_at_goal + energy_per_step + intensity_reward
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

    def _reward_fn_fitness_energy_and_delta_euclidean(
        self,
        animal: MjCambrianAnimal,
        info: Dict[str, Any],
    ) -> float:
        """NOTE: used for fitness, so may be larger than 1."""
        if self._num_resets == 1:
            assert "reward_at_goal" in self.env_config.reward_options
            assert "energy_penalty_per_pixel" in self.env_config.reward_options
            assert "energy_penalty_per_eye" in self.env_config.reward_options
            assert "delta_euclidean_weight" in self.env_config.reward_options

        reward_at_goal = self.env_config.reward_options["reward_at_goal"]
        if not self._is_at_goal(animal):
            reward_at_goal = 0

        energy_penalty_per_pixel = self.env_config.reward_options[
            "energy_penalty_per_pixel"
        ]
        energy_penalty_per_eye = self.env_config.reward_options[
            "energy_penalty_per_eye"
        ]
        energy_penalty = animal.num_eyes * energy_penalty_per_eye
        for eye in animal.eyes.values():
            energy_penalty += eye.num_pixels * energy_penalty_per_pixel

        delta_euclidean_weight = self.env_config.reward_options[
            "delta_euclidean_weight"
        ]
        delta_euclidean_reward = (
            self._reward_fn_delta_euclidean(animal, info) * delta_euclidean_weight
        )
        return reward_at_goal + energy_penalty + delta_euclidean_reward

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


def make_single_env(config: "MjCambrianConfig", seed: int, **kwargs) -> MjCambrianEnv:
    """Utility function for multiprocessed MjCambrianEnv."""

    def _init():
        env = MjCambrianEnv(config, **kwargs)
        env.reset(seed=seed)
        return env

    return _init


if __name__ == "__main__":
    import time
    from cambrian.utils.utils import MjCambrianArgumentParser
    from cambrian.utils.config import MjCambrianConfig

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

        action_map = {
            0: np.array([-0.9, -0.1])
        }  # {0: np.array([0, -0.5]), 10: np.array([1, -0.5])}

        t0 = time.time()
        step = 0
        while step < args.total_timesteps:
            if step in action_map:
                action = {
                    name: action_map[step] for name, animal in env.animals.items()
                }

            _, reward, _, _, _ = env.step(action)
            env.overlays["Step Reward"] = f"{next(iter(reward.values())):.2f}"

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
