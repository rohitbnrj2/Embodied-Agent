from typing import Dict, Any, Tuple, List, Optional, Callable, Concatenate, Self
from pathlib import Path
import pickle

import numpy as np
import gymnasium as gym
import mujoco as mj
from gymnasium import spaces
from stable_baselines3.common.utils import set_random_seed

from cambrian.animal import (
    MjCambrianAnimal,
    MjCambrianPointAnimal,
    MjCambrianAnimalConfig,
)
from cambrian.renderer import (
    MjCambrianRenderer,
    MjCambrianRendererConfig,
    resize_with_aspect_fill,
)
from cambrian.renderer.overlays import (
    MjCambrianViewerOverlay,
    MjCambrianTextViewerOverlay,
    MjCambrianImageViewerOverlay,
    MjCambrianCursor,
    TEXT_HEIGHT,
    TEXT_MARGIN,
)
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.cambrian_xml import MjCambrianXML, MjCambrianXMLConfig
from cambrian.utils.logger import get_logger


@config_wrapper
class MjCambrianEnvConfig(MjCambrianBaseConfig):
    """Defines a config for the cambrian environment.

    Attributes:
        xml (MjCambrianXMLConfig): The xml for the scene. This is the xml that will be
            used to create the environment. See `MjCambrianXML` for more info.

        reward_fn (MjCambrianRewardFn): The reward function type to use. See the
            `MjCambrianRewardFn` for more info.

        frame_skip (int): The number of mujoco simulation steps per `gym.step()` call.
        max_episode_steps (int): The maximum number of steps per episode.

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

        eval_overrides (Optional[Self]): Overrides to apply to the env config during
            evaluation. Merged directly with the env. The actual datatype is
            Self/MjCambrianEnvConfig but all attributes are optional.

        animals (List[MjCambrianAnimalConfig]): The configs for the animals.
            The key will be used as the default name for the animal, unless explicitly
            set in the animal config.
    """

    xml: MjCambrianXMLConfig

    reward_fn: Callable[Concatenate[MjCambrianAnimal, Dict[str, Any], ...], float]

    frame_skip: int
    max_episode_steps: int

    eval_overrides: Optional[Self] = None

    add_overlays: bool
    clear_overlays_on_reset: bool
    renderer: Optional[MjCambrianRendererConfig] = None

    animals: Dict[str, MjCambrianAnimalConfig]


class MjCambrianEnv(gym.Env):
    """A MjCambrianEnv defines a gymnasium environment that's based off mujoco.

    NOTES:
    - This is an overridden version of the MujocoEnv class. The two main differences is
    that we allow for /reset multiple agents and use our own custom renderer. It also
    reduces the need to create temporary xml files which MujocoEnv had to load. It's
    essentially a copy of MujocoEnv with the two aforementioned major changes.

    Args:
        config (MjCambrianEnvConfig): The config object.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: MjCambrianEnvConfig):
        self.config = config
        self.logger = get_logger()

        self.animals: Dict[str, MjCambrianAnimal] = {}
        self._create_animals()

        self.xml = self.generate_xml()

        try:
            self.model = mj.MjModel.from_xml_string(self.xml.to_string())
        except ValueError as e:
            print(self.xml)
            raise e

        self.data = mj.MjData(self.model)

        self.render_mode = "rgb_array"
        self.renderer: MjCambrianRenderer = None
        if renderer_config := self.config.renderer:
            self.render_mode = (
                "human" if "human" in self.config.renderer.render_modes else "rgb_array"
            )
            self.renderer = MjCambrianRenderer(renderer_config)

        self._episode_step = 0
        self._max_episode_steps = self.config.max_episode_steps
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
        for name, animal_config in self.config.animals.items():
            assert name not in self.animals
            self.animals[name] = MjCambrianPointAnimal(animal_config, name)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = MjCambrianXML.from_config(self.config.xml)

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

        # First, reset the mujoco simulation
        mj.mj_resetData(self.model, self.data)

        # Then, reset the animals
        info: Dict[str, Any] = {a: {} for a in self.animals}
        obs: Dict[str, Dict[str, Any]] = {}
        for name, animal in self.animals.items():
            obs[name] = animal.reset(self.model, self.data)

            info[name]["pos"] = animal.pos

        # We'll step the simulation once to allow for states to propogate
        self._step_mujoco_simulation(1)

        if self.renderer is not None:
            self.renderer.reset(self.model, self.data)

        # Update metadata variables
        self._episode_step = 0
        self._stashed_cumulative_reward = self._cumulative_reward
        self._cumulative_reward = 0
        self._num_resets += 1

        # Reset the caches
        if not self.record:
            self._rollout.clear()
            self._overlays.clear()
        elif self.config.clear_overlays_on_reset:
            self._overlays.clear()

        if self.record:
            # TODO make this cleaner
            self._rollout.setdefault("actions", [])
            self._rollout["actions"].append(
                [np.zeros_like(a.action_space.sample()) for a in self.animals.values()]
            )
            self._rollout.setdefault("positions", [])
            self._rollout["positions"].append([a.pos for a in self.animals.values()])

        # TODO
        # if expname := options.get("expname"):
        #     self._overlays["Exp"] = expname

        return self._update_obs(obs), self._update_info(info)

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

        The dynamics is updated through the `_step_mujoco_simulation` method.

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
        self._step_mujoco_simulation(self.config.frame_skip)

        # We'll then step each animal to render it's current state and get the obs
        obs: Dict[str, Any] = {}
        for name, animal in self.animals.items():
            obs[name] = animal.step()

            info[name]["action"] = action[name]

        obs = self._update_obs(obs)
        terminated = self._compute_terminated()
        truncated = self._compute_truncated()
        reward = self._compute_reward(terminated, truncated, info)
        info = self._update_info(info)

        self._episode_step += 1
        self._cumulative_reward += sum(reward.values())
        self._stashed_cumulative_reward = self._cumulative_reward

        self._overlays["Step"] = self._episode_step
        self._overlays["Cumulative Reward"] = round(self._cumulative_reward, 2)

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

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mj.mj_rnePostConstraint(self.model, self.data)

    def _update_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Overridable method to update the observations. This class will just return
        the observations as is, but subclasses can override this method to provide
        custom observation updates."""
        return obs

    def _update_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Overridable method to update the info. This class will just return the info
        as is, but subclasses can override this method to provide custom info updates.
        """
        return info

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

            # Call the reward fn
            rewards[name] = self.config.reward_fn(animal, info[name])

        return rewards

    def _compute_terminated(self) -> Dict[str, bool]:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure.

        The default implementation will always return False for all animals. This can
        be overridden in subclasses to provide custom termination conditions.
        """

        terminated: Dict[str, bool] = {}
        for name in self.animals:
            terminated[name] = False

        return terminated

    def _compute_truncated(self) -> bool:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure.

        The default implementation will always return False for all animals. This can
        be overridden in subclasses to provide custom termination conditions.
        """

        truncated: Dict[str, bool] = {}
        for name in self.animals:
            truncated[name] = self._episode_step >= (self._max_episode_steps - 1)

        return truncated

    def render(self) -> Dict[str, np.ndarray]:
        """Renders the environment.

        Returns:
            Dict[str, np.ndarray]: The rendered image for each render mode mapped to
                its corresponding str.

        TODO:
            - Make the cursor stuff clearer
        """

        assert self.renderer is not None, "Renderer has not been initialized! "
        "Ensure `use_renderer` is set to True in the constructor."

        if not self.config.add_overlays:
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
            if cursor.x + overlay_width > renderer_width:
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
                overlay_text = f"Res: {tuple(eye0.config.resolution)}"
                overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
                cursor.y += TEXT_HEIGHT
                overlay_text = f"FOV: {tuple(eye0.config.fov)}"
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
        observations/actions/etc. This method will create _all_ the observation
        spaces for the environment. But note that stable baselines3 only supports single
        agent environments (i.e. non-nested spaces.Dict), so ensure you wrap this env
        with a `wrappers.MjCambrianSingleAnimalEnvWrapper` if you want to use stable
        baselines3.
        """

        # Create the observation_spaces
        observation_spaces: Dict[str, spaces.Space] = {}
        for name, animal in self.animals.items():
            observation_space: spaces.Dict = animal.observation_space
            observation_spaces[name] = observation_space
        return spaces.Dict(observation_spaces)

    @property
    def action_spaces(self) -> spaces.Dict:
        """Creates the action spaces.

        This is part of the PettingZoo API.

        By default, this environment will support multi-animal
        observations/actions/etc. This method will create _all_ the action
        spaces for the environment. But note that stable baselines3 only supports single
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

    def save(self, path: str | Path, *, save_pkl: bool = True, **kwargs):
        """Saves the simulation output to the given path."""
        self.renderer.save(path, **kwargs)

        if save_pkl:
            self.logger.info(f"Saving rollout to {path.with_suffix('.pkl')}")
            with open(path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(self._rollout, f)
            self.logger.debug(f"Saved rollout to {path.with_suffix('.pkl')}")


if __name__ == "__main__":
    from cambrian.utils.config import MjCambrianConfig, run_hydra

    def environment_factory(config: MjCambrianEnvConfig):
        from cambrian.envs.object_env import MjCambrianObjectEnvConfig
        from cambrian.envs.maze_env import MjCambrianMazeEnvConfig

        if config.get_type() == MjCambrianObjectEnvConfig:
            from cambrian.envs.object_env import MjCambrianObjectEnv

            return MjCambrianObjectEnv(config)
        elif config.get_type() == MjCambrianMazeEnvConfig:
            from cambrian.envs.maze_env import MjCambrianMazeEnv

            return MjCambrianMazeEnv(config)
        else:
            return MjCambrianEnv(config)

    def run_mj_viewer(config: MjCambrianConfig):
        import mujoco.viewer

        env = environment_factory(config.env)
        env.reset(seed=config.seed)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running() and env.renderer.is_running():
                env.step(env.action_spaces.sample())
                env.render()
                viewer.sync()

    run_hydra(run_mj_viewer)
