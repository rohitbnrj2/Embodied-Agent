from typing import (
    Dict,
    Any,
    Tuple,
    List,
    Optional,
    Callable,
    Self,
    TypeAlias,
    Concatenate,
)
from pathlib import Path
import pickle

import numpy as np
import mujoco as mj
from gymnasium import spaces
from pettingzoo import ParallelEnv

from cambrian.animals.animal import MjCambrianAnimal, MjCambrianAnimalConfig
from cambrian.renderer import (
    MjCambrianRenderer,
    MjCambrianRendererConfig,
    MjCambrianRendererSaveMode,
    resize_with_aspect_fill,
)
from cambrian.renderer.overlays import (
    MjCambrianViewerOverlay,
    MjCambrianTextViewerOverlay,
    MjCambrianSiteViewerOverlay,
    MjCambrianImageViewerOverlay,
    MjCambrianCursor,
    TEXT_HEIGHT,
    TEXT_MARGIN,
)
from cambrian.utils.config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.cambrian_xml import MjCambrianXML, MjCambrianXMLConfig
from cambrian.utils.logger import get_logger

MjCambrianTerminationFn: TypeAlias = Callable[
    Concatenate["MjCambrianEnv", MjCambrianAnimal, Dict[str, Any], ...],
    bool,
]

MjCambrianTruncationFn: TypeAlias = Callable[
    Concatenate["MjCambrianEnv", MjCambrianAnimal, Dict[str, Any], ...],
    bool,
]

MjCambrianRewardFn: TypeAlias = Callable[
    Concatenate["MjCambrianEnv", MjCambrianAnimal, bool, bool, Dict[str, Any], ...],
    float,
]


@config_wrapper
class MjCambrianEnvConfig(MjCambrianBaseConfig):
    """Defines a config for the cambrian environment.

    Attributes:
        instance (Callable[[Self], "MjCambrianEnv"]): The class method to use to
            instantiate the environment.

        xml (MjCambrianXMLConfig): The xml for the scene. This is the xml that will be
            used to create the environment. See `MjCambrianXML` for more info.

        termination_fn (MjCambrianTerminationFn): The termination function to use. See
            the `MjCambrianTerminationFn` for more info.
        truncation_fn (MjCambrianTruncationFn): The truncation function to use. See the
            `MjCambrianTruncationFn` for more info.
        reward_fn (MjCambrianRewardFn): The reward function type to use. See the
            `MjCambrianRewardFn` for more info.

        frame_skip (int): The number of mujoco simulation steps per `gym.step()` call.
        max_episode_steps (int): The maximum number of steps per episode.
        n_eval_episodes (int): The number of episodes to evaluate for.

        add_overlays (bool): Whether to add overlays or not.
        clear_overlays_on_reset (bool): Whether to clear the overlays on reset or not.
            Consequence of setting to False is that if `add_position_tracking_overlay`
            is True and mazes change between evaluations, the sites will be drawn on top
            of each other which may not be desired. When record is False, the overlays
            are always cleared.
        render_animal_composite_only (Optional[bool]): If set, will only render the
            composite image all animals.
        renderer (Optional[MjCambrianViewerConfig]): The default viewer config to
            use for the mujoco viewer. If unset, no renderer will be used. Should
            set to None if `render` will never be called. This may be useful to
            reduce the amount of vram consumed by non-rendering environments.

        save_filename (Optional[str]): The filename to save recordings to. This is more
            of a placeholder for external scripts to use, if desired.

        animals (List[MjCambrianAnimalConfig]): The configs for the animals.
            The key will be used as the default name for the animal, unless explicitly
            set in the animal config.
    """

    instance: Callable[[Self], "MjCambrianEnv"]

    xml: MjCambrianXMLConfig

    termination_fn: MjCambrianTerminationFn
    truncation_fn: MjCambrianTruncationFn
    reward_fn: MjCambrianRewardFn

    frame_skip: int
    max_episode_steps: int
    n_eval_episodes: int

    add_overlays: bool
    clear_overlays_on_reset: bool
    render_animal_composite_only: Optional[bool] = None
    renderer: Optional[MjCambrianRendererConfig] = None

    save_filename: Optional[str] = None

    animals: Dict[str, MjCambrianAnimalConfig]


class MjCambrianEnv(ParallelEnv):
    """A MjCambrianEnv defines a gymnasium environment that's based off mujoco.

    NOTES:
    - This is an overridden version of the MujocoEnv class. The two main differences is
    that we allow for /reset multiple agents and use our own custom renderer. It also
    reduces the need to create temporary xml files which MujocoEnv had to load. It's
    essentially a copy of MujocoEnv with the two aforementioned major changes.

    Args:
        config (MjCambrianEnvConfig): The config object.
        name (Optional[str]): The name of the environment. This is added as an overlay
            to the renderer.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: MjCambrianEnvConfig, name: Optional[str] = None):
        self._config = config
        self._name = name or self.__class__.__name__
        self._logger = get_logger()

        self._animals: Dict[str, MjCambrianAnimal] = {}
        self._create_animals()

        self._xml = self.generate_xml()

        try:
            self._model = mj.MjModel.from_xml_string(self._xml.to_string())
        except:
            self._logger.error(
                f"Error creating model from xml\n{self._xml.to_string()}"
            )
            raise

        self._data = mj.MjData(self._model)

        self.render_mode = "rgb_array"
        self._renderer: MjCambrianRenderer = None
        if renderer_config := self._config.renderer:
            if "human" in self._config.renderer.render_modes:
                self.render_mode = "human"
            self._renderer = MjCambrianRenderer(renderer_config)

        self._episode_step = 0
        self._max_episode_steps = self._config.max_episode_steps
        self._num_resets = 0
        self._num_timesteps = 0
        self._stashed_cumulative_reward = 0
        self._cumulative_reward = 0

        self._record: bool = False
        self._rollout: Dict[str, Any] = {}
        self._overlays: Dict[str, Any] = {}

        # We'll store the info dict as a state within this class so that the truncation,
        # termination, and reward functions can use it for keeping a state. Like passing
        # the info dict to these functions allows them to edit them and keep around
        # information that is helpful for subsequent calls. It will always be reset
        # during the reset method and will only be maintained during an episode length.
        # Because the info dict is treated as stateful, take care in not adding new keys
        # on each step, as this will cause the info dict to grow until the end of the
        # episode.
        self._info: Dict[str, Dict[str, Any]]

    def _create_animals(self):
        """Helper method to create the animals."""
        for i, (name, animal_config) in enumerate(self._config.animals.items()):
            assert name not in self._animals
            self._animals[name] = animal_config.instance(animal_config, name, i)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = MjCambrianXML.from_string(self._config.xml)

        # Add the animals to the xml
        for animal in self._animals.values():
            xml += animal.generate_xml()

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
        if seed is not None and self._num_resets == 0:
            self.set_random_seed(seed)

        # First, reset the mujoco simulation
        mj.mj_resetData(self._model, self._data)

        # Reset the info dict. We'll update the stateful info dict here, as well.
        info: Dict[str, Dict[str, Any]] = {a: {} for a in self._animals}
        self._info = info

        # Then, reset the animals
        obs: Dict[str, Dict[str, Any]] = {}
        for name, animal in self._animals.items():
            obs[name] = animal.reset(self._model, self._data)

        # We'll step the simulation once to allow for states to propagate
        self._step_mujoco_simulation(1, info)

        # Now update the info dict
        if self._renderer is not None:
            self._renderer.reset(self._model, self._data)

        # Update metadata variables
        self._episode_step = 0
        self._stashed_cumulative_reward = self._cumulative_reward
        self._cumulative_reward = 0
        self._num_resets += 1

        # Reset the caches
        if not self.record:
            self._rollout.clear()
            self._overlays.clear()
        elif self._config.clear_overlays_on_reset:
            self._overlays.clear()

        if self.record:
            # TODO make this cleaner
            self._rollout.setdefault("actions", [])
            self._rollout["actions"].append(
                [np.zeros_like(a.action_space.sample()) for a in self._animals.values()]
            )
            self._rollout.setdefault("positions", [])
            self._rollout["positions"].append([a.qpos for a in self._animals.values()])

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
            action (Dict[str, Any]): The action to take for each animal.
                The keys define the animal name, and the values define the action for
                that animal.

        Returns:
            Dict[str, Any]: The observations for each animal.
            Dict[str, float]: The reward for each animal.
            Dict[str, bool]: Whether each animal has terminated.
            Dict[str, bool]: Whether each animal has truncated.
            Dict[str, Dict[str, Any]]: The info dict for each animal.
        """
        info = self._info

        # First, apply the actions to the animals and step the simulation
        for name, animal in self._animals.items():
            if not animal.config.trainable or animal.config.use_privileged_action:
                assert (
                    animal.config.trainable or name not in action
                ), f"Action for {name} found in action dict. "
                "This isn't allowed for non-trainable animals."
                action[name] = animal.get_action_privileged(self)

            assert name in action, f"Action for {name} not found in action dict."
            animal.apply_action(action[name])
            info[name]["prev_pos"] = animal.pos.copy()
            info[name]["action"] = action[name]

        # Then, step the mujoco simulation
        self._step_mujoco_simulation(self._config.frame_skip, info)

        # We'll then step each animal to render it's current state and get the obs
        obs: Dict[str, Any] = {}
        for name, animal in self._animals.items():
            obs[name] = animal.step()

        # Call helper methods to update the observations, rewards, terminated, and info
        obs = self._update_obs(obs)
        info = self._update_info(info)
        terminated = self._compute_terminated(info)
        truncated = self._compute_truncated(info)
        reward = self._compute_reward(terminated, truncated, info)

        self._episode_step += 1
        self._num_timesteps += 1
        self._cumulative_reward += sum(reward.values())
        self._stashed_cumulative_reward = self._cumulative_reward

        if self.record:
            self._rollout["actions"].append(list(action.values()))
            self._rollout["positions"].append([a.pos for a in self._animals.values()])

        if self.record or "human" in self._config.renderer.render_modes:
            self._overlays["Name"] = self._name
            self._overlays["Total Timesteps"] = self.num_timesteps
            self._overlays["Step"] = self._episode_step
            self._overlays["Cumulative Reward"] = round(self._cumulative_reward, 2)

        return obs, reward, terminated, truncated, info

    def _step_mujoco_simulation(self, n_frames: int, info: Dict[str, Dict[str, Any]]):
        """Sets the mujoco simulation. Will step the simulation `n_frames` times, each
        time checking if the animal has contacts."""
        # Initially set has_contacts to False for all animals
        for name in self._animals:
            info[name]["has_contacts"] = False

        # Check contacts at _every_ step.
        # NOTE: Doesn't process whether hits are terminal or not
        for _ in range(n_frames):
            mj.mj_step(self._model, self._data)

            # Check for contacts. We won't break here, but we'll store whether an
            # animal has contacts or not. If we didn't store during the simulation
            # step, contact checking would only occur after the frame skip, meaning
            # that, if during the course of the frame skip, the animal hits an object
            # and then moves away, the contact would not be detected.
            if self._data.ncon > 0:
                for name, animal in self._animals.items():
                    if not info[name]["has_contacts"]:
                        # Only check for has contacts if it hasn't been set to True
                        # This reduces redundant checks
                        info[name]["has_contacts"] = animal.has_contacts

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mj.mj_rnePostConstraint(self._model, self._data)

    def _update_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Overridable method to update the observations. This class will just return
        the observations as is, but subclasses can override this method to provide
        custom observation updates."""
        return obs

    def _update_info(
        self, info: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Overridable method to update the info. This class will just return the info
        as is, but subclasses can override this method to provide custom info updates.
        """
        for name, animal in self._animals.items():
            info[name]["qpos"] = animal.qpos

        return info

    def _compute_terminated(self, info: Dict[str, Any]) -> Dict[str, bool]:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure.

        The default implementation will always return False for all animals. This can
        be overridden in subclasses to provide custom termination conditions.
        """

        terminated: Dict[str, bool] = {}
        for name, animal in self._animals.items():
            terminated[name] = self._config.termination_fn(self, animal, info[name])

        return terminated

    def _compute_truncated(self, info: Dict[str, Any]) -> bool:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure.

        The default implementation will always return False for all animals. This can
        be overridden in subclasses to provide custom termination conditions.
        """

        truncated: Dict[str, bool] = {}
        for name, animal in self._animals.items():
            truncated[name] = self._config.truncation_fn(self, animal, info[name])

        return truncated

    def _compute_reward(
        self,
        terminated: Dict[str, bool],
        truncated: Dict[str, bool],
        info: Dict[str, Any],
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
        for name, animal in self._animals.items():
            rewards[name] = self._config.reward_fn(
                self, animal, terminated[name], truncated[name], info[name]
            )

        return rewards

    def render(self) -> Dict[str, np.ndarray]:
        """Renders the environment.

        Returns:
            Dict[str, np.ndarray]: The rendered image for each render mode mapped to
                its corresponding str.

        TODO:
            - Make the cursor stuff clearer
        """

        assert self._renderer is not None, "Renderer has not been initialized! "
        "Ensure `use_renderer` is set to True in the constructor."

        renderer = self._renderer
        renderer_width = renderer.width
        renderer_height = renderer.height

        if self._config.render_animal_composite_only:
            # NOTE: Uses the first animal
            animal = next(iter(self._animals.values()))
            if (composite := animal.create_composite_image()) is None:
                composite = np.zeros((1, 1, 3), dtype=np.float32)

            composite = np.flipud(composite)
            composite = np.transpose(composite, (1, 0, 2))
            composite = resize_with_aspect_fill(
                composite, renderer_width, renderer_height
            )
            if renderer._record:
                renderer._rgb_buffer.append(composite)
            return composite

        # Add site overlays for each animal
        i = self._num_resets * self._max_episode_steps + self._episode_step
        size = self._model.stat.extent * 1e-2
        for animal in self._animals.values():
            # Define a unique id for the site
            key = f"{animal.name}_pos_{i}"

            # If the animal is contacting an object, the color will be red; blue
            # otherwise
            color = (1, 0, 0, 1) if animal.has_contacts else (0, 1, 1, 1)

            # Add the overlay
            overlay = MjCambrianSiteViewerOverlay(animal.pos.copy(), color, size)
            self._overlays[key] = overlay

        overlays: List[MjCambrianViewerOverlay] = []

        cursor = MjCambrianCursor(x=0, y=renderer_height - TEXT_MARGIN * 2)
        for key, value in self._overlays.items():
            if issubclass(type(value), MjCambrianViewerOverlay):
                overlays.append(value)
            # else:
            #     cursor.y -= TEXT_HEIGHT + TEXT_MARGIN
            #     overlays.append(MjCambrianTextViewerOverlay(f"{key}: {value}", cursor))

        if not self._config.add_overlays:
            return renderer.render(overlays=overlays)

        # Set the overlay size to be a fraction of the renderer size relative to
        # the animal count. The overlay height will be set to 35% of the renderer from
        # the bottom
        num_animals = len(self._animals)
        overlay_width = int(renderer_width // num_animals) if num_animals > 0 else 0
        overlay_height = int(renderer_height * 0.35)
        overlay_size = (overlay_width, overlay_height)

        cursor = MjCambrianCursor(0, 0)
        for i, (name, animal) in enumerate(self._animals.items()):
            cursor.x = i * overlay_width
            cursor.y = 0
            if cursor.x + overlay_width > renderer_width:
                self._logger.warning("Renderer width is too small!!")
                continue

            if (composite := animal.create_composite_image()) is None:
                # Make the composite image black so we can still render other overlays
                composite = np.zeros((1, 1, 3), dtype=np.float32)

            # NOTE: flipud here since we always flipud when copying buffer from gpu,
            # and when reading the buffer again after drawing the overlay, it will be
            # flipped again. Flipping here means it will be the right side up. Do the
            # same with transpose
            new_composite = np.transpose(composite, (1, 0, 2))
            new_composite = resize_with_aspect_fill(new_composite, *overlay_size)
            new_composite = np.flipud(new_composite)

            overlays.append(MjCambrianImageViewerOverlay(new_composite * 255.0, cursor))

            cursor.x -= TEXT_MARGIN
            cursor.y -= TEXT_MARGIN
            overlay_text = f"Num Eyes: {animal.num_eyes}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            cursor.y += TEXT_HEIGHT
            if animal.num_eyes > 0:
                eye0 = next(iter(animal.eyes.values()))
                overlay_text = f"Res: {tuple(eye0.config.resolution)}"
                overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
                cursor.y += TEXT_HEIGHT
                overlay_text = f"FOV: {tuple(f'{f:.2f}' for f in eye0.config.fov)}"
                overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
                cursor.y = overlay_height - TEXT_HEIGHT * 2 + TEXT_MARGIN * 2
            overlay_text = f"Animal: {name}"
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))
            overlay_text = (
                f"Action: {', '.join([f'{a: .3f}' for a in animal.last_action])}"
            )
            cursor.y -= TEXT_HEIGHT
            overlays.append(MjCambrianTextViewerOverlay(overlay_text, cursor))

            cursor.x += overlay_width
            cursor.y = 0

        return renderer.render(overlays=overlays)

    @property
    def name(self) -> str:
        """Returns the name of the environment."""
        return self._name

    @property
    def xml(self) -> MjCambrianXML:
        """Returns the xml for the environment."""
        return self._xml

    @property
    def animals(self) -> Dict[str, MjCambrianAnimal]:
        """Returns the animals in the environment."""
        return self._animals

    @property
    def renderer(self) -> MjCambrianRenderer:
        """Returns the renderer for the environment."""
        return self._renderer

    @property
    def model(self) -> mj.MjModel:
        """Returns the mujoco model for the environment."""
        return self._model

    @property
    def data(self) -> mj.MjData:
        """Returns the mujoco data for the environment."""
        return self._data

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
        return self._num_timesteps

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
        return self._cumulative_reward

    @property
    def stashed_cumulative_reward(self) -> float:
        """Returns the previous cumulative reward."""
        return self._stashed_cumulative_reward

    @property
    def agents(self) -> List[str]:
        """Returns the agents in the environment.

        This is part of the PettingZoo API.
        """
        return list(self._animals.keys())

    @property
    def num_agents(self) -> int:
        """Returns the number of agents in the environment.

        This is part of the PettingZoo API.
        """
        return len(self.agents)

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
        for name, animal in self._animals.items():
            if animal.config.trainable:
                observation_spaces[name] = animal.observation_space
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
        for name, animal in self._animals.items():
            if animal.config.trainable:
                action_spaces[name] = animal.action_space
        return spaces.Dict(action_spaces)

    def observation_space(self, agent: str) -> spaces.Space:
        """Returns the observation space for the given agent.

        This is part of the PettingZoo API.
        """
        assert agent in list(
            self.observation_spaces.keys()
        ), f"Agent {agent} not found. Available: {list(self.observation_spaces.keys())}"
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Returns the action space for the given agent.

        This is part of the PettingZoo API.
        """
        assert agent in list(
            self.action_spaces.keys()
        ), f"Agent {agent} not found. Available: {list(self.action_spaces.keys())}"
        return self.action_spaces[agent]

    def state(self) -> np.ndarray:
        """Returns the state of the environment.

        This is part of the PettingZoo API.
        """
        raise NotImplementedError("Not implemented yet.")

    def set_random_seed(self, seed: int | float):
        """Sets the seed for the environment."""
        from stable_baselines3.common.utils import set_random_seed

        self._logger.info(f"Setting random seed to {seed}")
        set_random_seed(seed)

    @property
    def record(self):
        """Returns whether the environment is recording."""
        return self._record

    @record.setter
    def record(self, value: bool):
        """Sets whether the environment is recording."""
        self._record = value
        self._renderer.record = value

        if not self.record:
            self._rollout.clear()

    def save(self, path: str | Path, *, save_pkl: bool = True, **kwargs):
        """Saves the simulation output to the given path."""
        self._renderer.save(path, **kwargs)

        if save_pkl:
            self._logger.info(f"Saving rollout to {path.with_suffix('.pkl')}")
            with open(path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(self._rollout, f)
            self._logger.debug(f"Saved rollout to {path.with_suffix('.pkl')}")

    def close(self):
        """Closes the environment."""
        pass


if __name__ == "__main__":
    import argparse

    from cambrian.utils.config import MjCambrianConfig, run_hydra

    REGISTRY = {}

    def register_fn(fn: Callable):
        REGISTRY[fn.__name__] = fn
        return fn

    @register_fn
    def run_mj_viewer(config: MjCambrianConfig, **__):
        import mujoco.viewer

        env = config.env.instance(config.env)
        env.reset(seed=config.seed)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                # env.step(env.action_spaces.sample())
                viewer.sync()

    @register_fn
    def run_renderer(config: MjCambrianConfig, *, record: Any, **__):
        config.save(config.expdir / "config.yaml")

        env = config.env.instance(config.env)

        if max_steps := record:
            env.record = True

        env.reset(seed=config.seed)
        env.xml.write(config.expdir / "env.xml")

        action = {
            name: [-1.0, -0.0] for name, a in env.animals.items() if a.config.trainable
        }

        if "human" in config.env.renderer.render_modes:
            import glfw

            def custom_key_callback(_, key, *args, **__):
                if key == glfw.KEY_R:
                    env.reset()
                elif key == glfw.KEY_UP:
                    name = next(iter(action.keys()))
                    action[name][0] += 0.001
                    action[name][0] = min(1.0, action[name][0])
                elif key == glfw.KEY_DOWN:
                    name = next(iter(action.keys()))
                    action[name][0] -= 0.001
                    action[name][0] = max(-1.0, action[name][0])
                elif key == glfw.KEY_LEFT:
                    name = next(iter(action.keys()))
                    action[name][1] -= 0.01
                    action[name][1] = max(-1.0, action[name][1])
                elif key == glfw.KEY_RIGHT:
                    name = next(iter(action.keys()))
                    action[name][1] += 0.01
                    action[name][1] = min(1.0, action[name][1])

            env.renderer.viewer.custom_key_callback = custom_key_callback

        composites: Dict[str, List[np.ndarray]] = {}
        while env.renderer.is_running():
            if max_steps and env.episode_step > max_steps:
                break

            env.step(action.copy())
            env.render()

            if record:
                for name, animal in env.animals.items():
                    if (composite := animal.create_composite_image()) is not None:
                        composite = np.transpose(composite, (1, 0, 2))
                        composite = resize_with_aspect_fill(composite, 256, 256)
                        composite = (composite * 255).astype(np.uint8)
                        composites.setdefault(name, []).append(composite)

        if record:
            env.save(
                config.expdir / "eval",
                save_pkl=False,
                save_mode=MjCambrianRendererSaveMode.MP4
                | MjCambrianRendererSaveMode.GIF
                | MjCambrianRendererSaveMode.PNG,
            )

            # Save composites
            for name, composites in composites.items():
                import imageio

                path = config.expdir / f"{name}_composite.mp4"
                writer = imageio.get_writer(path, fps=30)
                for composite in composites:
                    writer.append_data(composite)
                writer.close()

                duration = 1000 / 30
                imageio.mimwrite(
                    path.with_suffix(".gif"), composites, duration=duration
                )

    def main(config: MjCambrianConfig, *, fn: str, **kwargs):
        if fn not in REGISTRY:
            raise ValueError(f"Unknown function {fn}")
        REGISTRY[fn](config, **kwargs)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fn", type=str, help="The method to run.", choices=REGISTRY.keys()
    )
    parser.add_argument(
        "--record",
        nargs="?",
        type=int,
        const=100,
        help="Record the simulation. Pass an optional int to specify how many "
        "iterations to record.",
        default=None,
    )
    run_hydra(main, parser=parser)
