from typing import Dict, Any, Tuple, List, Optional, Callable, Concatenate, Self
from pathlib import Path
import pickle

import numpy as np
import gymnasium as gym
import mujoco as mj
from gymnasium import spaces
from stable_baselines3.common.utils import set_random_seed

from cambrian.animal import MjCambrianAnimal, MjCambrianPointAnimal, MjCambrianAnimalConfig
from cambrian.renderer import (
    MjCambrianRenderer,
    MjCambrianRendererConfig,
    MjCambrianViewerOverlay,
    MjCambrianTextViewerOverlay,
    MjCambrianImageViewerOverlay,
    MjCambrianCursor,
    resize_with_aspect_fill,
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
        self.logger = get_logger(self.config)

        self.animals: Dict[str, MjCambrianAnimal] = {}
        self._create_animals()

        self.xml = self.generate_xml()

        self.model = mj.MjModel.from_xml_string(self.xml.to_string())
        self.data = mj.MjData(self.model)

        self.render_mode = (
            "human" if "human" in self.config.renderer.render_modes else "rgb_array"
        )
        self.renderer: MjCambrianRenderer = None
        if renderer_config := self.config.renderer:
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
            # The env renderer can see all sites and geoms
            # This is done by setting the _all_ sitegroups and geomgroups to True
            self.renderer.set_option("sitegroup", True, slice(None))
            self.renderer.set_option("geomgroup", True, slice(None))

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

        self._overlays["Exp"] = self.config.expname

        return self._update_obs(obs), self._update_info(info)

    def step(self, action: Dict[str, Any]) -> Tuple[
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
