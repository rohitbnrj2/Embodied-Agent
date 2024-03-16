from typing import Dict, Any, Tuple, Optional

import numpy as np
import mujoco as mj
from gymnasium import spaces

from cambrian.envs.env import MjCambrianEnvConfig, MjCambrianEnv
from cambrian.utils import get_body_id
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.cambrian_xml import MjCambrianXML, MjCambrianXMLConfig
from cambrian.renderer import resize_with_aspect_fill

@config_wrapper
class MjCambrianObjectConfig(MjCambrianBaseConfig):
    """Defines a config for an object in the environment.

    Attributes:
        xml: The xml for the object.

        terminate_if_close (bool): Whether to terminate the episode if the animal is
            close to the object. Termination indicates success.
        truncate_if_close (bool): Whether to truncate the episode if the animal is
            close to the object. Truncation indicates failure.
        reward_if_close (float): The reward to give the animal if it is close to the
            object.
        distance_to_object_threshold (float): The distance to the object at which the
            animal is assumed to be close to the object.

        use_as_obs (bool): Whether to use the object as an observation or not.
    """

    xml: MjCambrianXMLConfig

    terminate_if_close: bool
    truncate_if_close: bool
    reward_if_close: float
    distance_to_target_threshold: float

    use_as_obs: bool

    pos: Tuple[float, float, float]


@config_wrapper
class MjCambrianObjectEnvConfig(MjCambrianEnvConfig):
    """Defines a config for the cambrian environment with objects.

    Attributes:
        objects (Dict[str, MjCambrianObjectConfig]): The objects in the environment.
    """

    objects: Dict[str, MjCambrianObjectConfig]


class MjCambrianObjectEnv(MjCambrianEnv):
    """This is a subclass of `MjCambrianEnv` that adds support for goals."""

    def __init__(self, config: MjCambrianObjectEnvConfig):
        super().__init__(config)
        self.config: MjCambrianObjectEnvConfig  # type hint to our custom config version

        self.objects: Dict[str, MjCambrianObject] = {}
        self._create_objects()

    def _create_objects(self):
        """Creates the objects in the environment."""
        for name, obj_config in self.config.objects.items():
            self.objects[name] = MjCambrianObject(obj_config)

    def generate_xml(self) -> MjCambrianXML:
        """Generates the xml for the environment."""
        xml = super().generate_xml()

        # TODO: Add targets
        for obj in self.objects.values():
            xml += obj.generate_xml()

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
        # Reset each object
        # Update before super since it calls _update_obs and _update_info
        for obj in self.objects.values():
            obj.reset(self.model)

        return super().reset(seed=seed, options=options)

    def _update_obs(self, obs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Updates the observations for the environment."""
        obs = super()._update_obs(obs)

        # Update the object observations
        for name, obj in self.objects.items():
            if obj.config.use_as_obs:
                obs[name] = obj.config.pos

        return obs

    def _update_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Updates the info for the environment."""
        info = super()._update_info(info)

        # Update the object info
        for name, obj in self.objects.items():
            info["objects"][name] = obj.config.pos

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
        rewards = super()._compute_reward(terminated, truncated, info)

        for name, animal in self.animals.items():
            # Early exits
            if terminated[name] or truncated[name]:
                continue

            # Check if the animal is at the object
            for obj in self.objects.values():
                if obj.is_close(animal.pos):
                    rewards[name] += obj.config.reward_if_close

        return rewards

    def _compute_terminated(self) -> Dict[str, bool]:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure."""

        terminated = super()._compute_terminated()

        # Check if any animals are at the object
        for name, animal in self.animals.items():
            for obj in self.objects.values():
                if obj.is_close(animal.pos):
                    terminated[name] |= obj.config.terminate_if_close

        return terminated

    def _compute_truncated(self) -> bool:
        """Compute whether the env has terminated. Termination indicates success,
        whereas truncated indicates failure."""

        truncated = super()._compute_truncated()

        # Check if any animals are at the object
        for name, animal in self.animals.items():
            for obj in self.objects.values():
                if obj.is_close(animal.pos):
                    truncated[name] |= obj.config.truncate_if_close

        return truncated

    @property
    def observation_spaces(self) -> spaces.Dict:
        """Creates the observation spaces. Identical to `MjCambrianEnv` but with the
        addition of the object observations, if desired."""

        observation_spaces = super().observation_spaces

        # Add the object observations
        for animal_name in self.animals:
            observation_space = observation_spaces.spaces[animal_name]

            for name, obj in self.objects.items():
                if obj.config.use_as_obs:
                    observation_space.spaces[name] = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                    )

        return observation_spaces


class MjCambrianObject:
    def __init__(self, config: MjCambrianObjectConfig, name: str):
        self.config = config
        self.name = name

    def generate_xml(self) -> MjCambrianXML:
        return MjCambrianXML.from_string(self.config.xml)

    def reset(self, model: mj.MjModel) -> np.ndarray:
        body_id = get_body_id(model, f"{name}_body")
        assert body_id != -1, f"Body {name}_body not found in model"

        model.body_pos[body_id] = self.config.pos

        return model.body_pos[body_id]

    def is_close(self, pos: np.ndarray) -> bool:
        return (
            np.linalg.norm(pos - self.config.pos)
            < self.config.distance_to_target_threshold
        )


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
