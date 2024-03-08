import argparse
from typing import Any, List, Tuple, TYPE_CHECKING, Optional, Callable, Dict, Generator
from pathlib import Path
from dataclasses import dataclass
import contextlib
from functools import partial

import gymnasium as gym
import mujoco as mj
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

if TYPE_CHECKING:
    from cambrian.utils.config import MjCambrianConfig
    from cambrian.ml.model import MjCambrianModel


def safe_index(list_to_index: List[Any], value: Any) -> int:
    """Safely get the index of a value in a list, or -1 if not found. Normally,
    list.index() throws an exception if the value is not found."""
    try:
        return list_to_index.index(value)
    except ValueError:
        return -1


def get_include_path(
    model_path: str | Path, *, throw_error: bool = True
) -> Path | None:
    """Tries to find the model path. `model_path` can either be relative to the
    execution file, absolute, or relative to the path of the cambrian folder. The
    latter is the typical method, where `assets/<model>.xml` specifies the model path
    located in REPO_PATH/models/assets/<model>.xml.

    If the file can't be found, a FileNotFoundError is raised if throw_error is True. If
    throw_error is False, None is returned.
    """
    path = Path(model_path)
    if path.exists():
        pass
    elif (rel_path := Path(__file__).parent / path).exists():
        path = rel_path
    else:
        if throw_error:
            raise FileNotFoundError(f"Could not find path `{model_path}`.")
        else:
            return None

    return path


# ============


def evaluate_policy(
    env: VecEnv,
    model: "MjCambrianModel",
    num_runs: int,
    *,
    record_kwargs: Optional[Dict[str, Any]] = None,
    step_callback: Optional[Callable[[], bool]] = lambda: True,
    done_callback: Optional[Callable[[int], bool]] = lambda _: True,
):
    """Evaluate a policy.

    Args:
        env (gym.Env): The environment to evaluate the policy on. Assumed to be a
            VecEnv wrapper around a MjCambrianEnv.
        model (MjCambrianModel): The model to evaluate.
        num_runs (int): The number of runs to evaluate the policy on.

    Keyword Args:
        record_path (Optional[Path]): The path to save the video to. If None, the video
            is not saved. This is passed directly to MjCambrianEnv.renderer.save(), so
            see that method for more details.
    """
    # To avoid circular imports
    from cambrian.envs.env import MjCambrianEnv
    from cambrian.utils.logger import get_logger

    cambrian_env: MjCambrianEnv = env.envs[0].unwrapped
    if record_kwargs is not None:
        # don't set to `record_path is not None` directly bc this will delete overlays
        cambrian_env.record = True

    prev_init_goal_pos = None
    if (eval_goal_pos := cambrian_env.maze.config.eval_goal_pos) is not None:
        prev_init_goal_pos = cambrian_env.maze.config.init_goal_pos
        cambrian_env.maze.config.init_goal_pos = eval_goal_pos

    run = 0
    obs = env.reset()
    get_logger().info(f"Starting {num_runs} evaluation run(s)...")
    while run < num_runs:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        if done:
            get_logger().info(
                f"Run {run} done. Cumulative reward: {cambrian_env.cumulative_reward}"
            )

            if not done_callback(run):
                break

            run += 1

        env.render()

        if not step_callback():
            break

    if record_kwargs is not None:
        cambrian_env.save(**record_kwargs)
        cambrian_env.record = False

    if prev_init_goal_pos is not None:
        cambrian_env.maze.config.init_goal_pos = prev_init_goal_pos


# =============


class MjCambrianArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("config", type=str, help="Path to config file")
        self.add_argument(
            "-o",
            "--override",
            "--overrides",
            dest="overrides",
            action="extend",
            nargs="+",
            type=str,
            help="Override config values. Do <config>.<key>=<value>",
            default=[],
        )
        self.add_argument(
            "--defaults",
            type=str,
            action="extend",
            nargs="+",
            help="Path to yaml files containing defaults. Merged with the config file.",
            default=[],
        )

        self._args = None

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        self._args = args

        return args

    def parse_config(self, **kwargs) -> "MjCambrianConfig":
        assert self._args is not None, "parse_args() must be called first"

        from cambrian.utils.config import MjCambrianConfig

        return MjCambrianConfig.load(
            self._args.config,
            overrides=self._args.overrides,
            defaults=self._args.defaults,
            **kwargs,
        )


# =============


def generate_sequence_from_range(range: Tuple[float, float], num: int) -> List[float]:
    """"""
    return [np.average(range)] if num == 1 else np.linspace(*range, num)


def merge_dicts(d1: dict, d2: dict) -> dict:
    """Merge two dictionaries. d2 takes precedence over d1."""
    return {**d1, **d2}


@contextlib.contextmanager
def setattrs_temporary(
    *args: Tuple[Any, Dict[str, Any]]
) -> Generator[None, None, None]:
    """Temporarily set attributes of an object."""
    prev_values = []
    for obj, kwargs in args:
        prev_values.append({})
        for attr, value in kwargs.items():
            if isinstance(obj, dict):
                prev_values[-1][attr] = obj[attr]
                obj[attr] = value
            else:
                prev_values[-1][attr] = getattr(obj, attr)
                setattr(obj, attr, value)
    yield
    for (obj, _), kwargs in zip(args, prev_values):
        for attr, value in kwargs.items():
            if isinstance(obj, dict):
                obj[attr] = value
            else:
                setattr(obj, attr, value)


def get_gpu_memory_usage(return_total_memory: bool = True) -> Tuple[float, float]:
    """Get's the total and used memory of the GPU in GB."""
    assert torch.cuda.is_available(), "No CUDA device available"
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    free_memory = torch.cuda.mem_get_info()[0] / 1024**3
    used_memory = total_memory - free_memory

    if return_total_memory:
        return used_memory, total_memory
    else:
        return used_memory


def get_observation_space_size(observation_space: gym.spaces.Space) -> int:
    """Get the size of an observation space. Returns size in GB."""
    if isinstance(observation_space, gym.spaces.Box):
        return np.prod(observation_space.shape) / 1024**3
    elif isinstance(observation_space, gym.spaces.Discrete):
        return observation_space.n / 1024**3
    elif isinstance(observation_space, gym.spaces.Tuple):
        return sum(
            get_observation_space_size(space) for space in observation_space.spaces
        )
    elif isinstance(observation_space, gym.spaces.Dict):
        return sum(
            get_observation_space_size(space)
            for space in observation_space.spaces.values()
        )
    else:
        raise ValueError(
            f"Unsupported observation space type: {type(observation_space)}"
        )


# =============
# Mujoco utils


def get_body_id(model: mj.MjModel, body_name: str) -> int:
    """Get the ID of a Mujoco body."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)


def get_body_name(model: mj.MjModel, bodyadr: int) -> str:
    """Get the name of a Mujoco body."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bodyadr)


def get_geom_id(model: mj.MjModel, geom_name: str) -> int:
    """Get the ID of a Mujoco geometry."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)


def get_geom_name(model: mj.MjModel, geomadr: int) -> str:
    """Get the name of a Mujoco geometry."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geomadr)


def get_site_id(model: mj.MjModel, site_name: str) -> int:
    """Get the ID of a Mujoco geometry."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)


def get_site_name(model: mj.MjModel, siteadr: int) -> str:
    """Get the name of a Mujoco geometry."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, siteadr)


def get_joint_id(model: mj.MjModel, joint_name: str) -> int:
    """Get the ID of a Mujoco geometry."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)


def get_joint_name(model: mj.MjModel, jointadr: int) -> str:
    """Get the name of a Mujoco geometry."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jointadr)


def get_camera_id(model: mj.MjModel, camera_name: str) -> int:
    """Get the ID of a Mujoco camera."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name)


def get_camera_name(model: mj.MjModel, cameraadr: int) -> str:
    """Get the name of a Mujoco camera."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, cameraadr)


def get_light_id(model: mj.MjModel, light_name: str) -> int:
    """Get the ID of a Mujoco light."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_LIGHT, light_name)


def get_light_name(model: mj.MjModel, lightadr: int) -> str:
    """Get the name of a Mujoco light."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_LIGHT, lightadr)


@dataclass
class MjCambrianJoint:
    """Helper class which stores information about a Mujoco joint.

    Attributes:
        adr (int): The Mujoco joint ID (index into model.jnt_* arrays).
        qposadr (int): The index of the joint's position in the qpos array.
        numqpos (int): The number of positions in the joint.
        qveladr (int): The index of the joint's velocity in the qvel array.
        numqvel (int): The number of velocities in the joint.
    """

    adr: int
    qposadr: int
    numqpos: int
    qveladr: int
    numqvel: int

    @staticmethod
    def create(model: mj.MjModel, jntadr: int) -> "MjCambrianJoint":
        """Create a Joint object from a Mujoco model and joint body ID."""
        qposadr = model.jnt_qposadr[jntadr]
        qveladr = model.jnt_dofadr[jntadr]

        jnt_type = model.jnt_type[jntadr]
        if jnt_type == mj.mjtJoint.mjJNT_FREE:
            numqpos = 7
            numqvel = 6
        elif jnt_type == mj.mjtJoint.mjJNT_BALL:
            numqpos = 4
            numqvel = 3
        else:  # mj.mjtJoint.mjJNT_HINGE or mj.mjtJoint.mjJNT_SLIDE
            numqpos = 1
            numqvel = 1

        return MjCambrianJoint(jntadr, qposadr, numqpos, qveladr, numqvel)


@dataclass
class MjCambrianActuator:
    """Helper class which stores information about a Mujoco actuator.

    Attributes:
        adr (int): The Mujoco actuator ID (index into model.actuator_* arrays).
        low (float): The lower bound of the actuator's range.
        high (float): The upper bound of the actuator's range.
    """

    adr: int
    low: float
    high: float

    @property
    def ctrlrange(self) -> float:
        return self.high - self.low


@dataclass
class MjCambrianGeometry:
    """Helper class which stores information about a Mujoco geometry

    Attributes:
        adr (int): The Mujoco geometry ID (index into model.geom_* arrays).
        rbound (float): The radius of the geometry's bounding sphere.
        pos (np.ndarray): The position of the geometry relative to the body.
    """

    adr: int
    rbound: float
    pos: np.ndarray



def mujoco_wrapper(instance, **kwargs):
    """This wrapper will wrap a mujoco class and convert it into a dataclass which we
    can use to build structured configs. Mujoco classes don't have __init__ methods,
    so we'll use the __dict__ to get the fields of the class.

    Should be called as follows:
    _target_: cambrian.utils.config.mujoco_wrapper
    instance:
        _target_: <mujoco_class>
    """

    def setattrs(instance, **kwargs):
        try:
            for key, value in kwargs.items():
                setattr(instance, key, value)
        except Exception as e:
            raise ValueError(
                f"In mujoco_wrapper, got error when setting attribute "
                f"{key=} to {value=}: {e}"
            )
        return instance

    if isinstance(instance, partial):
        # If the instance is a partial, we'll setup a wrapper such that once the
        # partial is actually instantiated, we'll set the attributes of the instance
        # with the kwargs.
        partial_instance = instance
        config_kwargs = kwargs

        def wrapper(*args, **kwargs):
            instance = partial_instance(*args, **kwargs)
            # print(partial_instance, instance)
            return setattrs(instance, **config_kwargs)

        return wrapper
    else:
        return setattrs(instance, **kwargs)

def mujoco_flags_wrapper(instance, key, flag_type, **flags):
    def setattrs(instance, key, flag_type, **flags):
        attr = getattr(instance, key)
        for flag, value in flags.items():
            flag = getattr(flag_type, flag)
            attr[flag] = value
        return attr

    if isinstance(instance, partial):
        partial_instance = instance
        config_key = key
        config_type = flag_type
        config_flags = flags

        def wrapper(*args, **kwargs):
            instance = partial_instance(*args, **kwargs)
            return setattrs(instance, config_key, config_type, **config_flags)

        return wrapper
    else:
        return setattrs(instance, key, flag_type, **flags)