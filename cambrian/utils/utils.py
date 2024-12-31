import contextlib
import pickle
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
)

import mujoco as mj
import numpy as np
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecEnv

from cambrian.utils.logger import get_logger

if TYPE_CHECKING:
    from cambrian.agents import MjCambrianAgent
    from cambrian.ml.model import MjCambrianModel

# ============

device = get_device("auto")

# ============


def evaluate_policy(
    env: VecEnv,
    model: "MjCambrianModel",
    num_runs: int,
    *,
    record_kwargs: Optional[Dict[str, Any]] = None,
    step_callback: Optional[Callable[[], bool]] = lambda: True,
    done_callback: Optional[Callable[[int], bool]] = lambda _: True,
) -> float:
    """Evaluate a policy.

    Args:
        env (gym.Env): The environment to evaluate the policy on. Assumed to be a
            VecEnv wrapper around a MjCambrianEnv.
        model (MjCambrianModel): The model to evaluate.
        num_runs (int): The number of runs to evaluate the policy on.

    Keyword Args:
        record_kwargs (Dict[str, Any]): The keyword arguments to pass to the save
            method of the environment. If None, the environment will not be recorded.
        step_callback (Callable[[], bool]): The callback function to call at each step.
            If the function returns False, the evaluation will stop.
        done_callback (Callable[[int], bool]): The callback function to call when a run
            is done. If the function returns False, the evaluation will stop.

    Returns:
        float: The cumulative reward of the evaluation.
    """
    # To avoid circular imports
    from cambrian.envs import MjCambrianEnv
    from cambrian.utils.logger import get_logger

    cambrian_env: MjCambrianEnv = env.envs[0].unwrapped
    if record_kwargs is not None:
        # don't set to `record_path is not None` directly bc this will delete overlays
        cambrian_env.record()

    run = 0
    obs = env.reset()
    get_logger().info(f"Starting {num_runs} evaluation run(s)...")
    while run < num_runs:
        # get number of parameters
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        if done:
            get_logger().info(
                f"Run {run} done. "
                f"Cumulative reward: {cambrian_env.stashed_cumulative_reward}"
            )

            if not done_callback(run):
                break

            run += 1

        if record_kwargs is not None:
            env.render()

        if not step_callback():
            break

    if record_kwargs is not None:
        cambrian_env.save(**record_kwargs)
        cambrian_env.record(False)

    return cambrian_env.stashed_cumulative_reward


def moving_average(values, window, mode="valid"):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, mode=mode)


# =============


def save_data(data: Any, outdir: Path, pickle_file: Path):
    """Save the parsed data to a pickle file."""
    pickle_file = (outdir / pickle_file).resolve()
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_file, "wb") as f:
        pickle.dump(data, f)
    get_logger().info(f"Saved parsed data to {pickle_file}.")


def try_load_pickle(folder: Path, pickle_file: Path) -> Any | None:
    """Try to load the data from the pickle file."""
    pickle_file = (folder / pickle_file).resolve()
    if pickle_file.exists():
        get_logger().info(f"Loading parsed data from {pickle_file}...")
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        get_logger().info(f"Loaded parsed data from {pickle_file}.")
        return data

    get_logger().warning(f"Could not load {pickle_file}.")
    return None


# =============


def generate_sequence_from_range(
    range: Tuple[float, float], num: int, endpoint: bool = True
) -> List[float]:
    """Generate a sequence of numbers from a range. If num is 1, the average of the
    range is returned. Otherwise, a sequence of numbers is generated using np.linspace.

    Args:
        range (Tuple[float, float]): The range of the sequence.
        num (int): The number of elements in the sequence.

    Keyword Args:
        endpoint (bool): Whether to include the endpoint in the sequence.
    """
    sequence = (
        [np.average(range)] if num == 1 else np.linspace(*range, num, endpoint=endpoint)
    )
    return [float(x) for x in sequence]


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
    try:
        yield
    finally:
        for (obj, _), kwargs in zip(args, prev_values):
            for attr, value in kwargs.items():
                if isinstance(obj, dict):
                    obj[attr] = value
                else:
                    setattr(obj, attr, value)


def is_number(maybe_num: Any) -> bool:
    from numbers import Number

    return isinstance(maybe_num, Number)


def is_integer(maybe_int: Any) -> bool:
    if isinstance(maybe_int, int):
        return True
    if isinstance(maybe_int, str):
        return maybe_int.isdigit() or (
            maybe_int[1:].isdigit() if maybe_int[0] == "-" else False
        )
    if isinstance(maybe_int, np.ndarray):
        return np.all(np.mod(maybe_int, 1) == 0)
    return False


def make_odd(num: int | float) -> int:
    """Make a number odd by adding 1 if it is even. If `num` is a float, it is cast to
    an int."""
    return int(num) if num % 2 == 1 else int(num) + 1


def round_half_up(n: float) -> int:
    """Round a number to the nearest integer, rounding half up."""
    return int(np.floor(n + 0.5))


def safe_index(
    arr: List[Any], value: Any, *, default: Optional[int] = None
) -> int | None:
    """Safely get the index of a value in a list. If the value is not in the list, None
    is returned."""
    try:
        return arr.index(value)
    except ValueError:
        return default


@contextlib.contextmanager
def suppress_stdout_stderr():
    import os
    import sys

    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


@dataclass
class MjCambrianActuator:
    """Helper class which stores information about a Mujoco actuator.

    Attributes:
        adr (int): The Mujoco actuator ID (index into model.actuator_* arrays).
        trnadr (int): The index of the actuator's transmission in the model.
        ctrlrange (Tuple[float, float]): The control range of the actuator.
        ctrllimited (bool): Whether the actuator is control-limited.
    """

    adr: int
    trnadr: int
    ctrlrange: Tuple[float, float]
    ctrllimited: bool


@dataclass
class MjCambrianJoint:
    """Helper class which stores information about a Mujoco joint.

    Attributes:
        type (int): The Mujoco joint type (mj.mjtJoint).
        adr (int): The Mujoco joint ID (index into model.jnt_* arrays).
        qposadr (int): The index of the joint's position in the qpos array.
        numqpos (int): The number of positions in the joint.
        qveladr (int): The index of the joint's velocity in the qvel array.
        numqvel (int): The number of velocities in the joint.
    """

    type: int
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

        return MjCambrianJoint(jnt_type, jntadr, qposadr, numqpos, qveladr, numqvel)

    @property
    def qposadrs(self) -> List[int]:
        """Get the indices of the joint's positions in the qpos array."""
        return list(range(self.qposadr, self.qposadr + self.numqpos))

    @property
    def qveladrs(self) -> List[int]:
        """Get the indices of the joint's velocities in the qvel array."""
        return list(range(self.qveladr, self.qveladr + self.numqvel))


@dataclass
class MjCambrianGeometry:
    """Helper class which stores information about a Mujoco geometry

    Attributes:
        id (int): The Mujoco geometry ID (index into model.geom_* arrays).
        rbound (float): The radius of the geometry's bounding sphere.
        pos (np.ndarray): The position of the geometry relative to the body.
    """

    id: int
    rbound: float
    pos: np.ndarray


# ============


def agent_selected(agent: "MjCambrianAgent", agents: Optional[List[str]]):
    """Check if the agent is selected."""
    return agents is None or any(fnmatch(agent.name, pattern) for pattern in agents)


# =============
# Misc utils


def safe_eval(src: str, additional_vars: Dict[str, Any] = {}) -> Any:
    """
    Evaluate a string containing a Python expression in a safe manner.

    This function uses ``RestrictedPython`` to evaluate the expression,
    only allowing certain built-in functions and types, and any additional variables
    provided. It prevents execution of arbitrary code or access to unauthorized
    functions and methods.

    Args:
        src (str): The source code to evaluate.
        additional_vars (Dict[str, Any]): A dictionary of additional variables or
            functions to include in the evaluation environment.

    Returns:
        Any: The result of the evaluated expression.

    Examples:
        >>> safe_eval("1 + 2")
        3
        >>> safe_eval("max([1, 2, 3])")
        3
        >>> safe_eval("math.sqrt(a)", {'a': 1})
        4.0
    """
    from RestrictedPython import (
        compile_restricted,
        limited_builtins,
        safe_builtins,
        utility_builtins,
    )

    safe_globals = {
        **utility_builtins,
        **safe_builtins,
        **limited_builtins,
    }
    byte_code = compile_restricted(src, filename="<inline code>", mode="eval")
    return eval(byte_code, safe_globals, additional_vars)


# =============


class MjCambrianWrapper:
    def __init__(self, **kwargs):
        self.setattrs(self, **kwargs)

    @staticmethod
    def wrap(instance: Type | Any, **kwargs) -> Any:
        if isinstance(instance, type):
            instance = instance()
        MjCambrianWrapper.setattrs(instance, **kwargs)
        return instance

    @staticmethod
    def setattrs(instance, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                assert hasattr(
                    instance, key
                ), f"'{instance.__class__.__name__}' has no attribute '{key}'"
                MjCambrianWrapper.setattrs(getattr(instance, key), **value)
            else:
                assert hasattr(
                    instance, key
                ), f"'{instance.__class__.__name__}' has no attribute '{key}'"
                setattr(instance, key, value)


def wrap(instance: Type | Any, **kwargs) -> Any:
    return MjCambrianWrapper.wrap(instance=instance, **kwargs)


# =============


def set_matplotlib_style(*, use_scienceplots: bool = True):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme("paper", font_scale=1.5)
    sns.set_style("ticks")

    if use_scienceplots:
        try:
            import scienceplots  # noqa

            plt.style.use(["science", "nature"])
        except ImportError:
            get_logger().warning(
                "SciencePlots not found. Using default matplotlib style."
            )
