from typing import Any, List, Tuple, TYPE_CHECKING, Optional, Callable, Dict, Generator
from types import ModuleType
from pathlib import Path
from dataclasses import dataclass
import contextlib
import ast
import re
import torch

import mujoco as mj
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

if TYPE_CHECKING:
    from cambrian.ml.model import MjCambrianModel

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
    from cambrian.envs import MjCambrianEnv
    from cambrian.utils.logger import get_logger

    cambrian_env: MjCambrianEnv = env.envs[0].unwrapped
    if record_kwargs is not None:
        # don't set to `record_path is not None` directly bc this will delete overlays
        cambrian_env.record = True

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


def calculate_fitness(evaluations_npz: Path) -> float:
    """Calculate the fitness of the animal. This is done by taking the 3rd quartile of
    the evaluation rewards."""
    # Return negative infinity if the evaluations file doesn't exist
    if not evaluations_npz.exists():
        return -float("inf")

    def top_25_excluding_outliers(data: np.ndarray) -> float:
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        filtered_data = data[(data > q1 - 1.5 * iqr) & (data < q3 + 1.5 * iqr)]
        return float(np.mean(np.sort(filtered_data)[-len(filtered_data) // 4 :]))

    # The rewards array will be stored in a 2D array where each row represents each
    # evaluation run and each column represents the rewards for each evaluation step.
    # We may run multiple steps of the same or slightly different environment to reduce
    # variance. We will average the rewards across each row to get the final rewards.
    rewards = np.load(evaluations_npz)["results"]
    rewards = np.mean(rewards, axis=1)

    return top_25_excluding_outliers(rewards)


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

def is_integer(maybe_int: Any):
    return isinstance(maybe_int, int) or np.all(np.mod(maybe_int, 1) == 0)


def make_odd(num: int | float) -> int:
    """Make a number odd by adding 1 if it is even. If `num` is a float, it is cast to
    an int."""
    return int(num) if num % 2 == 1 else int(num) + 1


def safe_index(
    arr: List[Any], value: Any, *, default: Optional[int] = None
) -> int | None:
    """Safely get the index of a value in a list. If the value is not in the list, None
    is returned."""
    try:
        return arr.index(value)
    except ValueError:
        return default


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


def get_sensor_id(model: mj.MjModel, sensor_name: str) -> int:
    """Get the ID of a Mujoco sensor."""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, sensor_name)


def get_sensor_name(model: mj.MjModel, sensoradr: int) -> str:
    """Get the name of a Mujoco sensor."""
    return mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, sensoradr)


@dataclass
class MjCambrianActuator:
    """Helper class which stores information about a Mujoco actuator.

    Attributes:
        adr (int): The Mujoco actuator ID (index into model.actuator_* arrays).
        trnadr (int): The index of the actuator's transmission in the model.
        low (float): The lower bound of the actuator's range.
        high (float): The upper bound of the actuator's range.
    """

    adr: int
    trnadr: int
    low: float
    high: float


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
        group (int): The geometry group the geometry belongs to.
    """

    id: int
    rbound: float
    pos: np.ndarray
    group: int


# =============
# Misc utils


def format_string_with_obj_attributes(s, obj):
    """
    Replaces placeholders in a string with attribute values from a provided object.

    Args:
    s (str): The string containing placeholders in the format {attr.path}.
    obj (object): The object from which to fetch the attribute values.

    Returns:
    str: The formatted string with attribute values.

    Examples:
    >>> class Env:
    ...     def __init__(self):
    ...         self.test = "working"
    ...
    >>> class CustomPythonClass:
    ...     def __init__(self):
    ...         self.env = Env()
    ...
    >>> obj = CustomPythonClass()
    >>> format_string_with_obj_attributes("Test: {env.test}", obj)
    'Test: working'
    """

    def get_nested_attr(obj, attr_path):
        """Fetches the value of a nested attribute by traversing through the object attributes based on the dot-separated path."""
        for attr in attr_path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj

    def replace_attr(match):
        """Helper function to replace each placeholder in the string with the corresponding attribute value."""
        path = match.group(1)
        return str(get_nested_attr(obj, path))

    return re.sub(r"\{([^}]+)\}", replace_attr, s)


def literal_eval_with_callables(
    node_or_string, safe_callables: Dict[str, Callable] = {}, *, _env={}
):
    """
    Safely evaluate an expression node or a string containing a Python expression.
    The expression can contain literals, lists, tuples, dicts, unary and binary
    operators. Calls to functions specified in 'safe_callables' dictionary are allowed.

    Args:
        node_or_string (ast.Node or str): The expression node or string to evaluate.
        safe_callables (Dict[str, Callable]): A dictionary mapping function names to
            callable Python objects. Only these functions can be called within the
            expression.

    Returns:
        The result of the evaluated expression.

    Raises:
        ValueError: If the expression contains unsupported or malformed nodes or tries
            to execute unsupported operations.

    Examples:
        >>> literal_eval_with_callables("1 + 2")
        3
        >>> literal_eval_with_callables("sqrt(4)", {'sqrt': math.sqrt})
        2.0
    """
    if isinstance(node_or_string, str):
        node = ast.parse(node_or_string, mode="eval").body
    else:
        node = node_or_string

    # Copy safe_callables into _env to allow names to evaluate to callables, like if
    # a module is specified in safe_callables and they have attributes/constants.
    _env = {**safe_callables, **_env}

    op_map = {
        ast.Add: lambda x, y: x + y,
        ast.Sub: lambda x, y: x - y,
        ast.Mult: lambda x, y: x * y,
        ast.Div: lambda x, y: x / y,
        ast.Mod: lambda x, y: x % y,
        ast.Pow: lambda x, y: x**y,
        ast.FloorDiv: lambda x, y: x // y,
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
        ast.Eq: lambda x, y: x == y,
        ast.NotEq: lambda x, y: x != y,
        ast.Lt: lambda x, y: x < y,
        ast.LtE: lambda x, y: x <= y,
        ast.Gt: lambda x, y: x > y,
        ast.GtE: lambda x, y: x >= y,
        ast.Is: lambda x, y: x is y,
        ast.IsNot: lambda x, y: x is not y,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
        ast.BitAnd: lambda x, y: x & y,
        ast.BitOr: lambda x, y: x | y,
        ast.BitXor: lambda x, y: x ^ y,
        ast.LShift: lambda x, y: x << y,
        ast.RShift: lambda x, y: x >> y,
        ast.Invert: lambda x: ~x,
    }

    def _convert(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in _env:
                return _env[node.id]
        elif isinstance(node, ast.Attribute):
            obj = _convert(node.value)
            if isinstance(obj, ModuleType) and hasattr(obj, node.attr):
                attribute = getattr(obj, node.attr)
                return attribute if callable(attribute) else attribute
        elif isinstance(node, (ast.Tuple, ast.List)):
            return type(node.elts)(map(_convert, node.elts))
        elif isinstance(node, ast.Dict):
            return {_convert(k): _convert(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.UnaryOp):
            operand = _convert(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        elif isinstance(node, ast.BinOp) and type(node.op) in op_map:
            left = _convert(node.left)
            right = _convert(node.right)
            return op_map[type(node.op)](left, right)
        elif isinstance(node, ast.BoolOp) and type(node.op) in op_map:
            return op_map[type(node.op)](
                _convert(node.values[0]), _convert(node.values[1])
            )
        elif isinstance(node, ast.Compare) and type(node.ops[0]) in op_map:
            left = _convert(node.left)
            right = _convert(node.comparators[0])
            return op_map[type(node.ops[0])](left, right)
        elif isinstance(node, ast.IfExp):
            return _convert(node.body) if _convert(node.test) else _convert(node.orelse)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                obj = safe_callables.get(node.func.value.id)
                if obj is not None and hasattr(obj, node.func.attr):
                    method = getattr(obj, node.func.attr)
                    if callable(method):
                        return method(
                            *map(_convert, node.args),
                            **{kw.arg: _convert(kw.value) for kw in node.keywords},
                        )
            elif isinstance(node.func, ast.Name) and node.func.id in safe_callables:
                func = safe_callables[node.func.id]
                if callable(func):
                    return func(
                        *map(_convert, node.args),
                        **{kw.arg: _convert(kw.value) for kw in node.keywords},
                    )
        elif isinstance(node, ast.GeneratorExp):
            iter_node = node.elt
            results = []

            for comprehension in node.generators:
                iter_list = _convert(comprehension.iter)
                for item in iter_list:
                    _env[comprehension.target.id] = item
                    if all(_convert(cond) for cond in comprehension.ifs):
                        results.append(_convert(iter_node))

            return results
        elif isinstance(node, ast.ListComp):
            return _convert(ast.GeneratorExp(node.elt, node.generators))
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

        raise ValueError(f"Couldn't parse node ({type(node)}): {ast.dump(node)}")

    return _convert(node)


def safe_eval(src: Any):
    """This method will evaluate the source code in a safe manner. This is useful for
    evaluating expressions in the config file. This will only allow certain builtins,
    numpy, and will not allow any other code execution."""
    import math

    supported_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "list": list,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }
    return literal_eval_with_callables(src, {"math": math, **supported_builtins})


def calc_mtf_sum(psf, camera_px_pitch, face_freq, noise_floor=0.0):
    """
    Calculate MTF and the sum of MTF values for indices lower than max_index using PyTorch.

    Parameters:
    - psf: CxHxW tensor representing the point spread function.
    - camera_px_pitch: Pixel pitch of the camera in millimeters.
    - max_index: The threshold index below which the sum of MTF values is calculated.

    Returns:
    - frequencyAxis: The frequency axis for the MTF.
    - radialAvg: The radial average of the MTF.
    - sum_below_max_index: The sum of MTF values for indices lower than max_index.
    """
    # Calculate OTF
    otf = torch.fft.fftshift(torch.fft.fft2(psf), dim=(2, 3))

    mtf_orig = torch.abs(otf)
    # Calculate MTF
    mtf = torch.abs(otf) - noise_floor

    # Compute the radial average and sum below max_index
    _, _, rows, cols = mtf.shape
    if rows % 2 == 0:
        x = torch.linspace(-cols // 2, cols // 2 - 1, steps=cols)
        y = torch.linspace(-rows // 2, rows // 2 - 1, steps=rows)
    else:
        x = torch.linspace(-cols // 2 + 1, cols // 2, steps=cols)
        y = torch.linspace(-rows // 2 + 1, rows // 2, steps=rows)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    #     import pdb; pdb.set_trace()
    R = torch.sqrt(X**2 + Y**2)
    R = torch.round(R).type(torch.int64)
    #     import pdb; pdb.set_trace()

    # Define the pixel pitch (in millimeters)
    pixelPitch = camera_px_pitch * 1e3

    # print("pixel pitch (mm) :", pixelPitch)

    # Calculate Nyquist Frequency (in cycles/mm)
    nyquistFreq = 1 / (2 * pixelPitch)

    # Number of frequency points
    numPoints = rows // 2 + (0 if rows % 2 == 0 else 1)
    sensorSize = pixelPitch * numPoints
    # Get the lowest frequency
    lowestFreq = 1 / (sensorSize)
    frequencyAxis = torch.linspace(lowestFreq, nyquistFreq, steps=numPoints)

    for ind, freq in enumerate(frequencyAxis):
        if freq > face_freq:
            max_index = ind
            # print(max_index)
            break

    maxRadius = torch.round(torch.max(X)).type(torch.int64)

    mask_end = R <= maxRadius

    mask_in = R < max_index
    mask_out = (R >= max_index) * mask_end
    mtf_r = mtf[0, 0, ...]
    mtf_g = mtf[0, 1, ...]
    mtf_b = mtf[0, 2, ...]

    mtf_r_orig = mtf_orig[0, 0, ...]
    mtf_g_orig = mtf_orig[0, 1, ...]
    mtf_b_orig = mtf_orig[0, 2, ...]

    sum_value_in_r = torch.sum(mtf_r[mask_in])
    sum_value_in_g = torch.sum(mtf_g[mask_in])
    sum_value_in_b = torch.sum(mtf_b[mask_in])

    sum_value_out_r = torch.sum(mtf_r[mask_out])
    sum_value_out_g = torch.sum(mtf_g[mask_out])
    sum_value_out_b = torch.sum(mtf_b[mask_out])

    radialAvg_R = torch.zeros(maxRadius + 1, dtype=mtf.dtype)
    radialAvg_G = torch.zeros(maxRadius + 1, dtype=mtf.dtype)
    radialAvg_B = torch.zeros(maxRadius + 1, dtype=mtf.dtype)

    for r in range(maxRadius + 1):
        mask = R == r
        #         import pdb; pdb.set_trace()
        if torch.any(mask):
            radial_value = torch.mean(mtf_r_orig[mask])
            radialAvg_R[r] = radial_value
            radial_value = torch.mean(mtf_g_orig[mask])
            radialAvg_G[r] = radial_value
            radial_value = torch.mean(mtf_b_orig[mask])
            radialAvg_B[r] = radial_value

    return (
        sum_value_in_r,
        sum_value_in_g,
        sum_value_in_b,
        sum_value_out_r,
        sum_value_out_g,
        sum_value_out_b,
        frequencyAxis,
        mtf_orig,
        radialAvg_R,
        radialAvg_G,
        radialAvg_B,
        max_index,
    )
