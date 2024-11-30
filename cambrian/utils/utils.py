import ast
import contextlib
import pickle
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple

import mujoco as mj
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from cambrian.utils.logger import get_logger

if TYPE_CHECKING:
    from cambrian.agents import MjCambrianAgent
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
                f"Run {run} done. "
                f"Cumulative reward: {cambrian_env.stashed_cumulative_reward}"
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


# ============


def agent_selected(agent: "MjCambrianAgent", agents: Optional[List[str]]):
    """Check if the agent is selected."""
    return agents is None or any(fnmatch(agent.name, pattern) for pattern in agents)


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
        """Fetches the value of a nested attribute by traversing through the object
        attributes based on the dot-separated path."""
        for attr in attr_path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj

    def replace_attr(match):
        """Helper function to replace each placeholder in the string with the
        corresponding attribute value."""
        path = match.group(1)
        return str(get_nested_attr(obj, path))

    return re.sub(r"\{([^}]+)\}", replace_attr, s)


def literal_eval_with_callables(
    node_or_string,
    safe_callables: Dict[str, Callable] = {},
    safe_methods: Dict[Tuple[type, str], Callable] = {},
    *,
    _env={},
):
    """
    Safely evaluate an expression node or a string containing a Python expression.
    The expression can contain literals, lists, tuples, dicts, unary and binary
    operators. Calls to functions specified in 'safe_callables' dictionary are allowed.

    This function is designed to evaluate expressions in a controlled environment,
    preventing the execution of arbitrary code. It parses the input into an
    Abstract Syntax Tree (AST) and recursively evaluates each node, only allowing
    operations and function calls that are explicitly permitted.

    Args:
        node_or_string (Union[ast.AST, str]): The expression node or string to evaluate.
        safe_callables (Dict[str, Callable]): A dictionary mapping function names to
            callable Python objects. Only these functions can be called within the
            expression.
        safe_methods (Dict[Tuple[type, str], Callable]): A dictionary mapping
            (type, method_name) to callable methods. Only these methods can be called
            on objects within the expression.
        _env (Dict): Internal parameter for variable and function environment.
            Should not be set manually.

    Returns:
        Any: The result of the evaluated expression.

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
        # Parse the string expression into an AST node
        node = ast.parse(node_or_string, mode="eval").body
    else:
        node = node_or_string

    # Copy safe_callables into _env to allow names to evaluate to callables,
    # such as modules specified in safe_callables with their attributes/constants.
    _env = {**safe_callables, **_env}

    # Mapping of AST operator nodes to corresponding Python operations
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
            # Return the value of constants (e.g., numbers, strings)
            return node.value
        elif isinstance(node, ast.Name):
            # Handle variable names by looking them up in the environment
            if node.id in _env:
                return _env[node.id]
            else:
                raise ValueError(f"Name '{node.id}' is not defined.")
        elif isinstance(node, ast.Attribute):
            # Handle attribute access (e.g., obj.attr)
            obj = _convert(node.value)
            if isinstance(obj, ModuleType) and hasattr(obj, node.attr):
                attribute = getattr(obj, node.attr)
                return attribute if callable(attribute) else attribute
            else:
                raise ValueError(
                    f"Attribute '{node.attr}' not found on object '{obj}'."
                )
        elif isinstance(node, (ast.Tuple, ast.List)):
            # Handle tuple and list literals
            elements = []
            for elt in node.elts:
                if isinstance(elt, ast.Starred):
                    # Handle starred expressions (e.g., *args)
                    elements.extend(_convert(elt.value))
                else:
                    elements.append(_convert(elt))
            # Return the appropriate type (tuple or list)
            return tuple(elements) if isinstance(node, ast.Tuple) else elements
        elif isinstance(node, ast.Dict):
            # Handle dictionary literals
            return {_convert(k): _convert(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.UnaryOp):
            # Handle unary operations (e.g., +x, -x)
            operand = _convert(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
            else:
                raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
        elif isinstance(node, ast.BinOp) and type(node.op) in op_map:
            # Handle binary operations (e.g., x + y)
            left = _convert(node.left)
            right = _convert(node.right)
            return op_map[type(node.op)](left, right)
        elif isinstance(node, ast.BoolOp) and type(node.op) in op_map:
            # Handle boolean operations (e.g., x and y)
            values = [_convert(v) for v in node.values]
            result = values[0]
            for value in values[1:]:
                result = op_map[type(node.op)](result, value)
            return result
        elif isinstance(node, ast.Compare) and type(node.ops[0]) in op_map:
            # Handle comparison operations (e.g., x < y)
            left = _convert(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = _convert(comparator)
                if not op_map[type(op)](left, right):
                    return False
                left = right
            return True
        elif isinstance(node, ast.IfExp):
            # Handle ternary expressions (e.g., x if condition else y)
            return _convert(node.body) if _convert(node.test) else _convert(node.orelse)
        elif isinstance(node, ast.Call):
            # Handle function and method calls
            if isinstance(node.func, ast.Attribute):
                # Method call (e.g., obj.method())
                obj = _convert(node.func.value)
                method_name = node.func.attr
                if isinstance(obj, ModuleType):
                    method_key = (obj, method_name)
                    if method_key in safe_methods:
                        method = safe_methods[method_key]
                        return method(
                            *map(_convert, node.args),
                            **{kw.arg: _convert(kw.value) for kw in node.keywords},
                        )
                else:
                    method_key = (type(obj), method_name)
                    if method_key in safe_methods:
                        method = safe_methods[method_key]
                        # If the method is a callable, call it with the arguments and
                        # pass the object as the first argument
                        return method(
                            obj,
                            *map(_convert, node.args),
                            **{kw.arg: _convert(kw.value) for kw in node.keywords},
                        )
                raise ValueError(
                    f"Method '{method_name}' not "
                    f"allowed on type '{type(obj).__name__}'."
                )
            elif isinstance(node.func, ast.Name):
                # Function call (e.g., func())
                func_name = node.func.id
                if func_name in safe_callables:
                    func = safe_callables[func_name]
                    if callable(func):
                        return func(
                            *map(_convert, node.args),
                            **{kw.arg: _convert(kw.value) for kw in node.keywords},
                        )
                    else:
                        raise ValueError(f"Name '{func_name}' is not callable.")
                else:
                    raise ValueError(f"Function '{func_name}' is not allowed.")
            else:
                raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")
        elif isinstance(node, ast.GeneratorExp):
            # Handle generator expressions (e.g., (x for x in iterable))
            iter_node = node.elt
            results = []

            for comprehension in node.generators:
                iter_list = _convert(comprehension.iter)
                for item in iter_list:
                    # Assign the item to the target variable in the environment
                    _env[comprehension.target.id] = item
                    # Check if all conditions in the comprehension are True
                    if all(_convert(cond) for cond in comprehension.ifs):
                        results.append(_convert(iter_node))

            return results
        elif isinstance(node, ast.ListComp):
            # Handle list comprehensions by converting them to generator expressions
            return _convert(ast.GeneratorExp(node.elt, node.generators))
        elif isinstance(node, ast.Subscript):
            # Handle subscription (e.g., obj[index])
            obj = _convert(node.value)
            index = _convert(node.slice)
            return obj[index]
        elif isinstance(node, ast.Slice):
            # Handle slicing (e.g., start:stop:step)
            start = _convert(node.lower) if node.lower else None
            stop = _convert(node.upper) if node.upper else None
            step = _convert(node.step) if node.step else None
            return slice(start, stop, step)
        elif isinstance(node, ast.Starred):
            # Handle starred expressions in function calls or assignments
            return _convert(node.value)
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")

        # If none of the above cases matched, raise an error
        raise ValueError(f"Couldn't parse node ({type(node)}): {ast.dump(node)}")

    return _convert(node)


def safe_eval(src: Any, additional_vars: Dict[str, Any] = {}) -> Any:
    """
    Evaluate a string containing a Python expression in a safe manner.

    This function uses `literal_eval_with_callables` to evaluate the expression,
    only allowing certain built-in functions and types, and any additional variables
    provided. It prevents execution of arbitrary code or access to unauthorized
    functions and methods.

    Args:
        src (Any): The source code (string or AST node) to evaluate.
        additional_vars (Dict[str, Any]): A dictionary of additional variables or
            functions to include in the evaluation environment.

    Returns:
        Any: The result of the evaluated expression.

    Raises:
        ValueError: If the expression contains unsupported operations or cannot be
            evaluated.

    Examples:
        >>> safe_eval("1 + 2")
        3
        >>> safe_eval("max([1, 2, 3])")
        3
        >>> safe_eval("math.sqrt(16)", {'math': math})
        4.0
    """
    import math

    # Define supported built-in functions and types
    supported_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "tuple": tuple,
        "set": set,
        "dict": dict,
        "list": list,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }
    # Create the safe environment for evaluation
    safe_vars = {"math": math, **supported_builtins, **additional_vars}

    # Define method names that are considered unsafe
    unsafe_method_names = {
        "format",
        "format_map",
        "eval",
        "exec",
        "compile",
        "open",
        "read",
        "write",
        "input",
    }
    # Build the dictionary of safe methods for supported built-in types
    safe_methods = {
        (safe_var, method_name): method
        for safe_var in safe_vars.values()
        if hasattr(safe_var, "__dict__")
        for method_name, method in safe_var.__dict__.items()
        if callable(method)
        and not method_name.startswith("_")
        and method_name not in unsafe_method_names
    }
    try:
        # Evaluate the expression using the safe environment
        return literal_eval_with_callables(src, safe_vars, safe_methods)
    except ValueError as e:
        # Raise a new ValueError with additional context
        raise ValueError(f"Error evaluating expression '{src}': {e}") from e


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
