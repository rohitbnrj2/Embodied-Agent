from typing import Any, List
from pathlib import Path
from dataclasses import dataclass
import numpy as np

import mujoco as mj

def safe_index(l: List[Any], v: Any) -> int:
    """Safely get the index of a value in a list, or -1 if not found. Normally, 
    list.index() throws an exception if the value is not found.""" 
    try:
        return l.index(v)
    except ValueError:
        return -1

def get_model_path(model_path: str | Path, *, throw_error: bool = True) -> Path | None:
    """Tries to find the model path. `model_path` can either be relative to the 
    execution file, absolute, or relative to cambrian.evolution_envs.three_d.mujoco. The
    latter is the typical method, where `assets/<model>.xml` specifies the model path 
    located in cambrian/evolution_envs/three_d/mujoco/assets/<model>.xml.

    If the file can't be found, a FileNotFoundError is raised if throw_error is True. If
    throw_error is False, None is returned.
    """
    model_path = Path(model_path)
    if model_path.exists():
        pass
    elif (rel_model_path := Path(__file__).parent / model_path).exists():
        model_path = rel_model_path
    else:
        if throw_error:
            raise FileNotFoundError(f"Could not find model file {model_path}.")
        else:
            return None

    return model_path

# =============
# Mujoco utils

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
    def create(model: mj.MjModel, jntadr: int) -> 'MjCambrianJoint':
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
        else: # mj.mjtJoint.mjJNT_HINGE or mj.mjtJoint.mjJNT_SLIDE
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