from dataclasses import dataclass
from typing import Any, List

import mujoco as mj

def safe_index(l: List[Any], v: Any) -> int:
    """Safely get the index of a value in a list, or -1 if not found. Normally, 
    list.index() throws an exception if the value is not found.""" 
    try:
        return l.index(v)
    except ValueError:
        return -1

# =============
# Mujoco utils

@dataclass
class MjCambrianJoint:
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