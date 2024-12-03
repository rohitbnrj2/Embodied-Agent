"""This module defines the Cambrian eyes."""

from cambrian.eyes.approx_multi_eye import (
    MjCambrianApproxMultiEye,
    MjCambrianApproxMultiEyeConfig,
)
from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.eyes.multi_eye import MjCambrianMultiEye, MjCambrianMultiEyeConfig
from cambrian.eyes.optics import MjCambrianOpticsEye, MjCambrianOpticsEyeConfig

__all__ = [
    "MjCambrianEyeConfig",
    "MjCambrianEye",
    "MjCambrianMultiEyeConfig",
    "MjCambrianMultiEye",
    "MjCambrianApproxMultiEyeConfig",
    "MjCambrianApproxMultiEye",
    "MjCambrianOpticsEyeConfig",
    "MjCambrianOpticsEye",
]
