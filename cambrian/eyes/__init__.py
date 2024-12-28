"""This module defines the Cambrian eyes."""

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.eyes.multi_eye import MjCambrianMultiEye, MjCambrianMultiEyeConfig
from cambrian.eyes.multi_eye_approx import (
    MjCambrianMultiEyeApprox,
    MjCambrianMultiEyeApproxConfig,
)
from cambrian.eyes.optics import MjCambrianOpticsEye, MjCambrianOpticsEyeConfig

__all__ = [
    "MjCambrianEyeConfig",
    "MjCambrianEye",
    "MjCambrianMultiEyeConfig",
    "MjCambrianMultiEye",
    "MjCambrianMultiEyeApproxConfig",
    "MjCambrianMultiEyeApprox",
    "MjCambrianOpticsEyeConfig",
    "MjCambrianOpticsEye",
]
