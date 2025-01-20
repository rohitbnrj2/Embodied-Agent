"""Defines utilities for overlays in the Mujoco viewer."""

from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

import mujoco as mj
import numpy as np
import torch

from cambrian.utils.logger import get_logger

TEXT_HEIGHT = 20
TEXT_MARGIN = 5


@dataclass
class MjCambrianCursor:
    """This class is used to define a cursor for the overlay."""

    x: int
    y: int

    def __iter__(self):
        return iter((self.x, self.y))

    def copy(self):
        return replace(self)


class MjCambrianViewerOverlay:
    """This class is used to add an overlay to the viewer.

    Note:
        This is applied only to the passed scene, so other scenes (i.e. ones for the
        eyes) will not be affected.
    """

    def __init__(self, obj: Any, cursor: Optional[MjCambrianCursor] = None):
        self._obj = obj
        self._cursor = cursor.copy() if cursor is not None else None

    def draw_before_render(self, scene: mj.MjvScene):
        """Called before rendering the scene."""
        pass

    def draw_after_render(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
        """Called after rendering the scene."""
        pass


class MjCambrianTextViewerOverlay(MjCambrianViewerOverlay):
    """This class is used to add text to the viewer."""

    def draw_after_render(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
        viewport = viewport if self._cursor is None else mj.MjrRect(*self._cursor, 1, 1)
        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            self._obj,
            "",
            mjr_context,
        )


class MjCambrianImageViewerOverlay(MjCambrianViewerOverlay):
    """This class is used to add an image to the viewer."""

    def __init__(self, obj: torch.Tensor, cursor: Optional[MjCambrianCursor] = None):
        super().__init__(obj, cursor)
        self._obj_cpu = obj.cpu()

    def draw_after_render(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
        self._obj_cpu.copy_(self._obj, non_blocking=True)
        viewport = mj.MjrRect(*self._cursor, self._obj.shape[1], self._obj.shape[0])
        mj.mjr_drawPixels(self._obj_cpu.numpy().ravel(), None, viewport, mjr_context)


class MjCambrianSiteViewerOverlay(MjCambrianViewerOverlay):
    """This class is used to add a site to the viewer.

    Todo:
        Make this an image overlay where the pos is converted to pixel coordinates.
    """

    def __init__(
        self,
        pos: np.ndarray,
        rgba: Tuple[float, float, float, float],
        size: float,
        geom_kwargs: dict = dict(emission=0.25),
    ):
        super().__init__(pos)
        self._rgba = rgba
        self._size = size
        self._geom_kwargs = geom_kwargs

    def draw_before_render(self, scene: mj.MjvScene):
        if scene.ngeom >= scene.maxgeom:
            get_logger().warning(
                f"Max geom reached ({scene.maxgeom}). Cannot add more sites."
            )
            return

        scene.ngeom += 1
        mj.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mj.mjtGeom.mjGEOM_SPHERE,
            [self._size] * 3,
            self._obj,
            np.eye(3).flatten(),
            self._rgba,
        )
        for key, value in self._geom_kwargs.items():
            setattr(scene.geoms[scene.ngeom - 1], key, value)
