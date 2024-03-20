from typing import Optional, Tuple
from dataclasses import dataclass, replace

import numpy as np
import mujoco as mj

from cambrian.utils.logging import get_logger

TEXT_HEIGHT = 20
TEXT_MARGIN = 5


@dataclass
class MjCambrianCursor:
    x: int
    y: int

    def __iter__(self):
        return iter((self.x, self.y))

    def copy(self):
        return replace(self)


class MjCambrianViewerOverlay:
    def __init__(
        self, obj: np.ndarray | str, cursor: Optional[MjCambrianCursor] = None
    ):
        self.obj = obj
        self.cursor = cursor.copy() if cursor is not None else None

    def draw_before_render(self, scene: mj.MjvScene):
        """Called before rendering the scene."""
        pass

    def draw_after_render(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
        """Called after rendering the scene."""
        pass


class MjCambrianTextViewerOverlay(MjCambrianViewerOverlay):
    def draw_after_render(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
        viewport = viewport if self.cursor is None else mj.MjrRect(*self.cursor, 1, 1)
        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            self.obj,
            "",
            mjr_context,
        )


class MjCambrianImageViewerOverlay(MjCambrianViewerOverlay):
    def draw_after_render(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
        viewport = mj.MjrRect(*self.cursor, self.obj.shape[1], self.obj.shape[0])
        mj.mjr_drawPixels(self.obj.ravel(), None, viewport, mjr_context)


class MjCambrianSiteViewerOverlay(MjCambrianViewerOverlay):
    """TODO: make this an image overlay where the pos is converted to pixel
    coordinates.

    NOTE: This is applied only to the passed scene, so other scenes (i.e. ones for the
    eyes) will not be affected.
    """

    def __init__(
        self, pos: np.ndarray, rgba: Tuple[float, float, float, float], size: float
    ):
        super().__init__(pos)
        self.rgba = rgba
        self.size = size

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
            [self.size] * 3,
            self.obj,
            np.eye(3).flatten(),
            self.rgba,
        )
