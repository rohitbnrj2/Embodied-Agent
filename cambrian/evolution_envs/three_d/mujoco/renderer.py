from typing import List, Optional, Any
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, replace

import glfw
import numpy as np
import mujoco as mj
import OpenGL.GL as GL
import imageio
import cv2

from config import MjCambrianRendererConfig
from utils import get_camera_id, get_body_id

TEXT_HEIGHT = 20
TEXT_MARGIN = 5


def resize_with_aspect_fill(image: np.ndarray, width: int, height: int):
    original_height, original_width = image.shape[:2]
    ratio_original = original_width / original_height
    ratio_new = width / height

    # Resize the image while maintaining the aspect ratio
    border_type = cv2.BORDER_CONSTANT
    if ratio_original > ratio_new:
        # Original is wider relative to the new size
        resize_height = round(width / ratio_original)
        resized_image = cv2.resize(image, (width, resize_height))
        top = (height - resize_height) // 2
        bottom = height - resize_height - top
        result = cv2.copyMakeBorder(resized_image, top, bottom, 0, 0, border_type)
    else:
        # Original is taller relative to the new size
        resize_width = round(height * ratio_original)
        resized_image = cv2.resize(image, (resize_width, height))
        left = (width - resize_width) // 2
        right = width - resize_width - left
        result = cv2.copyMakeBorder(resized_image, 0, 0, left, right, border_type)

    return result


@dataclass
class MjCambrianCursor:
    x: int
    y: int

    def __iter__(self):
        return iter((self.x, self.y))

    def copy(self):
        return replace(self)


class MjCambrianViewerOverlay(ABC):
    def __init__(
        self, obj: np.ndarray | str, cursor: Optional[MjCambrianCursor] = None
    ):
        self.obj = obj
        self.cursor = cursor.copy() if cursor is not None else None

    @abstractmethod
    def draw(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
        pass


class MjCambrianTextViewerOverlay(MjCambrianViewerOverlay):
    def draw(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
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
    def draw(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
        viewport = mj.MjrRect(*self.cursor, self.obj.shape[1], self.obj.shape[0])
        mj.mjr_drawPixels(self.obj.ravel(), None, viewport, mjr_context)



class MjCambrianViewer(ABC):

    _gl_context: mj.gl_context.GLContext = None

    def __init__(self, config: MjCambrianRendererConfig):
        self.config = config

        self.model: mj.MjModel = None
        self.data: mj.MjData = None
        self.scene: mj.MjvScene = None
        self.scene_option = mj.MjvOption()
        self.camera: mj.MjvCamera = mj.MjvCamera()
        self.viewport: mj.MjrRect = None

        # self._gl_context: mj.gl_context.GLContext = None
        self._mjr_context: mj.MjrContext = None

    def reset(self, model: mj.MjModel, data: mj.MjData, width: int, height: int):
        self.model = model
        self.data = data

        self.reset_camera()

        self.scene = mj.MjvScene(model=model, maxgeom=self.config.max_geom)
        self.viewport = mj.MjrRect(0, 0, width, height)

        self.setup_contexts()


    @abstractmethod
    def setup_contexts(self):
        if MjCambrianViewer._gl_context is None:
            MjCambrianViewer._gl_context = mj.gl_context.GLContext(self.viewport.width, self.viewport.height)
        self._gl_context = MjCambrianViewer._gl_context
        self.make_context_current()

    def reset_camera(self):
        """Setup the camera."""
        camera_config = self.config.camera_config
        if camera_config is None:
            return

        assert not (
            camera_config.type and camera_config.type_str
        ), "Camera type and type_str are mutually exclusive."
        assert not (
            camera_config.fixedcamid and camera_config.fixedcamname
        ), "Camera fixedcamid and fixedcamname are mutually exclusive."
        assert not (
            camera_config.trackbodyid and camera_config.trackbodyname
        ), "Camera trackbodyid and trackbodyname are mutually exclusive."

        if camera_config.type is not None:
            self.camera.type = camera_config.type
        if camera_config.type_str is not None:
            type_str = f"mjCAMERA_{camera_config.type_str.upper()}"
            self.camera.type = getattr(mj.mjtCamera, type_str)
        if camera_config.fixedcamid is not None:
            self.camera.fixedcamid = camera_config.fixedcamid
        if camera_config.fixedcamname is not None:
            fixedcamname = camera_config.fixedcamname
            self.camera.fixedcamid = get_camera_id(self.model, fixedcamname)
        if camera_config.trackbodyid is not None:
            self.camera.trackbodyid = camera_config.trackbodyid
        if camera_config.trackbodyname is not None:
            trackbodyname = camera_config.trackbodyname
            self.camera.trackbodyid = get_body_id(self.model, trackbodyname)
        if camera_config.distance is not None:
            self.camera.distance = camera_config.distance
            if camera_config.distance_factor is not None:
                self.camera.distance *= camera_config.distance_factor
        if camera_config.azimuth is not None:
            self.camera.azimuth = camera_config.azimuth
        if camera_config.elevation is not None:
            self.camera.elevation = camera_config.elevation
        if camera_config.lookat is not None:
            self.camera.lookat[:] = camera_config.lookat

    @abstractmethod
    def update(self, width: int, height: int):
        # Subclass should override this method such that this is not possible
        assert width == self.viewport.width and height == self.viewport.height

        mj.mjv_updateScene(
            self.model,
            self.data,
            self.scene_option,
            None,  # mjvPerturb
            self.camera,
            mj.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

    def render(self, *, overlays: List[MjCambrianViewerOverlay] = []):
        self.update(self.viewport.width, self.viewport.height)

        self.make_context_current()
        mj.mjr_render(self.viewport, self.scene, self._mjr_context)
        self.draw_overlays(overlays)

    def read_pixels(self) -> np.ndarray:
        width, height = self.viewport.width, self.viewport.height
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        mj.mjr_readPixels(pixels.ravel(), None, self.viewport, self._mjr_context)
        return np.flipud(pixels)

    def draw_overlays(self, overlays: List[MjCambrianViewerOverlay]):
        for overlay in overlays:
            overlay.draw(self._mjr_context, self.viewport)

    @abstractmethod
    def make_context_current(self):
        pass

    @abstractmethod
    def is_running(self):
        pass

    def close(self):
        self._gl_context.free()

    # ===================

    @property
    def width(self) -> int:
        return self.viewport.width

    @property
    def height(self) -> int:
        return self.viewport.height


class MjCambrianOffscreenViewer(MjCambrianViewer):
    _mjr_context: mj.MjrContext = None

    def reset(self, model: mj.MjModel, data: mj.MjData, width: int, height: int):
        super().reset(model, data, width, height)

        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr_context)

    def setup_contexts(self):
        super().setup_contexts()

        if MjCambrianOffscreenViewer._mjr_context is None:
            MjCambrianOffscreenViewer._mjr_context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_50)
        self._mjr_context = MjCambrianOffscreenViewer._mjr_context

    def update(self, width: int, height: int):
        if self.viewport.width != width or self.viewport.height != height:
            self.make_context_current()
            self.viewport = mj.MjrRect(0, 0, width, height)
            mj.mjr_resizeOffscreen(width, height, self._mjr_context)

        super().update(width, height)

    def make_context_current(self):
        self._gl_context.make_current()

    def is_running(self):
        return True


class MjCambrianOnscreenViewer(MjCambrianViewer):
    _mjr_context: mj.MjrContext = None

    def reset(self, model: mj.MjModel, data: mj.MjData, width: int, height: int):
        self.config.setdefault("resizeable", False)
        self.config.setdefault("fullscreen", False)

        self._last_mouse_x: int = 0
        self._last_mouse_y: int = 0
        self._is_paused: bool = False

        glfw.init()

        glfw.window_hint(glfw.RESIZABLE, int(self.config.resizeable))
        glfw.window_hint(glfw.VISIBLE, 1)
        self.window = glfw.create_window(width, height, "MjCambrian", None, None)
        self.default_window_pos = glfw.get_window_pos(self.window)
        self.fullscreen(self.config.fullscreen)

        super().reset(model, data, width, height)

        window_width, _ = glfw.get_window_size(self.window)
        self._scale = width / window_width

        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, self._mjr_context)
        glfw.swap_interval(1)

    def setup_contexts(self):
        super().setup_contexts()

        if MjCambrianOnscreenViewer._mjr_context is None:
            MjCambrianOnscreenViewer._mjr_context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_50)
        self._mjr_context = MjCambrianOnscreenViewer._mjr_context

    def make_context_current(self):
        super().make_context_current()
        glfw.make_context_current(self.window)

    def update(self, width: int, height: int):
        if self.viewport.width != width or self.viewport.height != height:
            self.make_context_current()
            self.viweport = mj.MjrRect(0, 0, width, height)
            GL.glViewport(0, 0, width, height)

        super().update(width, height)

    def render(self, *, overlays: List[MjCambrianViewerOverlay] = []):
        if self.window is None:
            print("WARNING: Tried to render destroyed window.")
            return
        elif glfw.window_should_close(self.window):
            print("WARNING: Tried to render closed or closing window.")
            return

        width, height = glfw.get_framebuffer_size(self.window)
        self.viewport = mj.MjrRect(0, 0, width, height)

        super().render(overlays=overlays)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

        if self._is_paused:
            self.render(overlays=overlays)

    def is_running(self):
        return not (self.window is None or glfw.window_should_close(self.window))

    def close(self):
        if self.window is not None:
            if glfw.get_current_context() == self.window:
                glfw.make_context_current(None)
            glfw.set_window_should_close(self.window, True)
            glfw.destroy_window(self.window)
            self.window = None

            glfw.terminate()

        super().close()

    # ===================

    def fullscreen(self, fullscreen: bool):
        if self.window is None:
            print("WARNING: Tried to set fullscreen to destroyed window.")
            return

        if fullscreen:
            monitor = glfw.get_primary_monitor()
            video_mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self.window,
                monitor,
                0,
                0,
                video_mode.size.width,
                video_mode.size.height,
                video_mode.refresh_rate,
            )

    # ===================

    def _cursor_pos_callback(self, window, xpos, ypos):
        left_button_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)
        right_button_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT)
        if not (left_button_pressed or right_button_pressed):
            return

        shift_pressed = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if right_button_pressed:
            MOVE_H, MOVE_V = mj.mjtMouse.mjMOUSE_MOVE_H, mj.mjtMouse.mjMOUSE_MOVE_V
            action = MOVE_H if shift_pressed else MOVE_V
        elif left_button_pressed:
            ROT_H, ROT_V = mj.mjtMouse.mjMOUSE_ROTATE_H, mj.mjtMouse.mjMOUSE_ROTATE_V
            action = ROT_H if shift_pressed else ROT_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)
        reldx, reldy = dx / width, dy / height

        mj.mjv_moveCamera(self.model, action, reldx, reldy, self.scene, self.camera)

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, action, mods):
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

    def _scroll_callback(self, window, xoffset, yoffset):
        mj.mjv_moveCamera(
            self.model,
            mj.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.05 * yoffset,
            self.scene,
            self.camera,
        )

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return

        # Close window.
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

        # Switch cameras
        if key == glfw.KEY_TAB:
            self.camera.fixedcamid += 1
            self.camera.type = mj.mjtCamera.mjCAMERA_FIXED
            if self.camera.fixedcamid >= self.model.ncam:
                self.camera.fixedcamid = -1
                self.camera.type = mj.mjtCamera.mjCAMERA_FREE

        # Pause simulation
        if key == glfw.KEY_SPACE:
            self._is_paused = not self._is_paused


class MjCambrianRenderer:
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config: MjCambrianRendererConfig):
        self.config = config
        self.config.setdefault("max_geom", 1000)
        self.config.setdefault("use_shared_context", True)

        assert all(
            mode in self.metadata["render.modes"] for mode in self.render_modes
        ), f"Invalid render mode found. Valid modes are {self.metadata['render.modes']}"

        self.viewer: MjCambrianViewer = None
        if "human" in self.render_modes:
            self.viewer = MjCambrianOnscreenViewer(self.config)
        else:
            self.viewer = MjCambrianOffscreenViewer(self.config)

        self._image_buffer: List[np.ndarray] = []

        self._record: bool = False

    def reset(self, model: mj.MjModel, data: mj.MjData) -> np.ndarray | None:
        self.config.setdefault("width", model.vis.global_.offwidth)
        self.config.setdefault("height", model.vis.global_.offheight)

        if self.config.width > model.vis.global_.offwidth:
            model.vis.global_.offwidth = self.config.width
        if self.config.height > model.vis.global_.offheight:
            model.vis.global_.offheight = self.config.height

        self.viewer.reset(model, data, self.config.width, self.config.height)

        return self.render()

    def render(
        self, *, overlays: List[MjCambrianViewerOverlay] = []
    ) -> np.ndarray | None:
        self.viewer.render(overlays=overlays)

        if "rgb_array" in self.render_modes:
            pixels = self.viewer.read_pixels()
            if self._record:
                self._image_buffer.append(pixels)
            return pixels

    def is_running(self):
        return self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def __del__(self):
        self.close()

    # ===================

    def set_option(self, option: str, value: Any, index: Optional[slice | int]):
        assert hasattr(self.viewer.scene_option, option), f"Invalid option {option}."
        if index is not None:
            getattr(self.viewer.scene_option, option)[index] = value
        else:
            setattr(self.viewer.scene_option, option, value)

    # ===================

    def save(self, path: Path | str, *, save_types: List[str] = ["gif", "mp4"]):
        AVAILABLE_SAVE_TYPES = ["gif", "mp4", "png"]
        assert all(
            save_type in AVAILABLE_SAVE_TYPES for save_type in save_types
        ), f"Invalid save type found. Valid types are {AVAILABLE_SAVE_TYPES}."

        assert self._record, "Cannot save without recording."
        assert len(self._image_buffer) > 0, "Cannot save empty buffer."

        print(f"Saving visualizations at {path}...")

        path = Path(path)
        if "gif" in save_types:
            duration = 1000 / self.config.fps
            gif = path.with_suffix(".gif")
            imageio.mimwrite(gif, self._image_buffer, loop=0, duration=duration)
        if "mp4" in save_types:
            mp4 = path.with_suffix(".mp4")
            writer = imageio.get_writer(mp4, fps=self.config.fps)
            for image in self._image_buffer:
                writer.append_data(image)
            writer.close()
        if "png" in save_types:
            png = path.with_suffix(".png")
            imageio.imwrite(png, self._image_buffer[-1])

        print(f"Saved visualization at {path}")

    @property
    def record(self) -> bool:
        return self._record

    @record.setter
    def record(self, record: bool):
        assert not (record and self._record), "Already recording."
        assert "rgb_array" in self.render_modes, "Cannot record without rgb_array mode."

        if not record:
            self._image_buffer.clear()

        self._record = record

    @property
    def render_modes(self) -> List[str]:
        return self.config.render_modes

    # ===================

    @property
    def width(self) -> int:
        return self.viewer.width

    @property
    def height(self) -> int:
        return self.viewer.height


if __name__ == "__main__":
    import yaml
    from pathlib import Path
    from cambrian_xml import MjCambrianXML
    import cv2

    YAML = """
    render_modes: ['human']

    max_geom: 1000

    width: 640
    height: 480

    resizeable: true
    fullscreen: True

    use_shared_context: true
    """

    xml = MjCambrianXML(Path(__file__).parent / "models" / "test.xml")
    xml.add(
        xml.find(".//worldbody"),
        "camera",
        pos="0 0 0.2",
        quat="0.5 0.5 0.5 0.5",
        resolution="640 480",
    )

    model = mj.MjModel.from_xml_string(xml.to_string())
    data = mj.MjData(model)
    mj.mj_step(model, data)

    config = MjCambrianRendererConfig.from_dict(yaml.safe_load(YAML))
    renderer = MjCambrianRenderer(config)
    renderer.reset(model, data)

    while renderer.is_running():
        image = renderer.render()
        if image is None:
            continue

        cv2.imshow("image", image[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    renderer.close()
