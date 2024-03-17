from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

import glfw
import numpy as np
import mujoco as mj
import OpenGL.GL as GL
import cv2

from cambrian.renderer.overlays import MjCambrianViewerOverlay
from cambrian.utils.logger import get_logger
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig


@config_wrapper
class MjCambrianRendererConfig(MjCambrianBaseConfig):
    """The config for the renderer. Used for type hinting.

    A renderer corresponds to a single camera. The renderer can then view the scene in
    different ways, like offscreen (rgb_array) or onscreen (human).

    Attributes:
        render_modes (List[str]): The render modes to use for the renderer. See
            `MjCambrianRenderer.metadata["render.modes"]` for options.

        maxgeom (Optional[int]): The maximum number of geoms to render.

        width (int): The width of the rendered image. For onscreen renderers, if this
            is set, the window cannot be resized. Must be set for offscreen renderers.
        height (int): The height of the rendered image. For onscreen renderers, if this
            is set, the window cannot be resized. Must be set for offscreen renderers.

        fullscreen (Optional[bool]): Whether to render in fullscreen or not. If True,
            the width and height are ignored and the window is rendered in fullscreen.
            This is only valid for onscreen renderers.

        camera (Optional[MjCambrianCameraConfig]): The camera config to use for
            the renderer.
        scene_options (Optional[Dict[str, Any]]): The scene options to use for the
            renderer. Keys are the name of the option as defined in MjvOption. For
            array options (like `flags`), the value should be another dict where the
            keys are the indices/mujoco enum keys and the values are the values to set.

        use_shared_context (bool): Whether to use a shared context or not.
            If True, the renderer will share a context with other renderers. This is
            useful for rendering multiple renderers at the same time. If False, the
            renderer will create its own context. This is computationally expensive if
            there are many renderers.
    """

    render_modes: List[str]

    width: Optional[int] = None
    height: Optional[int] = None

    fullscreen: Optional[bool] = None

    camera: mj.MjvCamera
    scene: mj.MjvScene
    scene_options: mj.MjvOption

    use_shared_context: bool


GL_CONTEXT: mj.gl_context.GLContext = None
MJR_CONTEXT: mj.MjrContext = None


class MjCambrianViewer(ABC):
    def __init__(self, config: MjCambrianRendererConfig):
        self.config = config
        self.logger = get_logger()

        self.model: mj.MjModel = None
        self.data: mj.MjData = None
        self.viewport: mj.MjrRect = None
        self.scene: mj.MjvScene = None
        self.scene_options: mj.MjvOption = None
        self.camera: mj.MjvCamera = None

        self._gl_context: mj.gl_context.GLContext = None
        self._mjr_context: mj.MjrContext = None

    def reset(self, model: mj.MjModel, data: mj.MjData, width: int, height: int):
        self.model = model
        self.data = data

        self.scene = self.config.scene(model=model)
        self.scene_options = self.config.scene_options
        self.camera = self.config.camera

        # NOTE: All shared contexts must match either onscreen or offscreen. And their
        # height and width most likely must match as well. If the existing context
        # is onscreen and we're requesting offscreen, override use_shared_context (and
        # vice versa).
        global GL_CONTEXT, MJR_CONTEXT
        if self.config.use_shared_context:
            if (
                MJR_CONTEXT
                and MJR_CONTEXT.currentBuffer != self.get_framebuffer_option()
            ):
                self.logger.warning(
                    "Overriding use_shared_context. First buffer and current buffer don't match."
                )
                self.config.use_shared_context = False

        font_scale = mj.mjtFontScale.mjFONTSCALE_50
        if self.config.use_shared_context:
            if GL_CONTEXT is None:
                GL_CONTEXT = mj.gl_context.GLContext(width, height)
            self._gl_context = GL_CONTEXT
            self.make_context_current()
            if MJR_CONTEXT is None:
                MJR_CONTEXT = mj.MjrContext(self.model, font_scale)
            self._mjr_context = MJR_CONTEXT
        elif self.viewport is None or width != self.width or height != self.height:
            if self._gl_context is not None:
                del self._gl_context
            if self._mjr_context is not None:
                del self._mjr_context

            self._gl_context = mj.gl_context.GLContext(width, height)
            self.make_context_current()
            self._mjr_context = mj.MjrContext(self.model, font_scale)
        self._mjr_context.readDepthMap = mj.mjtDepthMap.mjDEPTH_ZEROFAR

        self.viewport = mj.MjrRect(0, 0, width, height)

        mj.mjr_setBuffer(self.get_framebuffer_option(), self._mjr_context)

    @abstractmethod
    def update(self, width: int, height: int):
        # Subclass should override this method such that this is not possible
        assert width == self.viewport.width and height == self.viewport.height

        mj.mjv_updateScene(
            self.model,
            self.data,
            self.scene_options,
            None,  # mjvPerturb
            self.camera,
            mj.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

    def render(self, *, overlays: List[MjCambrianViewerOverlay] = []):
        self.make_context_current()
        self.update(self.viewport.width, self.viewport.height)

        for overlay in overlays:
            overlay.draw_before_render(self.scene)

        mj.mjr_render(self.viewport, self.scene, self._mjr_context)

        for overlay in overlays:
            overlay.draw_after_render(self._mjr_context, self.viewport)

    def read_pixels(self, read_depth: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        width, height = self.viewport.width, self.viewport.height
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32) if read_depth else None

        mj.mjr_readPixels(rgb, depth, self.viewport, self._mjr_context)

        return np.flipud(rgb), np.flipud(depth) if read_depth else None

    @abstractmethod
    def make_context_current(self):
        pass

    @abstractmethod
    def get_framebuffer_option(self) -> int:
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
    def get_framebuffer_option(self) -> int:
        return mj.mjtFramebuffer.mjFB_OFFSCREEN.value

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
    def __init__(self, config: MjCambrianRendererConfig):
        super().__init__(config)

        self.window = None
        self.default_window_pos: Tuple[int, int] = None
        self._scale: float = None

        self._last_mouse_x: int = None
        self._last_mouse_y: int = None
        self._is_paused: bool = None

    def reset(self, model: mj.MjModel, data: mj.MjData, width: int, height: int):
        self._last_mouse_x: int = 0
        self._last_mouse_y: int = 0
        self._is_paused: bool = False

        if self.window is None:
            if not glfw.init():
                raise Exception("GLFW failed to initialize.")

            gl_context = None
            if self.config.use_shared_context:
                global GL_CONTEXT
                if GL_CONTEXT is None:
                    GL_CONTEXT = mj.gl_context.GLContext(width, height)
                gl_context = GL_CONTEXT._context
            self.window = glfw.create_window(
                width, height, "MjCambrian", None, gl_context
            )
            if not self.window:
                glfw.terminate()
                raise Exception("GLFW failed to create window.")

            glfw.show_window(self.window)

            self.default_window_pos = glfw.get_window_pos(self.window)
        glfw.set_window_size(self.window, width, height)
        self.fullscreen(self.config.fullscreen if self.config.fullscreen else False)

        super().reset(model, data, width, height)

        window_width, _ = glfw.get_window_size(self.window)
        self._scale = width / window_width

        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        glfw.swap_interval(1)

    def make_context_current(self):
        glfw.make_context_current(self.window)
        super().make_context_current()

    def get_framebuffer_option(self) -> int:
        return mj.mjtFramebuffer.mjFB_WINDOW.value

    def update(self, width: int, height: int):
        if self.viewport.width != width or self.viewport.height != height:
            self.make_context_current()
            self.viewport = mj.MjrRect(0, 0, width, height)
            GL.glViewport(0, 0, width, height)

        super().update(width, height)

    def render(self, *, overlays: List[MjCambrianViewerOverlay] = []):
        if self.window is None:
            self.logger.warning("Tried to render destroyed window.")
            return
        elif glfw.window_should_close(self.window):
            self.logger.warning("Tried to render closed or closing window.")
            return

        self.make_context_current()
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
            self.logger.warning("Tried to set fullscreen to destroyed window.")
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
            print(self.camera)

        # Pause simulation
        if key == glfw.KEY_SPACE:
            self._is_paused = not self._is_paused


class MjCambrianRenderer:
    metadata = {"render.modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, config: MjCambrianRendererConfig):
        self.config = config
        self.logger = get_logger()

        assert all(
            mode in self.metadata["render.modes"] for mode in self.render_modes
        ), f"Invalid render mode found. Valid modes are {self.metadata['render.modes']}"
        assert (
            "depth_array" not in self.render_modes or "rgb_array" in self.render_modes
        ), "Cannot render depth_array without rgb_array."

        self.viewer: MjCambrianViewer = None
        if "human" in self.render_modes:
            self.viewer = MjCambrianOnscreenViewer(self.config)
        else:
            self.viewer = MjCambrianOffscreenViewer(self.config)

        self._rgb_buffer: List[np.ndarray] = []

        self._record: bool = False

    def reset(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> np.ndarray | None:
        self.config.width = width or self.config.width or model.vis.global_.offwidth
        self.config.height = height or self.config.height or model.vis.global_.offheight

        if self.config.width > model.vis.global_.offwidth:
            model.vis.global_.offwidth = self.config.width
        if self.config.height > model.vis.global_.offheight:
            model.vis.global_.offheight = self.config.height

        self.viewer.reset(model, data, self.config.width, self.config.height)

        return self.render(resetting=True)

    def render(
        self, *, overlays: List[MjCambrianViewerOverlay] = [], resetting: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray] | None:
        self.viewer.render(overlays=overlays)

        if not any(mode in self.render_modes for mode in ["rgb_array", "depth_array"]):
            return

        rgb, depth = self.viewer.read_pixels("depth_array" in self.render_modes)
        if self._record and not resetting:
            self._rgb_buffer.append(rgb)

        return (rgb, depth) if "depth_array" in self.render_modes else rgb

    def is_running(self):
        return self.viewer.is_running()

    def close(self):
        return
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()

    def __del__(self):
        self.close()

    # ===================

    def save(self, path: Path | str, *, save_types: List[str] = ["webp"]):
        AVAILABLE_SAVE_TYPES = ["gif", "mp4", "png", "webp"]
        assert all(
            save_type in AVAILABLE_SAVE_TYPES for save_type in save_types
        ), f"Invalid save type found. Valid types are {AVAILABLE_SAVE_TYPES}."

        assert self._record, "Cannot save without recording."
        assert len(self._rgb_buffer) > 0, "Cannot save empty buffer."

        self.logger.info(f"Saving visualizations at {path}...")

        path = Path(path)
        rgb_buffer = np.array(self._rgb_buffer)
        if len(rgb_buffer) > 1:
            rgb_buffer = rgb_buffer[:-1]
        fps = 50
        if "mp4" in save_types:
            import imageio

            mp4 = path.with_suffix(".mp4")
            writer = imageio.get_writer(mp4, fps=fps)
            for image in rgb_buffer:
                writer.append_data(image)
            writer.close()
        if "png" in save_types:
            import imageio

            png = path.with_suffix(".png")
            imageio.imwrite(png, rgb_buffer[-1])
        if "gif" in save_types:
            import imageio

            duration = 1000 / fps
            gif = path.with_suffix(".gif")
            imageio.mimwrite(gif, rgb_buffer, loop=0, duration=duration)
        if "webp" in save_types:
            import webp

            webp.mimwrite(path.with_suffix(".webp"), rgb_buffer, fps=fps, lossless=True)

        self.logger.debug(f"Saved visualization at {path}")

    @property
    def record(self) -> bool:
        return self._record

    @record.setter
    def record(self, record: bool):
        assert not (record and self._record), "Already recording."
        assert "rgb_array" in self.render_modes, "Cannot record without rgb_array mode."

        if not record:
            self._rgb_buffer.clear()

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

    @property
    def ratio(self) -> float:
        return self.width / self.height


if __name__ == "__main__":
    from cambrian.utils.cambrian_xml import MjCambrianXML
    from cambrian.utils.config import MjCambrianConfig, run_hydra

    def main(config: MjCambrianConfig):
        xml = MjCambrianXML.from_config(config.env.xml)
        model = mj.MjModel.from_xml_string(xml.to_string())
        data = mj.MjData(model)
        mj.mj_step(model, data)

        renderer = MjCambrianRenderer(config.env.renderer)
        renderer.reset(model, data)

        while renderer.is_running():
            renderer.render()

    # Recommended to use these args:
    # env.xml.base_xml_path=models/test.xml env/renderer=fixed
    run_hydra(main)
