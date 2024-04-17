from typing import List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Flag, auto
from copy import deepcopy

import glfw
import numpy as np
import mujoco as mj
import OpenGL.GL as GL

from cambrian.renderer.overlays import MjCambrianViewerOverlay
from cambrian.utils.logger import get_logger
from cambrian.utils.config import (
    MjCambrianBaseConfig,
    config_wrapper,
    MjCambrianFlagWrapperMeta,
)


class MjCambrianRendererSaveMode(Flag, metaclass=MjCambrianFlagWrapperMeta):
    GIF = auto()
    MP4 = auto()
    PNG = auto()
    WEBP = auto()


@config_wrapper
class MjCambrianRendererConfig(MjCambrianBaseConfig):
    """The config for the renderer. Used for type hinting.

    A renderer corresponds to a single camera. The renderer can then view the scene in
    different ways, like offscreen (rgb_array) or onscreen (human).

    Attributes:
        render_modes (List[str]): The render modes to use for the renderer. See
            `MjCambrianRenderer.metadata["render.modes"]` for options.

        width (Optional[int]): The width of the rendered image. For onscreen renderers,
            if this is set, the window cannot be resized. Must be set for offscreen
            renderers.
        height (Optional[int]): The height of the rendered image. For onscreen
            renderers, if this is set, the window cannot be resized. Must be set for
            offscreen renderers.

        fullscreen (Optional[bool]): Whether to render in fullscreen or not. If True,
            the width and height are ignored and the window is rendered in fullscreen.
            This is only valid for onscreen renderers.

        scene (mj.MjvScene): The scene to render.
        scene_options (mj.MjvOption): The options to use for rendering.
        camera (mj.MjvCamera): The camera to use for rendering.

        use_shared_context (bool): Whether to use a shared context or not.
            If True, the renderer will share a context with other renderers. This is
            useful for rendering multiple renderers at the same time. If False, the
            renderer will create its own context. This is computationally expensive if
            there are many renderers.

        save_mode (Optional[MjCambrianRendererSaveMode]): The save modes to use for
            saving the rendered images. See `MjCambrianRenderer.SaveMode` for options.
            Must be set if `save` is called without save modes passed directly.
    """

    render_modes: List[str]

    width: Optional[int] = None
    height: Optional[int] = None

    fullscreen: Optional[bool] = None

    scene: mj.MjvScene
    scene_options: mj.MjvOption
    camera: mj.MjvCamera

    use_shared_context: bool

    save_mode: Optional[MjCambrianRendererSaveMode] = None


# TODO: If these are in the global scope, they don't throw an error when the script ends
GL_CONTEXT: mj.gl_context.GLContext = None
MJR_CONTEXT: mj.MjrContext = None


class MjCambrianViewer(ABC):
    def __init__(self, config: MjCambrianRendererConfig):
        self._config = config
        self._logger = get_logger()

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._viewport: mj.MjrRect = None
        self._scene: mj.MjvScene = None
        self._scene_options: mj.MjvOption = None
        self._camera: mj.MjvCamera = None

        self._gl_context: mj.gl_context.GLContext = None
        self._mjr_context: mj.MjrContext = None
        self._font = mj.mjtFontScale.mjFONTSCALE_50

        self._rgb_uint8: np.ndarray = np.array([])
        self._rgb_float32: np.ndarray = np.array([])
        self._depth: np.ndarray = np.array([])

    def reset(self, model: mj.MjModel, data: mj.MjData, width: int, height: int):
        self._model = model
        self._data = data

        # Only create the scene once
        if self._scene is None:
            self._scene = self._config.scene(model=model)
        self._scene_options = deepcopy(self._config.scene_options)
        self._camera = deepcopy(self._config.camera)

        self._initialize_contexts(width, height)

        self._viewport = mj.MjrRect(0, 0, width, height)

        # Initialize the buffers
        if self._rgb_uint8.shape[0] != height or self._rgb_uint8.shape[1] != width:
            self._rgb_uint8 = np.empty((height, width, 3), dtype=np.uint8)
            self._rgb_float32 = np.empty((height, width, 3), dtype=np.float32)
            self._depth = np.empty((height, width), dtype=np.float32)

    def _initialize_contexts(self, width: int, height: int):
        global GL_CONTEXT, MJR_CONTEXT

        # NOTE: All shared contexts must match either onscreen or offscreen. And their
        # height and width most likely must match as well. If the existing context
        # is onscreen and we're requesting offscreen, override use_shared_context (and
        # vice versa).
        use_shared_context = self._config.use_shared_context
        if use_shared_context and MJR_CONTEXT:
            if MJR_CONTEXT.currentBuffer != self.get_framebuffer_option():
                self._logger.warning(
                    "Overriding use_shared_context. "
                    "First buffer and current buffer don't match."
                )
                use_shared_context = False

        if use_shared_context:
            # Initialize or reuse the GL context
            GL_CONTEXT = GL_CONTEXT or mj.gl_context.GLContext(width, height)
            self._gl_context = GL_CONTEXT
            self.make_context_current()

            MJR_CONTEXT = MJR_CONTEXT or mj.MjrContext(self._model, self._font)
            self._mjr_context = MJR_CONTEXT
        elif self._viewport is None or width != self.width or height != self.height:
            # If the viewport is None (i.e. this is the first reset), or the window
            # has been resized, create a new context. We'll need to clean up the old
            # context if it exists.
            if self._gl_context is not None:
                del self._gl_context
            if self._mjr_context is not None:
                del self._mjr_context

            # Initialize the new contexts
            self._gl_context = mj.gl_context.GLContext(width, height)
            self.make_context_current()
            self._mjr_context = mj.MjrContext(self._model, self._font)
        self._mjr_context.readDepthMap = mj.mjtDepthMap.mjDEPTH_ZEROFAR
        mj.mjr_setBuffer(self.get_framebuffer_option(), self._mjr_context)

    @abstractmethod
    def update(self, width: int, height: int):
        # Subclass should override this method such that this is not possible
        assert width == self._viewport.width and height == self._viewport.height

        mj.mjv_updateScene(
            self._model,
            self._data,
            self._scene_options,
            None,  # mjvPerturb
            self._camera,
            mj.mjtCatBit.mjCAT_ALL,
            self._scene,
        )

    def render(self, *, overlays: List[MjCambrianViewerOverlay] = []):
        self.make_context_current()
        self.update(self._viewport.width, self._viewport.height)

        for overlay in overlays:
            overlay.draw_before_render(self._scene)

        mj.mjr_render(self._viewport, self._scene, self._mjr_context)

        for overlay in overlays:
            overlay.draw_after_render(self._mjr_context, self._viewport)

    def read_pixels(self, read_depth: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        rgb_uint8, depth = self._rgb_uint8, self._depth if read_depth else None
        mj.mjr_readPixels(rgb_uint8, depth, self._viewport, self._mjr_context)

        # Flipud the images
        # NOTE: If you plan to convert to a pytorch tensor, negative indices aren't
        # supported, so you may need to copy the array later on.
        rgb_uint8, depth = rgb_uint8[::-1, ...], (
            depth[::-1, ...] if read_depth else None
        )

        # Convert to float32
        rgb_float32 = self._rgb_float32
        np.divide(rgb_uint8, np.array([255.0], np.float32), out=rgb_float32)

        # Transpose the rgb/depth to be W x H x C
        rgb_float32 = rgb_float32.transpose(1, 0, 2)
        if read_depth:
            depth = depth.transpose(1, 0)

        # Return the flipped images
        return rgb_float32, depth

    @abstractmethod
    def make_context_current(self):
        pass

    @abstractmethod
    def get_framebuffer_option(self) -> int:
        pass

    @abstractmethod
    def is_running(self):
        pass

    # ===================

    @property
    def width(self) -> int:
        return self._viewport.width

    @width.setter
    def width(self, width: int):
        self._viewport.width = width

    @property
    def height(self) -> int:
        return self._viewport.height

    @height.setter
    def height(self, height: int):
        self._viewport.height = height

    @property
    def camera(self) -> mj.MjvCamera:
        return self._camera

    @property
    def config(self) -> MjCambrianRendererConfig:
        return self._config


class MjCambrianOffscreenViewer(MjCambrianViewer):
    def get_framebuffer_option(self) -> int:
        return mj.mjtFramebuffer.mjFB_OFFSCREEN.value

    def update(self, width: int, height: int):
        if self._viewport.width != width or self._viewport.height != height:
            self.make_context_current()
            self._viewport = mj.MjrRect(0, 0, width, height)
            mj.mjr_resizeOffscreen(width, height, self._mjr_context)

        super().update(width, height)

    def make_context_current(self):
        self._gl_context.make_current()

    def is_running(self):
        return True


class MjCambrianOnscreenViewer(MjCambrianViewer):
    def __init__(self, config: MjCambrianRendererConfig):
        super().__init__(config)

        self._window = None
        self.default_window_pos: Tuple[int, int] = None
        self._scale: float = None

        self._last_mouse_x: int = None
        self._last_mouse_y: int = None
        self._is_paused: bool = None
        self.custom_key_callback: Callable = None

    def reset(self, model: mj.MjModel, data: mj.MjData, width: int, height: int):
        self._last_mouse_x: int = 0
        self._last_mouse_y: int = 0
        self._is_paused: bool = False

        if self._window is None:
            self._initialize_window(width, height)
        glfw.set_window_size(self._window, width, height)
        self.fullscreen(self._config.fullscreen if self._config.fullscreen else False)

        super().reset(model, data, width, height)

        window_width, _ = glfw.get_window_size(self._window)
        self._scale = width / window_width

        glfw.set_cursor_pos_callback(self._window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self._window, self._mouse_button_callback)
        glfw.set_scroll_callback(self._window, self._scroll_callback)
        glfw.set_key_callback(self._window, self._key_callback)

        glfw.swap_interval(1)

    def _initialize_window(self, width: int, height: int):
        global GL_CONTEXT, MJR_CONTEXT

        if not glfw.init():
            raise Exception("GLFW failed to initialize.")

        gl_context = None
        if self._config.use_shared_context:
            from mujoco.glfw import GLContext as GLFWGLContext

            GL_CONTEXT = GL_CONTEXT or GLFWGLContext(width, height)
            assert isinstance(GL_CONTEXT, GLFWGLContext), (
                f"The mujoco gl context must be of type {GLFWGLContext} to use "
                f"the OnscreenViewer, but got {type(GL_CONTEXT)} instead. "
                "Set the env variable `MUJOCO_GL` to `glfw` to use the correct context."
            )
            gl_context = GL_CONTEXT._context
        self._window = glfw.create_window(width, height, "MjCambrian", None, gl_context)
        if not self._window:
            glfw.terminate()
            raise Exception("GLFW failed to create window.")

        glfw.show_window(self._window)

        self.default_window_pos = glfw.get_window_pos(self._window)

    def make_context_current(self):
        glfw.make_context_current(self._window)
        super().make_context_current()

    def get_framebuffer_option(self) -> int:
        return mj.mjtFramebuffer.mjFB_WINDOW.value

    def update(self, width: int, height: int):
        if self._viewport.width != width or self._viewport.height != height:
            self.make_context_current()
            self._viewport = mj.MjrRect(0, 0, width, height)
            GL.glViewport(0, 0, width, height)

        super().update(width, height)

    def render(self, *, overlays: List[MjCambrianViewerOverlay] = []):
        if self._window is None:
            self._logger.warning("Tried to render destroyed window.")
            return
        elif glfw.window_should_close(self._window):
            self._logger.warning("Tried to render closed or closing window.")
            return

        self.make_context_current()
        width, height = glfw.get_framebuffer_size(self._window)
        self._viewport = mj.MjrRect(0, 0, width, height)

        super().render(overlays=overlays)

        glfw.swap_buffers(self._window)
        glfw.poll_events()

        if self._is_paused:
            self.render(overlays=overlays)

    def is_running(self):
        return not (self._window is None or glfw.window_should_close(self._window))

    # ===================

    def fullscreen(self, fullscreen: bool):
        if self._window is None:
            self._logger.warning("Tried to set fullscreen to destroyed window.")
            return

        if fullscreen:
            monitor = glfw.get_primary_monitor()
            video_mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self._window,
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

        mj.mjv_moveCamera(self._model, action, reldx, reldy, self._scene, self._camera)

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, action, mods):
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

    def _scroll_callback(self, window, xoffset, yoffset):
        mj.mjv_moveCamera(
            self._model,
            mj.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.05 * yoffset,
            self._scene,
            self._camera,
        )

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return

        # Close window.
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

        # Switch cameras
        if key == glfw.KEY_TAB:
            self._camera.fixedcamid += 1
            self._camera.type = mj.mjtCamera.mjCAMERA_FIXED
            if self._camera.fixedcamid >= self._model.ncam:
                self._camera.fixedcamid = -1
                self._camera.type = mj.mjtCamera.mjCAMERA_FREE

        # Pause simulation
        if key == glfw.KEY_SPACE:
            self._is_paused = not self._is_paused

        # Custom key callback
        if self.custom_key_callback is not None:
            self.custom_key_callback(window, key, scancode, action, mods)


class MjCambrianRenderer:
    metadata = {"render.modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, config: MjCambrianRendererConfig):
        self._config = config
        self._logger = get_logger()

        assert all(
            mode in self.metadata["render.modes"] for mode in self._config.render_modes
        ), f"Invalid render mode found. Valid modes are {self.metadata['render.modes']}"
        assert (
            "depth_array" not in self._config.render_modes
            or "rgb_array" in self._config.render_modes
        ), "Cannot render depth_array without rgb_array."

        self._viewer: MjCambrianViewer = None
        if "human" in self._config.render_modes:
            self._viewer = MjCambrianOnscreenViewer(self._config)
        else:
            self._viewer = MjCambrianOffscreenViewer(self._config)

        self._rgb_buffer: List[np.ndarray] = []

        self._record: bool = False

    def reset(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> np.ndarray | None:
        width = width or self._config.width or model.vis.global_.offwidth
        height = height or self._config.height or model.vis.global_.offheight

        if width > model.vis.global_.offwidth:
            model.vis.global_.offwidth = width
        if height > model.vis.global_.offheight:
            model.vis.global_.offheight = height

        self._viewer.reset(model, data, width, height)

        return self.render(resetting=True)

    def render(
        self, *, overlays: List[MjCambrianViewerOverlay] = [], resetting: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray] | None:
        self._viewer.render(overlays=overlays)

        if not any(
            mode in self._config.render_modes for mode in ["rgb_array", "depth_array"]
        ):
            return

        rgb, depth = self._viewer.read_pixels(
            "depth_array" in self._config.render_modes
        )
        if self._record and not resetting:
            self._rgb_buffer.append(rgb.copy().transpose(1, 0, 2))

        return (rgb, depth) if "depth_array" in self._config.render_modes else rgb

    def is_running(self):
        return self._viewer.is_running()

    # ===================

    def save(
        self,
        path: Path | str,
        *,
        save_mode: Optional[MjCambrianRendererSaveMode] = None,
        fps: int = 50,
    ):
        save_mode = save_mode or self._config.save_mode

        assert self._record, "Cannot save without recording."
        assert len(self._rgb_buffer) > 0, "Cannot save empty buffer."

        self._logger.info(f"Saving visualizations at {path}...")

        path = Path(path)
        rgb_buffer = (np.array(self._rgb_buffer) * 255.0).astype(np.uint8)

        if save_mode & MjCambrianRendererSaveMode.MP4:
            import imageio

            try:
                mp4 = path.with_suffix(".mp4")
                writer = imageio.get_writer(mp4, fps=fps)
                for image in rgb_buffer:
                    writer.append_data(image)
                writer.close()
            except TypeError:
                self._logger.warning(
                    "imageio is not compiled with ffmpeg. "
                    "You may need to install it with `pip install imageio[ffmpeg]`."
                )
        if save_mode & MjCambrianRendererSaveMode.PNG:
            import imageio

            png = path.with_suffix(".png")
            imageio.imwrite(png, rgb_buffer[-1])
        if save_mode & MjCambrianRendererSaveMode.GIF:
            import imageio

            duration = 1000 / fps
            gif = path.with_suffix(".gif")
            imageio.mimwrite(gif, rgb_buffer, loop=0, duration=duration)
        if save_mode & MjCambrianRendererSaveMode.WEBP:
            import webp

            webp.mimwrite(path.with_suffix(".webp"), rgb_buffer, fps=fps, lossless=True)

        self._logger.debug(f"Saved visualization at {path}")

    @property
    def record(self) -> bool:
        return self._record

    @record.setter
    def record(self, record: bool):
        assert not (record and self._record), "Already recording."
        assert (
            "rgb_array" in self._config.render_modes
        ), "Cannot record without rgb_array mode."

        if not record:
            self._rgb_buffer.clear()

        self._record = record

    # ===================

    @property
    def config(self) -> MjCambrianRendererConfig:
        return self._config

    @property
    def viewer(self) -> MjCambrianViewer:
        return self._viewer

    @property
    def width(self) -> int:
        return self._viewer.width

    @property
    def height(self) -> int:
        return self._viewer.height

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
