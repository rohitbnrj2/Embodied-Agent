from typing import Dict, Optional, Tuple, List, Any
import numpy as np
from pathlib import Path

import glfw
import mujoco as mj
import OpenGL.GL as GL
import cv2

from config import MjCambrianRendererConfig


def resize_with_aspect_fill(image, width, height):
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


class MjCambrianViewer:
    """This is the base viewer class. It is an abstract class.

    This is based on the `gymnasium.envs.mujoco.mujoco_renderer.BaseRender` class.

    Derived classes should be implemented that implement different render mechanisms,
    such as offscreen or onscreen viewing. These derived classes are assumed to include
    base functionality and class members, which are implemented here.

    Args:
        model (mj.MjModel): The MuJoCo model.
        data (mj.MjData): The MuJoCo data.
        config (MjCambrianRendererConfig): The renderer configuration.
        camera (mj.MjvCamera): The camera to use. This is shared between all viewers.
    """

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        config: MjCambrianRendererConfig,
        camera: mj.MjvCamera,
    ):
        self.config = config
        self.config.setdefault("max_geom", 1000)

        self.model: mj.MjModel = model
        self.data: mj.MjData = data
        self.scene = mj.MjvScene(model=model, maxgeom=self.config.max_geom)
        self.scene_option = mj.MjvOption()
        self.camera: mj.MjvCamera = camera
        self.viewport: mj.MjrRect = None

        if self.width is None:
            self.width = self.model.vis.global_.offwidth
        elif self.width > self.model.vis.global_.offwidth:
            self.model.vis.global_.offwidth = self.width

        if self.height is None:
            self.height = self.model.vis.global_.offheight
        elif self.height > self.model.vis.global_.offheight:
            self.model.vis.global_.offheight = self.height

        self.setup_camera()

        self._is_closed: bool = False
        self._image_overlays: List[Tuple[np.ndarray, Tuple[int, int]]] = []
        self._text_overlays: List[Tuple[str, Tuple[int, int]]] = []

    def setup_context(self):
        self.viewport = mj.MjrRect(0, 0, self.width, self.height)
        self._gl_context = mj.gl_context.GLContext(self.width, self.height)
        self.make_context_current()
        self._mjr_context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_50)

    def setup_camera(self):
        """Setup the camera."""
        if self.config.camera_config is None:
            return

        if self.config.camera_config.type is not None:
            self.camera.type = self.config.camera_config.type
        if self.config.camera_config.fixedcamid is not None:
            self.camera.fixedcamid = self.config.camera_config.fixedcamid
        if self.config.camera_config.trackbodyid is not None:
            self.camera.trackbodyid = self.config.camera_config.trackbodyid
        if self.config.camera_config.distance is not None:
            self.camera.distance = self.config.camera_config.distance
            if self.config.camera_config.distance_factor is not None:
                self.camera.distance *= self.config.camera_config.distance_factor
        if self.config.camera_config.azimuth is not None:
            self.camera.azimuth = self.config.camera_config.azimuth
        if self.config.camera_config.elevation is not None:
            self.camera.elevation = self.config.camera_config.elevation
        if self.config.camera_config.lookat is not None:
            self.camera.lookat[:] = self.config.camera_config.lookat

    def update(self, width: Optional[int] = None, height: Optional[int] = None):
        """Update the underlying scene and camera.

        Derived classes should implement additional functionality, such as resizing the
        window/buffer if the width/height has changed.
        """
        self.width = self.width if width is None else width
        self.height = self.height if height is None else height

        mj.mjv_updateScene(
            self.model,
            self.data,
            self.scene_option,
            None,  # mjvPerturb
            self.camera,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )

    def make_context_current(self):
        """Make the OpenGL context current."""
        self._gl_context.make_current()

    def render(self):
        if self._is_closed:
            print("WARNING: Attempting to render a closed viewer.")
            return

        self.make_context_current()
        self.update(self.width, self.height)

        mj.mjr_render(self.viewport, self.scene, self._mjr_context)
        self._draw_overlays()

    def add_overlay(
        self,
        overlay: np.ndarray | str,
        pos: Optional[Tuple[int, int] | np.ndarray] = None,
    ):
        """Alias to add_image_overlay and add_text_overlay."""
        if isinstance(overlay, np.ndarray):
            self.add_image_overlay(overlay, pos)
        elif isinstance(overlay, str):
            self.add_text_overlay(overlay, pos)
        else:
            raise ValueError(f"Invalid overlay type `{type(overlay)}`.")

    def add_image_overlay(self, overlay: np.ndarray, pos: Tuple[int, int] | np.ndarray):
        """This method implements drawing the overlay on top of the image. This is
        useful for drawing agent camera views, etc.

        NOTE: All overlays are removed after each render, so they must be re-added
        each time.

        Args:
            overlay (np.ndarray): The overlay. Will be placed at the given pos on
                top of the image.
            pos (Tuple[int, int] | np.ndarray): The position of the overlay. This is
                relative to the bottom-left corner of the image. The height and width
                is assumed from the image. Fmt: (left, top)
        """
        self._image_overlays.append((overlay, tuple(pos)))

    def add_text_overlay(
        self,
        overlay: Any,
        pos: Optional[Tuple[int, int] | np.ndarray] = None,
    ):
        """This method implements drawing the overlay on top of the image. This is
        for text specifically.

        NOTE: All overlays are removed after each render, so they must be re-added
        each time.

        Args:
            overlay (Any): The overlay text. Assumes str(overlay) is valid.
            pos (Optional[Tuple[int, int] | np.ndarray]): The position of the overlay.
                If None, the text is placed in the bottom left of the image. If not
                None, the text bottom left is placed at the given pos. Fmt: (left, top)
        """
        self._text_overlays.append((str(overlay), tuple(pos)))

    def _draw_overlays(self):
        """Draw the overlays on top of the image."""

        # Not sure why this is necessary
        GL.glDisable(GL.GL_DEPTH_TEST)

        for overlay, pos in self._image_overlays:
            viewport = mj.MjrRect(*pos, overlay.shape[1], overlay.shape[0])
            mj.mjr_drawPixels(overlay.ravel(), None, viewport, self._mjr_context)

        for overlay, pos in self._text_overlays:
            viewport = self.viewport if pos is None else mj.MjrRect(*pos, 1, 1)
            mj.mjr_overlay(
                mj.mjtFont.mjFONT_NORMAL,
                mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,
                overlay,
                "",
                self._mjr_context,
            )

        # Not sure why this is necessary
        GL.glEnable(GL.GL_DEPTH_TEST)

    def clear_overlays(self):
        """Clear all overlays. Called by the renderer owner."""
        self._image_overlays.clear()
        self._text_overlays.clear()

    def close(self):
        """Closes the viewer and frees the OpenGL context."""
        self._gl_context.free()
        self._is_closed = True

    # ====================

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def width(self) -> int:
        return self.config.width

    @width.setter
    def width(self, value: int):
        """Set the width of the viewer. This will also update the viewport. If the
        viewport is None, it will be set to the correct width when it's instantiated
        in the constructor."""
        self.config.width = value
        if self.viewport is not None:
            self.viewport.width = value

    @property
    def height(self) -> int:
        return self.config.height

    @height.setter
    def height(self, value: int):
        """Set the height of the viewer. This will also update the viewport. If the
        viewport is None, it will be set to the correct height when it's instantiated
        in the constructor."""
        self.config.height = value
        if self.viewport is not None:
            self.viewport.height = value


class MjCambrianOffscreenViewer(MjCambrianViewer):
    """This is an offscreen viewer. It renders the scene to an offscreen buffer and
    then reads the pixels from the buffer. It implements the viewing through glfw.

    This class is based on the `gymnasium.envs.mujoco.mujoco_renderer.OffscreenViewer`.
    """

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        config: MjCambrianRendererConfig,
        camera: mj.MjvCamera,
    ):
        assert (
            config.width is not None and config.height is not None
        ), "Width and height must be specified for offscreen rendering."

        super().__init__(model, data, config, camera)
        self.setup_context()

        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr_context)

    def update(self, width: Optional[int] = None, height: Optional[int] = None):
        """Update the underlying scene and camera.

        If the width and height are different from the current width and height, then
        the offscreen buffer is resized.

        NOTE: on low-end GPUs, resizing the buffer can be slow if there are many
        viewers running on the system. It may be performant (though memory inefficient
        from a RAM perspective) to set config.use_shared_context to False.
        """
        width = self.config.width if width is None else width
        height = self.config.height if height is None else height

        if self.width != width or self.height != height:
            self.make_context_current()
            mj._render.mjr_resizeOffscreen(width, height, self._mjr_context)

        super().update(width, height)

    def render(self) -> np.ndarray:
        """Render the scene to an offscreen buffer and read the pixels from the buffer."""
        super().render()

        out = np.empty((self.height, self.width, 3), dtype=np.uint8)
        mj.mjr_readPixels(out, None, self.viewport, self._mjr_context)

        return np.flipud(out)


class MjCambrianOnscreenViewer(MjCambrianViewer):
    """This is an onscreen viewer. It renders the scene to an onscreen glfw window.

    It is based on of the `gymnasium.envs.mujoco.mujoco_renderer.WindowViewer`. Main
    differences are that it uses a more basic viewer without a menu and other features
    implemented by this main class. These can be added in the future, if desired,
    though the initial idea of this class was to simplify the interface.
    """

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        config: MjCambrianRendererConfig,
        camera: mj.MjvCamera,
    ):
        is_resizeable = config.width is None or config.height is None
        is_fullscreen = config.fullscreen is not None and config.fullscreen

        super().__init__(model, data, config, camera)

        self._last_mouse_x: int = 0
        self._last_mouse_y: int = 0
        self._is_paused: bool = False

        glfw.init()

        video_mode = glfw.get_video_mode(glfw.get_primary_monitor())
        monitor_width, monitor_height = video_mode.size
        width = self.config.width if self.config.width else monitor_width // 2
        height = self.config.height if self.config.height else monitor_height // 2

        glfw.window_hint(glfw.RESIZABLE, int(is_resizeable))
        glfw.window_hint(glfw.VISIBLE, 1)
        self.window = glfw.create_window(width, height, "MjCambrian", None, None)
        self.default_window_pos = glfw.get_window_pos(self.window)
        self.fullscreen(is_fullscreen)

        self.setup_context()

        self.width, self.height = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = self.width / window_width

        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW, self._mjr_context)
        glfw.swap_interval(1)

    def make_context_current(self):
        super().make_context_current()
        glfw.make_context_current(self.window)

    def render(self) -> None:
        if self.window is None:
            return
        elif glfw.window_should_close(self.window):
            self.close()
            return

        self.width, self.height = glfw.get_framebuffer_size(self.window)

        super().render()

        glfw.swap_buffers(self.window)
        glfw.poll_events()

        if self._is_paused:
            self.render()

    def close(self):
        super().close()

        if self.window is not None:
            if glfw.get_current_context() == self.window:
                glfw.make_context_current(None)
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    # ====================

    def fullscreen(self, fullscreen: bool):
        """Set the window to be fullscreen or not."""
        if self.window is None:
            return

        if fullscreen:
            size = glfw.get_video_mode(glfw.get_primary_monitor()).size
            glfw.set_window_monitor(
                self.window,
                glfw.get_primary_monitor(),
                0,
                0,
                size.width,
                size.height,
                glfw.DONT_CARE,
            )
        else:
            glfw.set_window_monitor(
                self.window,
                None,
                *self.default_window_pos,
                self.width,
                self.height,
                glfw.DONT_CARE,
            )

    # ====================

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


VIEWERS: Dict[str, MjCambrianViewer] = {}


class MjCambrianRenderer:
    """
    NOTE: This render doesn't currently support depth or segmentation sensing.
    """

    metadata = {"render.modes": ["rgb_array", "human"]}

    def __init__(self, config: MjCambrianRendererConfig):
        self.config = config
        self.config.setdefault("use_shared_context", True)

        self.model: mj.MjModel = None
        self.data: mj.MjData = None
        self.camera: mj.MjvCamera = mj.MjvCamera()  # shared camera

        # Maps render_mode to viewer.
        self._viewers: Dict[str, MjCambrianViewer] = {}

        self._record = False
        self._image_buffer: List[np.ndarray] = []

    def reset(self, model: mj.MjModel, data: mj.MjData) -> Dict[str, np.ndarray]:
        self.model = model
        self.data = data

        self.config.setdefault("width", self.model.vis.global_.offwidth)
        self.config.setdefault("height", self.model.vis.global_.offheight)

        self._image_buffer.clear()

        for render_mode in self.config.render_modes:
            if render_mode in self._viewers:
                continue
            self._viewers[render_mode] = self._create_viewer(model, data, render_mode)

        return self.render()

    def update(self, width: int, height: int):
        assert width is not None and height is not None, "Width and height must be set."

        self.config.width = width
        self.config.height = height

        for viewer in self._viewers.values():
            viewer.update(width, height)

    def render(self) -> np.ndarray | None:
        image: np.ndarray | None = None
        for viewer in self._viewers.values():
            # Since the viewers may be shared between renderers, we need to call update
            # to update the underlying buffers, if needed.
            viewer.update(self.config.width, self.config.height)

            if isinstance(viewer, MjCambrianOffscreenViewer):
                image = viewer.render()
                if image is None:
                    self.close()
            else:
                viewer.render()

            viewer.clear_overlays()

        if self.record:
            assert image is not None, "Image must not be None when recording."
            self._image_buffer.append(image)

        return image

    def add_overlay(
        self,
        overlay: np.ndarray | str,
        pos: Tuple[int, int] | np.ndarray,
        *,
        render_mode: Optional[str] = None,
    ):
        """Add an overlay to the image. Alias to add_image_overlay and add_text_overlay."""
        assert len(pos) == 2, "pos must be a tuple of length 2."

        for mode, viewer in self._viewers.items():
            if render_mode is not None and mode != render_mode:
                continue
            viewer.add_overlay(overlay, pos)

    def add_image_overlay(
        self,
        overlay: np.ndarray,
        pos: Tuple[int, int] | np.ndarray,
        *,
        render_mode: Optional[str] = None,
    ):
        """Add an image overlay to the image.

        Args:
            overlay (np.ndarray): The overlay. Will be placed at the given pos on
                top of the image.
            pos (Tuple[int, int] | np.ndarray): The position of the overlay. This is
                relative to the bottom-left corner of the image. The height and width is
                assumed from the image. Fmt: (left, top)

        Keyword Args:
            render_mode (Optional[str]): The render mode to add the overlay to. If
                None, the overlay is added to all render modes.
        """
        assert len(pos) == 2, "pos must be a tuple of length 2."

        for mode, viewer in self._viewers.items():
            if render_mode is not None and mode != render_mode:
                continue
            viewer.add_image_overlay(overlay, pos)

    def add_text_overlay(
        self,
        overlay: Any,
        pos: Optional[Tuple[int, int] | np.ndarray] = None,
        *,
        render_mode: Optional[str] = None,
    ):
        """Add an text overlay to the image.

        Args:
            overlay (Any): The overlay. Assumes str(overlay) is valid.
            pos (Optional[Tuple[int, int] | np.ndarray]): The position of the overlay.
                This is relative to the bottom-left corner of the image. The height and
                width is assumed from the image. Fmt: (left, top)

        Keyword Args:
            render_mode (Optional[str]): The render mode to add the overlay to. If
                None, the overlay is added to all render modes.
        """
        for mode, viewer in self._viewers.items():
            if render_mode is not None and mode != render_mode:
                continue
            viewer.add_text_overlay(overlay, pos)

    def close(self):
        for viewer in self._viewers.values():
            viewer.close()

        glfw.terminate()

    def __del__(self):
        self.close()

    # ====================

    @property
    def record(self) -> bool:
        return self._record

    @record.setter
    def record(self, value: bool):
        assert (
            "rgb_array" in self.config.render_modes
        ), "Must be in rgb_array mode to record."
        self._record = value

    def save_gif(self, path: Path | str):
        import imageio

        duration = 1000 * 1 / 50
        print(f"Saving visualization at {path}...")
        imageio.mimsave(path, self._image_buffer, loop=0, duration=duration)
        print(f"Saved visualization at {path}")

    # ====================

    def set_option(self, option: str, value: Any, index: Optional[slice | int] = None):
        """Set an option in the underlying scene."""
        for viewer in self._viewers.values():
            assert hasattr(viewer.scene_option, option), f"Invalid option `{option}`."
            if index is not None:
                getattr(viewer.scene_option, option)[index] = value
            else:
                setattr(viewer.scene_option, option, value)

    # ====================

    @property
    def viewer(self) -> MjCambrianViewer:
        assert len(self._viewers) == 1, "There must be exactly one viewer."
        return list(self._viewers.values())[0]

    @property
    def viewers(self) -> Dict[str, MjCambrianViewer]:
        return self._viewers

    @property
    def is_running(self) -> bool:
        for viewer in self._viewers.values():
            if viewer.is_closed:
                return False
        return True

    def _create_viewer(
        self, model: mj.MjModel, data: mj.MjData, render_mode: str
    ) -> MjCambrianViewer:
        viewer: MjCambrianViewer = None
        if self.config.use_shared_context and render_mode in VIEWERS:
            viewer = VIEWERS[render_mode]
        elif render_mode == "rgb_array":
            viewer = MjCambrianOffscreenViewer(model, data, self.config.copy(), self.camera)
        elif render_mode == "human":
            viewer = MjCambrianOnscreenViewer(model, data, self.config.copy(), self.camera)
        else:
            raise ValueError(f"Invalid render mode `{render_mode}`.")

        if self.config.use_shared_context:
            VIEWERS[render_mode] = viewer
        return viewer


if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path
    from cambrian_xml import MjCambrianXML

    parser = argparse.ArgumentParser(description="MjCambrianRenderer")

    args = parser.parse_args()

    YAML = """
    render_modes: ['human']

    max_geom: 1000

    width: 640
    height: 480

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
    renderer.update(config.width, config.height)

    while renderer.is_running:
        h, w = 100, 100
        renderer.add_overlay(
            np.full((h, w, 3), [255, 255, 0], dtype=np.uint8), (10, 10)
        )
        image = renderer.render()
        if image is None:
            continue

        cv2.imshow("image", image[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
