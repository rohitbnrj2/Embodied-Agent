from typing import List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, replace

import glfw
import numpy as np
import mujoco as mj
import OpenGL.GL as GL
import imageio
import cv2

from cambrian.evolution_envs.three_d.mujoco.config import MjCambrianRendererConfig
from cambrian.evolution_envs.three_d.mujoco.utils import get_camera_id, get_body_id

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

def convert_depth_to_rgb(model: mj.MjModel, depth: np.ndarray) -> np.ndarray:
    """https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/renderer.py"""
    # Get the distances to the near and far clipping planes.
    extent = model.stat.extent
    near = model.vis.map.znear * extent
    far = model.vis.map.zfar * extent

    # Calculate OpenGL perspective matrix values in float32 precision
    # so they are close to what glFrustum returns
    # https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml
    zfar = np.float32(far)
    znear = np.float32(near)
    c_coef = -(zfar + znear) / (zfar - znear)
    d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)

    # In reverse Z mode the perspective matrix is transformed by the following
    c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
    d_coef = np.float32(-0.5) * d_coef

    # We need 64 bits to convert Z from ndc to metric depth without noticeable
    # losses in precision
    out_64 = depth.astype(np.float64)

    # Undo OpenGL projection
    # Note: We do not need to take action to convert from window coordinates
    # to normalized device coordinates because in reversed Z mode the mapping
    # is identity
    out_64 = d_coef / (out_64 + c_coef)

    # Cast result back to float32 for backwards compatibility
    # This has a small accuracy cost
    depth[:] = out_64.astype(np.float32)

    return depth

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

GL_CONTEXT: mj.gl_context.GLContext = None
MJR_CONTEXT: mj.MjrContext = None

class MjCambrianViewer(ABC):
    def __init__(self, config: MjCambrianRendererConfig):
        self.config = config

        self.model: mj.MjModel = None
        self.data: mj.MjData = None
        self.scene: mj.MjvScene = None
        self.scene_option = mj.MjvOption()
        self.camera: mj.MjvCamera = mj.MjvCamera()
        self.viewport: mj.MjrRect = None

        self._gl_context: mj.gl_context.GLContext = None
        self._mjr_context: mj.MjrContext = None

    def reset(self, model: mj.MjModel, data: mj.MjData, width: int, height: int):
        self.model = model
        self.data = data

        self.reset_camera()

        self.scene = mj.MjvScene(model=model, maxgeom=self.config.max_geom)

        # Disable ~all mj flags
        self.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = False
        self.scene.flags[mj.mjtRndFlag.mjRND_WIREFRAME] = False
        self.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = False
        self.scene.flags[mj.mjtRndFlag.mjRND_ADDITIVE] = False
        self.scene.flags[mj.mjtRndFlag.mjRND_SKYBOX] = False
        self.scene.flags[mj.mjtRndFlag.mjRND_FOG] = False
        self.scene.flags[mj.mjtRndFlag.mjRND_HAZE] = False
        self.scene.flags[mj.mjtRndFlag.mjRND_SEGMENT] = False
        self.scene.flags[mj.mjtRndFlag.mjRND_IDCOLOR] = False
        self.scene.flags[mj.mjtRndFlag.mjRND_CULL_FACE] = True

        # NOTE: All shared contexts much match either onscreen or offscreen. And their
        # height and width most likely must match as well.
        font_scale = mj.mjtFontScale.mjFONTSCALE_50
        if self.config.use_shared_context:
            global GL_CONTEXT, MJR_CONTEXT
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

    def reset_camera(self):
        """Setup the camera."""
        camera_config = self.config.camera_config
        if camera_config is None:
            return

        assert not (
            camera_config.type and camera_config.typename
        ), "Camera type and typename are mutually exclusive."
        assert not (
            camera_config.fixedcamid and camera_config.fixedcamname
        ), "Camera fixedcamid and fixedcamname are mutually exclusive."
        assert not (
            camera_config.trackbodyid and camera_config.trackbodyname
        ), "Camera trackbodyid and trackbodyname are mutually exclusive."

        if camera_config.type is not None:
            self.camera.type = camera_config.type
        if camera_config.typename is not None:
            typename = f"mjCAMERA_{camera_config.typename.upper()}"
            self.camera.type = getattr(mj.mjtCamera, typename)
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

    def read_pixels(self) -> Tuple[np.ndarray, np.ndarray]:
        width, height = self.viewport.width, self.viewport.height
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32)
        mj.mjr_readPixels(rgb, depth, self.viewport, self._mjr_context)
        return np.flipud(rgb), np.flipud(depth)

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
    def reset(self, model: mj.MjModel, data: mj.MjData, width: int, height: int):
        super().reset(model, data, width, height)

        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr_context)

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

        self.config.setdefault("resizeable", False)
        self.config.setdefault("fullscreen", False)

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
            glfw.init()

            glfw.window_hint(glfw.RESIZABLE, int(self.config.resizeable))
            glfw.window_hint(glfw.VISIBLE, 1)
            self.window = glfw.create_window(width, height, "MjCambrian", None, None)
            self.default_window_pos = glfw.get_window_pos(self.window)
        else:
            glfw.set_window_size(self.window, width, height)
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
    metadata = {"render.modes": ["human", "rgb_array", "depth_array"]}

    def __init__(self, config: MjCambrianRendererConfig):
        self.config = config
        self.config.setdefault("max_geom", 1000)

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

    def reset(self, model: mj.MjModel, data: mj.MjData) -> np.ndarray | None:
        self.config.setdefault("width", model.vis.global_.offwidth)
        self.config.setdefault("height", model.vis.global_.offheight)

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

        rgb, depth = self.viewer.read_pixels()
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
        assert len(self._rgb_buffer) > 0, "Cannot save empty buffer."

        print(f"Saving visualizations at {path}...")

        path = Path(path)
        rgb_buffer = np.array(self._rgb_buffer[:-1])
        if "mp4" in save_types:
            mp4 = path.with_suffix(".mp4")
            writer = imageio.get_writer(mp4, fps=self.config.fps)
            for image in rgb_buffer:
                writer.append_data(image)
            writer.close()
        if "png" in save_types:
            png = path.with_suffix(".png")
            imageio.imwrite(png, rgb_buffer[-1])
        if "gif" in save_types:
            duration = 1000 / self.config.fps
            gif = path.with_suffix(".gif")
            imageio.mimwrite(gif, rgb_buffer, loop=0, duration=duration)

        print(f"Saved visualization at {path}")

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


if __name__ == "__main__":
    import argparse
    import yaml
    import time
    from pathlib import Path
    from cambrian_xml import MjCambrianXML

    parser = argparse.ArgumentParser()

    parser.add_argument("--speed-test", action="store_true", help="Run speed test.")

    args = parser.parse_args()


    YAML = """
    render_modes: ['rgb_array', 'depth_array']

    max_geom: 1000

    width: 640
    height: 480

    resizeable: true
    fullscreen: true

    use_shared_context: true

    camera_config:
        typename: 'fixed'
        fixedcamid: 0
    """

    xml = MjCambrianXML(Path(__file__).parent / "models" / "test.xml")
    xml.add(
        xml.find(".//worldbody"),
        "camera",
        pos="0 0 0.5",
        quat="0.5 0.5 0.5 0.5",
        resolution="10 10",
    )

    model = mj.MjModel.from_xml_string(xml.to_string())
    data = mj.MjData(model)
    mj.mj_step(model, data)

    config = MjCambrianRendererConfig.from_dict(yaml.safe_load(YAML))
    renderer = MjCambrianRenderer(config)
    renderer.reset(model, data)

    if args.speed_test:
        print("Starting speed test...")
        num_frames = 100
        t0 = time.time()
        for _ in range(num_frames):
            renderer.render()
        t1 = time.time()
        print(f"Rendered {num_frames} frames in {t1 - t0} seconds.")
        print(f"Average FPS: {num_frames / (t1 - t0)}")
        exit()

    while renderer.is_running():
        out = renderer.render()
        if out is None:
            continue

        if isinstance(out, tuple):
            rgb, depth = out
        else:
            rgb, depth = out, None

        cv2.imshow("image", rgb[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if depth is not None:
            depth_rgb = convert_depth_to_rgb(model, depth)
            cv2.imshow("depth", depth_rgb)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    renderer.close()
