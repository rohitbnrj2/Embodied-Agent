from cambrian.renderer.renderer import (
    MjCambrianRenderer,
    MjCambrianRendererConfig,
    MjCambrianRendererSaveMode,
)  # noqa
from cambrian.renderer.render_utils import (
    resize_with_aspect_fill,
    convert_depth_to_rgb,
)  # noqa

# MuJoCo supports specifying the MuJoCo OpenGL backend via the MUJOCO_GL environment 
# variable. On Mac, Only the `cgl` backend is instantiated even though glfw is 
# supported. We'll override the default backend to use glfw instead of cgl if the
# MUJOCO_GL environment variable is set to `glfw`.
import platform
if platform.system() == 'Darwin':
    import os
    if os.environ.get('MUJOCO_GL', '').lower().strip() == 'glfw':
        import mujoco as mj
        from mujoco.glfw import GLContext
        mj.gl_context.GLContext = GLContext

__all__ = [
    "MjCambrianRenderer",
    "MjCambrianRendererConfig",
    "MjCambrianRendererSaveMode",
    "resize_with_aspect_fill",
    "convert_depth_to_rgb",
]
