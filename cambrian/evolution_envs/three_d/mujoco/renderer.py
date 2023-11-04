import mujoco as mj

class MjCambrianRenderer(mj.Renderer):
    """This is an extension of the mujoco renderer helper class. It allows dynamically
    changing the width and height of the rendering window. See `mj.Renderer` for further
    documentation.

    TODO: Honestly, we probably could just write our own and not even use the mujoco one
    NOTE: In mujoco, it's possible to get depth and rgb in the same render call;
    however, the mj.Renderer.render method abstracts this away and requires two renders
    to occur. We should def write our own to reduce overhead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_context(self, width: int, height: int):
        """This method facilitates the dynamic resizing of the rendering window. It will
        reset the OpenGL context and the mujoco renderer context to match the new height
        and width."""

        # Check the buffer width/height
        buffer_width = self._model.vis.global_.offwidth
        buffer_height = self._model.vis.global_.offheight
        if width > buffer_width:
            raise ValueError(
                f"""
    Image width {width} > framebuffer width {buffer_width}. Either reduce the image
    width or specify a larger offscreen framebuffer in the model XML using the
    clause:
    <visual>
    <global offwidth="my_width"/>
    </visual>""".lstrip()
            )

        if height > buffer_height:
            raise ValueError(
                f"""
    Image height {height} > framebuffer height {buffer_height}. Either reduce the
    image height or specify a larger offscreen framebuffer in the model XML using
    the clause:
    <visual>
    <global offheight="my_height"/>
    </visual>""".lstrip()
            )

        if self._width == width and self._height == height:
            return

        self._width = width
        self._height = height
        self._rect = mj._render.MjrRect(0, 0, width, height)

        self._gl_context.make_current()
        mj._render.mjr_resizeOffscreen(width, height, self._mjr_context)