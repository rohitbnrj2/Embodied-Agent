from typing import List, Tuple
import numpy as np
import mujoco as mj


class Renderer(mj.Renderer):
    """This is an extension of the mujoco renderer helper class. It allows dynamically
    changing the width and height of the rendering window. See `mj.Renderer` for further
    documentation.

    TODO: Honestly, we probably could just write our own and not even use the mujoco one
    not sure about the performance implications of deleting and creating new contexts
    on each image render. Could maybe have one renderer per eye?
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_context(self, width: int, height: int):
        """This method facilitates the dynamic resizing of the rendering window. It will
        reset the OpenGL context and the mujoco renderer context to match the new height
        and width."""

        self._width = width
        self._height = height
        self._rect = mj._render.MjrRect(0, 0, width, height)

        # Clear the old contexts
        del self._gl_context
        del self._mjr_context

        # Make the new ones
        self._gl_context = mj.gl_context.GLContext(width, height)
        self._gl_context.make_current()
        self._mjr_context = mj._render.MjrContext(
            self._model, mj._enums.mjtFontScale.mjFONTSCALE_150.value
        )
        mj._render.mjr_setBuffer(
            mj._enums.mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context
        )


class Eye:
    def __init__(self, name: str, model: mj.MjModel, data: mj.MjData):
        self.name = name
        self._model = model
        self._data = data

        self.fixedcamid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, name)
        assert self.fixedcamid != -1, f"Camera '{name}' not found."

        self._camera = mj.MjvCamera()
        self._camera.type = mj.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = self.fixedcamid

        # populates with default width and height, to be overwritten anyways
        self._renderer = Renderer(model)
        # TOOD(ktiwary): figure out we creating renderes adds unnecessary overhead
        self._depth_renderer = Renderer(model)
        self._depth_renderer.enable_depth_rendering()

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self._data.cam_xpos[self.fixedcamid])

    @position.setter
    def position(self, position: List):
        self._data.cam_xpos[self.fixedcamid] += np.asarray(position)

    @property
    def rotation(self) -> np.ndarray:
        mat0 = np.asarray(self._model.cam_mat0[self.fixedcamid]).reshape(3, 3)
        xmat = np.asarray(self._data.cam_xmat[self.fixedcamid]).reshape(3, 3)
        return np.matmul(mat0, xmat)

    @rotation.setter
    def rotation(self, rotation: List):
        mat0 = self._model.cam_mat0[self.fixedcamid].reshape(3, 3)
        rot = np.asarray(rotation).reshape(3, 3)
        self._data.cam_xmat[self.fixedcamid] = np.matmul(mat0, rot).flatten()

    def render(self) -> np.ndarray:
        self._renderer.reset_context(*self.resolution)
        self._renderer.update_scene(self._data, self._camera)
        return self._renderer.render()

    def render_depth(self) -> np.ndarray:
        self._depth_renderer.reset_context(*self.resolution)
        self._depth_renderer.update_scene(self._data, self._camera)
        return self._depth_renderer.render()

    def render_psf(self) -> np.ndarray:
        rgb = self.render()
        depth = self.render_depth()
        # depth dependent convolution with the psf 

    def _get_cam_attr(self, obj, attr):
        return getattr(obj, attr)[self.fixedcamid]

    def _set_cam_attr(self, value, obj, attr):
        getattr(obj, attr)[self.fixedcamid] = value

    fovy: float = property(
        lambda self: self._get_cam_attr(self._model, "cam_fovy"),
        lambda self, value: self._set_cam_attr(value, self._model, "cam_fovy"),
    )
    sensorsize: Tuple[float, float] = property(
        lambda self: self._get_cam_attr(self._model, "cam_sensorsize"),
        lambda self, value: self._set_cam_attr(value, self._model, "cam_sensorsize"),
    )
    resolution: Tuple[int, int] = property(
        lambda self: self._get_cam_attr(self._model, "cam_resolution"),
        lambda self, value: self._set_cam_attr(value, self._model, "cam_resolution"),
    )
    intrinsic: Tuple[float, float, float, float] = property(
        lambda self: self._get_cam_attr(self._model, "cam_intrinsic"),
        lambda self, value: self._set_cam_attr(value, self._model, "cam_intrinsic"),
    )


if __name__ == "__main__":
    # ==================
    # Parse the args
    # ==================

    import argparse
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--title", type=str, default=None)
    parser.add_argument(
        "-xn", "--xnum", type=int, default=10, help="Number of eyes in the x direction"
    )
    parser.add_argument(
        "-yn", "--ynum", type=int, default=10, help="Number of eyes in the y direction"
    )
    parser.add_argument(
        "-xr",
        "--xrange",
        type=int,
        default=np.pi / 2,
        help="Range in the x direction",
    )
    parser.add_argument(
        "-yr",
        "--yrange",
        type=int,
        default=np.pi / 2,
        help="Range in the y direction",
    )
    parser.add_argument(
        "-w", "--width", type=int, default=20, help="Width in pixels of each eye"
    )
    parser.add_argument(
        "-H", "--height", type=int, default=20, help="Height in pixels of each eye"
    )
    parser.add_argument("--no-demo", action="store_true", help="Don't run the demo")
    parser.add_argument("--plot", action="store_true", help="Plot the demo")
    parser.add_argument("--save", action="store_true", help="Save the demo")
    parser.add_argument("--test", action="store_true", help="Run the tests")
    parser.add_argument("--supercloud", action="store_true", help="Run the tests")

    args = parser.parse_args()

    if args.supercloud:
        # set some supercloud specific params
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ["DISPLAY"] = ":0"
        os.environ["MUJOCO_GL"]="osmesa"
        # export LD_PRELOAD=/usr/lib/libGL.so.1 && export MUJOCO_GL=egl
        # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

    if not args.plot and not args.save and not args.no_demo:
        print("Warning: No output specified. Use --plot or --save to see the demo.")

    if not args.test:
        print("Warning: No tests specified. Use --test to run the tests.")

    # ==================
    # Create the cameras
    # ==================

    XML = f"""
    <mujoco>
        <statistic center="0 0 0.1" extent="0.6" meansize=".05"/>
        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="-20" elevation="-20" ellipsoidinertia="true"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        </asset>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
            <body pos="0 0 0.1">
                <camera name="eye" pos="-0.15 0 0.1" quat="0.5 0.5 0.5 0.5" mode="fixed" resolution="{args.width} {args.height}"/>
                <camera name="eye1" pos="0 0 0" mode="fixed"/>
                <camera name="eye2" pos="0 0 0.1" mode="fixed" resolution="10 10"/>

                <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
                <geom name="box1" type="box" size="0.1 0.1 0.1" pos="-1 0 0.2" condim="1" rgba="0 0 1 1" quat="1 0 0 0"/>
                <geom name="box2" type="box" size="0.1 0.1 0.1" pos="-1 0.25 0.2" condim="1" rgba="0 1 0 1" quat="1 0 0 0"/>
                <geom name="box3" type="box" size="0.1 0.1 0.1" pos="-1 -0.25 0.2" rgba="1 0 0 1" condim="1" quat="1 0 0 0"/>
            </body>
        </worldbody>
    </mujoco>
    """

    model = mj.MjModel.from_xml_string(XML)
    data = mj.MjData(model)
    mj.mj_step(model, data)

    # ==============
    # Run some tests
    # ==============

    if args.test:
        eye1 = Eye("eye1", model, data)
        eye2 = Eye("eye2", model, data)

        eyes = [eye1.fixedcamid, eye2.fixedcamid]

        assert eye1.fovy == 45.0
        eye1.fovy = 10
        assert eye1.fovy == 10.0
        assert model.cam_fovy[eye1.fixedcamid] == 10.0
        assert np.allclose(model.cam_fovy[eyes], [10.0, 45.0])

        assert np.allclose(eye2.sensorsize, [0.0, 0.0])
        assert np.allclose(model.cam_sensorsize[eyes], [[0.0, 0.0], [0.0, 0.0]])
        eye1.sensorsize = [0.1, 1.1]
        assert np.allclose(eye1.sensorsize, [0.1, 1.1])
        assert np.allclose(model.cam_sensorsize[eyes], [[0.1, 1.1], [0.0, 0.0]])

        # should be all black
        image1 = eye1.render()
        assert np.allclose(image1.shape, [1, 1, 3])  # default is 1, 1
        assert np.allclose(image1, np.full_like(image1, [29, 48, 68]))

        image2 = eye2.render()
        assert np.allclose(image2.shape, [10, 10, 3])

        assert np.allclose(eye1.position, [0.0, 0.0, 0.1])
        assert np.allclose(eye2.position, [0.0, 0.0, 0.2])
        assert np.allclose(model.cam_pos0[eyes], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
        assert np.allclose(eye1.rotation, np.eye((3)))
        eye2.position = [0.1, 0.2, 0.3]
        assert np.allclose(eye2.position, [0.1, 0.2, 0.5])

    # ==============
    # Demo
    # ==============
    if args.no_demo:
        exit()

    # Quick test to make sure the renderer is working
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R

    eye = Eye("eye", model, data)

    X_NUM = args.xnum
    Y_NUM = args.ynum
    X_RANGE = args.xrange
    Y_RANGE = args.yrange

    fig, ax = plt.subplots(X_NUM, Y_NUM, figsize=(Y_NUM, X_NUM))
    if X_NUM == 1 and Y_NUM == 1:
        ax = np.array([[ax]])

    for i in range(X_NUM):
        x_angle = -X_RANGE / 2 + X_RANGE / (X_NUM + 1) * (i + 1)
        for j in range(Y_NUM):
            y_angle = -Y_RANGE / 2 + Y_RANGE / (Y_NUM + 1) * (j + 1)

            r = R.from_euler("xy", [x_angle, y_angle])
            eye.rotation = r.as_matrix()

            print(f"Rendering eye at ({x_angle:0.2f}, {y_angle:0.2f})...")
            image = eye.render()
            print(f"Done rendering eye at ({x_angle:0.2f}, {y_angle:0.2f}).")

            if args.plot:
                ax[Y_NUM - i - 1, X_NUM - j - 1].imshow(image)

                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])

    fig.suptitle(args.title)
    plt.subplots_adjust(wspace=0, hspace=0)

    if args.save:
        # save the figure without the frame
        plt.axis('off')
        plt.savefig(f"{args.title}.png", bbox_inches="tight", dpi=300)

    if args.plot:
        fig_manager = plt.get_current_fig_manager()
        fig_manager.full_screen_toggle()
        plt.show()