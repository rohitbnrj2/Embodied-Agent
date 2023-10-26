from typing import List, Tuple
import numpy as np

import mujoco as mj
from gymnasium import spaces

from cambrian_xml import MjCambrianXML
from config import MjCambrianEyeConfig
from renderer import MjCambrianRenderer

RENDERER: MjCambrianRenderer = None


class MjCambrianEye:
    """Defines an eye for the cambrian environment. It essentially wraps a mujoco Camera
    object and provides some helper methods for rendering and generating the XML. The
    eye is attached to the parent body such that movement of the parent body will move
    the eye.

    Args:
        config (MjCambrianEyeConfig): The configuration for the eye.
    """

    _renderer: MjCambrianRenderer = None
    """
    NOTE: this is a static renderer; it's shared across all eyes. First eye will
          initialize. reset_context should be called before rendering to reset the
          resolution
    NOTE #2: the renderer is initialized by an outside entity with access to the 
             mj.MjModel and mj.MjData objects.
    """

    def __init__(self, config: MjCambrianEyeConfig):
        self.config = config

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._camera: mj.MjvCamera = None

    def generate_xml(
        self, parent_xml: MjCambrianXML, parent_body_name: str
    ) -> MjCambrianXML:
        """Generate the xml for the eye.

        In order to combine the xml for an eye with the xml for the animal that it's
        attached to, we need to replicate the path with which we want to attach the eye.
        For instance, if the body with which we want to attach the eye to is at
        `mujoco/worldbody/torso`, then we need to replicate that path in the new xml.
        This is kind of difficult with the `xml` library, but we'll utilize the
        `CambrianXML` helpers for this.

        Args:
            parent_xml (MjCambrianXML): The xml of the parent body. Used as a reference to
            extract the path of the parent body.
            parent_body_name (str): The name of the parent body. Will search for the
            body tag with this name, i.e. <body name="<parent_body_name>" ...>.
        """

        xml = MjCambrianXML.make_empty()

        # Get the parent body reference
        parent_body = parent_xml.find(".//body", name=parent_body_name)
        assert parent_body is not None, f"Could not find body '{parent_body_name}'."

        # Iterate through the path and add the parent elements to the new xml
        parent = None
        elements, _ = parent_xml.get_path(parent_body)
        for element in elements:
            if (
                temp_parent := xml.find(f".//{element.tag}", **element.attrib)
            ) is not None:
                # If the element already exists, then we'll use that as the parent
                parent = temp_parent
                continue
            parent = xml.add(parent, element.tag, **element.attrib)

        # Finally add the camera element at the end
        assert parent is not None
        xml.add(parent, "camera", **self.config.to_xml_kwargs())

        return xml

    def reset(self, model: mj.MjModel, data: mj.MjData):
        """Sets up the camera for rendering. This should be called before rendering
        the first time."""
        self._model = model
        self._data = data

        fixedcamid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, self.name)
        assert fixedcamid != -1, f"Camera '{self.name}' not found."

        self._camera = mj.MjvCamera()
        self._camera.type = mj.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = fixedcamid

        # Update the resolution
        self.resolution = [int(x) for x in self.config.resolution.split(" ")]

        # TODO: Set the new fov based on the additional pixels that need to be rendered 
        # for the psf. probs wanna use focal
        # self.fovy *= self.padded_resolution[1] / self.resolution[1]

        # Create the render. It's static, so we only want to create it once for all eyes
        global RENDERER
        if RENDERER is None:
            RENDERER = MjCambrianRenderer(model)
            RENDERER._scene_option.geomgroup[1] = 0

        return self.step()

    def step(self) -> np.ndarray:
        """Simply calls `render(return_depth=False)`.
        See `render()` for more information."""
        return self.render(return_depth=False)

    def render(
        self, return_depth: bool = True
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Render the image from the camera. If `return_depth` is True, returns the both depth image and rgb, otherwise only rgb.

        NOTE: The actual rendered resolution is `self.padded_resolution`. This is so
        that the convolution with the psf filter doesn't cut off any of the image. After
        the psf is applied, the image will be cropped to `self.resolution`.
        """
        RENDERER.reset_context(*self.padded_resolution)
        RENDERER.update_scene(self._data, self._camera)

        rgb = RENDERER.render().transpose(1, 0, 2)  # convert to (W, H, C)

        if return_depth:
            RENDERER.enable_depth_rendering()
            depth = RENDERER.render().transpose(1, 0)  # convert to (W, H)
            RENDERER.disable_depth_rendering()

        # TODO: do psf stuff here

        rgb = self._crop(rgb)
        depth = self._crop(depth) if return_depth else None

        return rgb if not return_depth else (rgb, depth)

    def _crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the image to the resolution specified in the config."""
        resolution = self.resolution
        cw, ch = int(np.ceil(resolution[0] / 2)), int(np.ceil(resolution[1] / 2))
        ox, oy = 1 if resolution[0] == 1 else 0, 1 if resolution[1] == 1 else 0
        bl = (image.shape[0] // 2 - cw + ox, image.shape[1] // 2 - ch + oy)
        tr = (image.shape[0] // 2 + cw, image.shape[1] // 2 + ch)
        return image[bl[0] : tr[0], bl[1] : tr[1]]

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def fixedcamid(self) -> int:
        return self._camera.fixedcamid

    @property
    def observation_space(self) -> spaces.Box:
        """The observation space is just the rgb image."""

        observation_space = spaces.Box(
            0, 255, shape=(*self.resolution, 3), dtype=np.uint8
        )
        return observation_space

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the resolution of the camera.

        This method might be called before `self._model` is set, so we need to parse
        the config to get the resolution
        """
        if self._model is None:
            return [int(x) for x in self.config.resolution.split(" ")]
        else:
            return self._get_mj_attr(self._model, "cam_resolution")

    @resolution.setter
    def resolution(self, value: Tuple[int, int]):
        """Set the resolution of the camera.

        Like the getter, this might be called before `self._model` is set, so we need to
        parse the config to get the resolution
        """
        if self._model is None:
            self.config.resolution = f"{value[0]} {value[1]}"
        else:
            self._set_mj_attr(value, self._model, "cam_resolution")

    @property
    def padded_resolution(self) -> Tuple[int, int]:
        """This is the resolution padded with the filter size / 2. The actual render
        call should use this resolution instead and the cropped after the psf is
        applied."""
        filter_size = self.config.filter_size
        return (
            self.resolution[0] + filter_size[0],
            self.resolution[1] + filter_size[1],
        )

    def _get_mj_attr(self, obj, attr):
        """Helper method for getting an attribute from the mujoco data structures."""
        return getattr(obj, attr)[self.fixedcamid]

    def _set_mj_attr(self, value, obj, attr):
        """Helper method for setting an attribute from the mujoco data structures."""
        getattr(obj, attr)[self.fixedcamid] = value

    fovy: float = property(
        lambda self: self._get_mj_attr(self._model, "cam_fovy"),
        lambda self, value: self._set_mj_attr(value, self._model, "cam_fovy"),
    )
    sensorsize: Tuple[float, float] = property(
        lambda self: self._get_mj_attr(self._model, "cam_sensorsize"),
        lambda self, value: self._set_mj_attr(value, self._model, "cam_sensorsize"),
    )
    intrinsic: Tuple[float, float, float, float] = property(
        lambda self: self._get_mj_attr(self._model, "cam_intrinsic"),
        lambda self, value: self._set_mj_attr(value, self._model, "cam_intrinsic"),
    )


if __name__ == "__main__":
    # ==================
    # Parse the args
    # ==================

    import argparse

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
    parser.add_argument(
        "-sc", "--supercloud", action="store_true", help="Supercloud specific config"
    )

    args = parser.parse_args()

    if args.supercloud:
        import os

        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ["DISPLAY"] = ":0"
        os.environ["MUJOCO_GL"] = "osmesa"

    if not args.plot and not args.save and not args.no_demo:
        print("Warning: No output specified. Use --plot or --save to see the demo.")

    if not args.test:
        print("Warning: No tests specified. Use --test to run the tests.")

    # ==================
    # Create the cameras
    # ==================

    XML = """
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
            <body name="body" pos="0 0 0.1">
                <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
                <geom name="box1" type="box" size="0.1 0.1 0.1" pos="-1 0 0.2" condim="1" rgba="0 0 1 1" quat="1 0 0 0"/>
                <geom name="box2" type="box" size="0.1 0.1 0.1" pos="-1 0.25 0.2" condim="1" rgba="0 1 0 1" quat="1 0 0 0"/>
                <geom name="box3" type="box" size="0.1 0.1 0.1" pos="-1 -0.25 0.2" rgba="1 0 0 1" condim="1" quat="1 0 0 0"/>
            </body>
        </worldbody>
    </mujoco>
    """
    xml = MjCambrianXML.from_string(XML)

    from scipy.spatial.transform import Rotation as R
    default_rot = R.from_euler("xz", [np.pi / 2, np.pi / 2])

    if not args.no_demo:

        X_NUM = args.xnum
        Y_NUM = args.ynum
        X_RANGE = args.xrange
        Y_RANGE = args.yrange


        eyes: List[MjCambrianEye] = []
        for i in range(X_NUM):
            x_angle = -X_RANGE / 2 + X_RANGE / (X_NUM + 1) * (i + 1)
            for j in range(Y_NUM):
                y_angle = -Y_RANGE / 2 + Y_RANGE / (Y_NUM + 1) * (j + 1)

                quat = (default_rot * R.from_euler("xy", [x_angle, y_angle])).as_quat()

                eye_config = MjCambrianEyeConfig(
                    name=f"eye_{i}_{j}",
                    resolution=f"{args.width} {args.height}",
                    pos="0 0 0.2",
                    quat=" ".join(map(str, quat)),
                )
                eye = MjCambrianEye(eye_config)
                eyes.append(eye)

                xml += eye.generate_xml(xml, "body")
        print(xml)
    if args.test:
        eye1_config = MjCambrianEyeConfig(
            name="eye1", resolution="1 1", quat=" ".join(map(str, default_rot.as_quat()))
        )
        eye1 = MjCambrianEye(eye1_config)
        xml += eye1.generate_xml(xml, "body")

        eye2_config = MjCambrianEyeConfig(
            name="eye2", resolution="10 10", quat=" ".join(map(str, default_rot.as_quat()))
        )
        eye2 = MjCambrianEye(eye2_config)
        xml += eye2.generate_xml(xml, "body")

    model = mj.MjModel.from_xml_string(str(xml))
    data = mj.MjData(model)
    mj.mj_step(model, data)

    # ==============
    # Run some tests
    # ==============

    if args.test:
        eye1.reset(model, data)
        eye2.reset(model, data)

        test_eyes = [eye1.fixedcamid, eye2.fixedcamid]

        assert eye1.fovy == 45.0
        eye1.fovy = 10
        assert eye1.fovy == 10.0
        assert model.cam_fovy[eye1.fixedcamid] == 10.0
        assert np.allclose(model.cam_fovy[test_eyes], [10.0, 45.0])

        assert np.allclose(eye2.sensorsize, [0.0, 0.0])
        assert np.allclose(model.cam_sensorsize[test_eyes], [[0.0, 0.0], [0.0, 0.0]])
        eye1.sensorsize = [0.1, 1.1]
        assert np.allclose(eye1.sensorsize, [0.1, 1.1])
        assert np.allclose(model.cam_sensorsize[test_eyes], [[0.1, 1.1], [0.0, 0.0]])

        # should be all black
        image1 = eye1.render(return_depth=False)
        assert np.allclose(image1.shape, [1, 1, 3])  # default is 1, 1
        assert np.allclose(image1, np.full_like(image1, [5, 10, 15])), image1

        image2 = eye2.render(return_depth=False)
        print(image2.shape)
        assert np.allclose(image2.shape, [10, 10, 3])

    # ==============
    # Demo
    # ==============
    if args.no_demo:
        exit()

    # Quick test to make sure the renderer is working
    # TODO: The axes images and subplots are in weird spots. not sure why i need to flip
    import time
    import matplotlib.pyplot as plt

    if args.plot or args.save:
        fig, ax = plt.subplots(Y_NUM, X_NUM, figsize=(Y_NUM, X_NUM))
        if X_NUM == 1 and Y_NUM == 1:
            ax = np.array([[ax]])
        ax = np.flipud(ax)
        assert X_NUM == Y_NUM

    times = []
    start = time.time()
    for i in range(X_NUM):
        for j in range(Y_NUM):
            eye = eyes[i * X_NUM + j]
            t0 = time.time()
            image = eye.reset(model, data)
            t1 = time.time()
            times.append(t1 - t0)

            if args.plot or args.save:
                ax[j, i].imshow(image.transpose(1, 0, 2))

                ax[j, i].set_xticks([])
                ax[j, i].set_yticks([])
                ax[j, i].set_xticklabels([])
                ax[j, i].set_yticklabels([])

    print("Total time (including setup/plotting/etc):", time.time() - start)
    print("Total time (minus setup/plotting/etc):", sum(times))
    print("Median time:", np.median(times))
    print("Max time:", np.max(times))
    print("Min time:", np.min(times))

    if args.plot or args.save:
        fig.suptitle(args.title)
        plt.subplots_adjust(wspace=0, hspace=0)

    if args.save:
        # save the figure without the frame
        plt.axis("off")
        plt.savefig(f"{args.title}.png", bbox_inches="tight", dpi=300)

    if args.plot:
        fig_manager = plt.get_current_fig_manager()
        fig_manager.full_screen_toggle()
        plt.show()
