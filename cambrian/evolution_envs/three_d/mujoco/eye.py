import time
from typing import Tuple, Dict
import numpy as np

import mujoco as mj
from gymnasium import spaces

from cambrian.evolution_envs.three_d.mujoco.cambrian_xml import MjCambrianXML
from cambrian.evolution_envs.three_d.mujoco.config import (
    MjCambrianEyeConfig,
    MjCambrianRendererConfig,
)
from cambrian.evolution_envs.three_d.mujoco.renderer import MjCambrianRenderer


class MjCambrianEye:
    """Defines an eye for the cambrian environment. It essentially wraps a mujoco Camera
    object and provides some helper methods for rendering and generating the XML. The
    eye is attached to the parent body such that movement of the parent body will move
    the eye.

    Args:
        config (MjCambrianEyeConfig): The configuration for the eye.
    """

    def __init__(self, config: MjCambrianEyeConfig):
        self.config = self._check_config(config)

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._last_obs: np.ndarray = None

        self._renderer = MjCambrianRenderer(config.renderer_config)
        self._render_depth = "depth_array" in config.renderer_config.render_modes

    def _check_config(self, config: MjCambrianEyeConfig) -> MjCambrianEyeConfig:
        """This will automatically set some of the config values if they are not
        specified.

        filter_size: set to [0, 0] if not specified
        fovy: if set, it must be between 0 and 180. focal, sensorsize, and fov must not
            be set.
        fov: if set, both values must be between 0 and 180. fovy and sensorsize must
            not be set.
        focal: if unset, set to [0.1, 0.1]. only set if fov is set.
        sensorsize: if unset, will set to the value calculated from fov and the padded
            resolution. the equation is sensorsize = (2 * focal * tan(fov / 2) * scale)
            where scale is padded_resolution / resolution. only set if fov is set.
        """
        assert config.name is not None, "Must specify a name for the eye."
        assert config.mode is not None, "Must specify a mode for the eye."
        assert config.pos is not None, "Must specify a position for the eye."
        assert config.quat is not None, "Must specify a quaternion for the eye."
        assert config.resolution is not None, "Must specify a resolution for the eye."
        assert (
            "rgb_array" in config.renderer_config.render_modes
        ), "Must specify 'rgb_array' in the render modes for the renderer config."

        config.setdefault("filter_size", [0, 0])
        if config.filter_size != [0, 0]:
            assert (
                "depth_array" in config.renderer_config.render_modes
            ), "Must specify 'depth_array' in the render modes for the renderer config."

        if config.fovy is not None:
            assert 0 < config.fovy < 180, "Invalid fovy."
            assert config.focal is None, "Cannot set both fovy and focal"
            assert config.sensorsize is None, "Cannot set both fovy and sensorsize"
            assert config.fov is None, "Cannot set both fovy and fov"

        if config.fov is not None:
            assert 0 < config.fov[0] < 180 and 0 < config.fov[1] < 180, "Invalid fov."
            assert config.fovy is None, "Cannot set both fov and fovy"
            assert config.sensorsize is None, "Cannot set both fov and sensorsize"

            config.setdefault("focal", [0.1, 0.1])

            # Adjust the FOV based on the padded resolution
            original_width, original_height = config.resolution
            padded_width, padded_height = (
                original_width + config.filter_size[0],
                original_height + config.filter_size[1],
            )

            scale_factor_width = padded_width / original_width
            scale_factor_height = padded_height / original_height

            # sensorsize = (2 * focal * tan(fov / 2)
            fovx, fovy = config.fov
            focalx, focaly = config.focal
            config.sensorsize = [
                2 * focalx * np.tan(np.radians(fovx) / 2) * scale_factor_width,
                2 * focaly * np.tan(np.radians(fovy) / 2) * scale_factor_height,
            ]

        # Set the height/width of the renderer equal to the resolution of the image
        config.renderer_config.width, config.renderer_config.height = config.resolution

        return config

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

        self.config.renderer_config.width = self.padded_resolution[0]
        self.config.renderer_config.height = self.padded_resolution[1]
        self._renderer.reset(model, data)

        # All animal geomgroups start at 2, and so we'll hide all them
        # We'll also hide all the sites after 2
        self._renderer.set_option("geomgroup", False, slice(2, None))
        self._renderer.set_option("sitegroup", False, slice(2, None))

        fixedcamid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, self.name)
        assert fixedcamid != -1, f"Camera '{self.name}' not found."
        self._renderer.viewer.camera.type = mj.mjtCamera.mjCAMERA_FIXED
        self._renderer.viewer.camera.fixedcamid = fixedcamid

        return self.step()

    def step(self) -> np.ndarray:
        """Simply calls `render` and sets the last observation.
        See `render()` for more information."""
        obs = self.render()
        self._last_obs = obs.copy()
        return obs

    def render(self) -> np.ndarray:
        """Render the image from the camera. If `return_depth` is True, returns the
        both depth image and rgb, otherwise only rgb.

        NOTE: The actual rendered resolution is `self.padded_resolution`. This is so
        that the convolution with the psf filter doesn't cut off any of the image. After
        the psf is applied, the image will be cropped to `self.resolution`.
        """

        rgb = self._renderer.render()
        if self._render_depth:
            rgb, depth = rgb

        # TODO: Apply PSF here

        return self._crop(rgb)

    def _crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the image to the resolution specified in the config."""
        if self.config.filter_size == [0, 0]:
            return image

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
    def observation_space(self) -> spaces.Box:
        """The observation space is just the rgb image.

        Fmt: Height, Width
        """

        observation_space = spaces.Box(
            0, 255, shape=(*self.resolution[::-1], 3), dtype=np.uint8
        )
        return observation_space

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the resolution of the camera.

        This method might be called before `self._model` is set, so we need to parse
        the config to get the resolution
        """
        return self.config.resolution

    @resolution.setter
    def resolution(self, value: Tuple[int, int]):
        """Set the resolution of the camera.

        Like the getter, this might be called before `self._model` is set, so we need to
        parse the config to get the resolution
        """
        self.config.resolution = value
        if self._model is not None:
            self._set_mj_attr(self.padded_resolution, self._model, "cam_resolution")

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

    @property
    def last_obs(self) -> np.ndarray:
        """The last observation returned by `self.render()`."""
        return self._last_obs

    @property
    def fov(self) -> Tuple[float, float]:
        """The field of view of the camera in degrees."""
        return self.config.fov


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
    parser.add_argument("--render-modes", nargs="+", default=["rgb_array", "human"])
    parser.add_argument("--quiet", action="store_true", help="Don't print the xml")

    args = parser.parse_args()

    if not args.plot and not args.save and not args.no_demo:
        print("Warning: No output specified. Use --plot or --save to see the demo.")

    # ==================
    # Create the cameras
    # ==================
    from pathlib import Path
    from scipy.spatial.transform import Rotation as R

    xml = MjCambrianXML(Path(__file__).parent / "models" / "test.xml")

    default_rot = R.from_euler("xz", [np.pi / 2, np.pi / 2])

    if not args.no_demo:
        X_NUM = args.xnum
        Y_NUM = args.ynum
        X_RANGE = args.xrange
        Y_RANGE = args.yrange

        renderer_config = MjCambrianRendererConfig(
            render_modes=args.render_modes, use_shared_context=True
        )

        eyes: Dict[str, MjCambrianEye] = {}
        for i in range(X_NUM):
            x_angle = -X_RANGE / 2 + X_RANGE / (X_NUM + 1) * (i + 1)
            for j in range(Y_NUM):
                y_angle = -Y_RANGE / 2 + Y_RANGE / (Y_NUM + 1) * (j + 1)

                quat = (default_rot * R.from_euler("xy", [y_angle, x_angle])).as_quat()

                name = f"eye_{i}_{j}"
                eye_config = MjCambrianEyeConfig(
                    mode="fixed",
                    name=name,
                    resolution=[args.width, args.height],
                    pos=[0, 0, 0.3],
                    quat=list(quat),
                    renderer_config=renderer_config.copy(),
                )
                eye = MjCambrianEye(eye_config)
                eyes[name] = eye

                xml += eye.generate_xml(xml, "body")
        if not args.quiet:
            print(xml)

    model = mj.MjModel.from_xml_string(str(xml))
    data = mj.MjData(model)
    mj.mj_step(model, data)

    # ==============
    # Demo
    # ==============
    if args.no_demo:
        exit()

    for eye in eyes.values():
        eye.reset(model, data)

    # Quick test to make sure the renderer is working
    # TODO: The axes images and subplots are in weird spots. not sure why i need to flip
    import time
    import matplotlib.pyplot as plt

    times = []
    images = []
    start = time.time()
    for i in range(X_NUM):
        images.append([])
        for j in range(Y_NUM):
            eye = eyes[f"eye_{i}_{j}"]
            t0 = time.time()
            image = eye.step()
            t1 = time.time()
            times.append(t1 - t0)

            images[i].append(image)

    print("Total time (including setup/plotting/etc):", time.time() - start)
    print("Total time (minus setup/plotting/etc):", sum(times))
    print("Median time:", np.median(times))
    print("Max time:", np.max(times))
    print("Min time:", np.min(times))

    images = np.array(images)
    composite_image = np.vstack([np.hstack(image_row) for image_row in reversed(images)])

    if args.plot or args.save:
        plt.title(args.title)
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.imshow(composite_image)

    if args.save:
        # save the figure without the frame
        plt.axis("off")
        plt.savefig(f"{args.title}.png", bbox_inches="tight", dpi=300)

    if args.plot:
        fig_manager = plt.get_current_fig_manager()
        fig_manager.full_screen_toggle()
        plt.show()
