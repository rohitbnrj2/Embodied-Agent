from typing import Tuple, Dict, Optional, Any, Iterable
import time
import numpy as np

import cv2
import mujoco as mj
from gymnasium import spaces

from cambrian.renderer import MjCambrianRenderer, MjCambrianRendererConfig
from cambrian.optics import MjCambrianOptics, MjCambrianOpticsConfig
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.cambrian_xml import MjCambrianXML


@config_wrapper
class MjCambrianEyeConfig(MjCambrianBaseConfig):
    """Defines the config for an eye. Used for type hinting.

    Attributes:
        pos (Optional[Tuple[float, float, float]]): The initial position of the camera.
            This is computed by the animal from the coord during placement. Fmt: xyz
        quat (Optional[Tuple[float, float, float, float]]): The initial rotation of the
            camera. This is computed by the animal from the coord during placement.
            Fmt: wxyz.
        fov (Tuple[float, float]): Independent of the `fovy` field in the MJCF
            xml. Used to calculate the sensorsize field. Specified in degrees. Mutually
            exclusive with `fovy`. If `focal` is unset, it is set to 1, 1. Will override
            `sensorsize`, if set. Fmt: fovx fovy.
        focal (Tuple[float, float]): The focal length of the camera.
            Fmt: focal_x focal_y.
        resolution (Tuple[int, int]): The width and height of the rendered image.
            Fmt: width height.

        coord (Tuple[float, float]): The x and y coordinates of the eye.
            This is used to determine the placement of the eye on the animal.
            Specified in degrees. This attr isn't actually used by eye, but by the
            animal. The eye has no knowledge of the geometry it's trying to be placed
            on. Fmt: lat lon

        optics (Optional[MjCambrianOpticsConfig]): The optics config to use for the eye.
            Optics is disabled if this is unset.

        renderer (MjCambrianRendererConfig): The renderer config to use for the
            underlying renderer. The width and height of the renderer will be set to the
            padded resolution (resolution + int(psf_filter_size/2)) of the eye.
    """

    pos: Optional[Tuple[float, float, float]] = None
    quat: Optional[Tuple[float, float, float, float]] = None
    fov: Tuple[float, float]
    focal: Tuple[float, float]
    resolution: Tuple[int, int]

    coord: Tuple[float, float]

    optics: Optional[MjCambrianOpticsConfig] = None

    renderer: MjCambrianRendererConfig


class MjCambrianEye:
    """Defines an eye for the cambrian environment. It essentially wraps a mujoco Camera
    object and provides some helper methods for rendering and generating the XML. The
    eye is attached to the parent body such that movement of the parent body will move
    the eye.

    Args:
        config (MjCambrianEyeConfig): The configuration for the eye.
        name (str): The name of the eye.
    """

    def __init__(self, config: MjCambrianEyeConfig, name: str):
        self.config = config
        self.name = name

        self._model: mj.MjModel = None
        self._data: mj.MjData = None
        self._prev_obs: np.ndarray = None

        self._optics: MjCambrianOptics = None
        if self.config.optics is not None:
            self._optics = MjCambrianOptics(self.config.optics)
            assert (
                "depth_array" in self.config.renderer.render_modes
            ), "Must specify 'depth_array' in the render modes for the renderer config."

            # If optics is enabled, update the resolution to the padded resolution
            self.config.resolution = self.config.optics.padded_resolution

        assert (
            "rgb_array" in self.config.renderer.render_modes
        ), "Must specify 'rgb_array' in the render modes for the renderer config."
        self._renderer = MjCambrianRenderer(config.renderer)

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
        assert parent is not None, f"Could not find parent for '{parent_body_name}'"

        # Calculate the mujoco parameters for the camera
        fovx, fovy = self.config.fov
        focalx, focaly = self.config.focal
        sensorsize = [
            float(2 * focalx * np.tan(np.radians(fovx) / 2)),
            float(2 * focaly * np.tan(np.radians(fovy) / 2)),
        ]

        # Finally add the camera element at the end
        xml.add(
            parent,
            "camera",
            name=self.config.name,
            mode="fixed",
            pos=" ".join(map(str, self.config.pos)),
            quat=" ".join(map(str, self.config.quat)),
            resolution=" ".join(map(str, self.config.resolution)),
            focal=" ".join(map(str, self.config.focal)),
            sensorsize=" ".join(map(str, sensorsize)),
        )

        return xml

    def reset(self, model: mj.MjModel, data: mj.MjData):
        """Sets up the camera for rendering. This should be called before rendering
        the first time."""
        self._model = model
        self._data = data

        self._renderer.reset(model, data, *self.config.resolution)
        if self._optics is not None:
            self._optics.reset(model)

        # We'll hide all the geomgroups and sitegroups after 2
        # We can then put stuff at groups > 2 and it will be hidden to the animal
        self._renderer.set_option("geomgroup", False, slice(2, None))
        self._renderer.set_option("sitegroup", False, slice(2, None))

        return self.step()

    def step(self) -> np.ndarray:
        """Simply calls `render` and sets the last observation.
        See `render()` for more information."""
        obs = self.render()
        self._prev_obs = obs.copy()
        return obs

    def render(self) -> np.ndarray:
        """Render the image from the camera. If `return_depth` is True, returns the
        both depth image and rgb, otherwise only rgb.

        NOTE: The actual rendered resolution is `self.sensor_resolution`. This is so
        that the convolution with the psf filter doesn't cut off any of the image. After
        the psf is applied, the image will be cropped to `self.resolution`.
        """

        rgb = self._renderer.render()

        if self._optics is not None:
            rgb, depth = rgb
            rgb = rgb.astype(np.float32) / 255.0
            rgb, _ = self._optics.forward(rgb, depth)
        else:
            rgb = rgb.astype(np.float32) / 255.0

        return self._postprocess(rgb)

    def _postprocess(self, image: np.ndarray) -> np.ndarray:
        """Downsamples image and normalizes it to [0, 1].
        image: (H, W, 3) float32 array in [0, 1]
        """
        # 1. Apply animal angular resolution (downsample the image)
        # 2. crop the imaging plane to the eye resolution
        if self._optics is not None:
            image = self._crop(image)
            return np.clip(image, 0, 1).astype(np.float32)
        return image

    def _crop(self, image: np.ndarray) -> np.ndarray:
        """Crop the image to the resolution specified in the config."""
        resolution = [self.resolution[1], self.resolution[0]]
        cw, ch = int(np.ceil(resolution[0] / 2)), int(np.ceil(resolution[1] / 2))
        ox, oy = 1 if resolution[0] % 2 == 1 else 0, 1 if resolution[1] % 2 == 1 else 0
        bl = (image.shape[0] // 2 - cw + ox, image.shape[1] // 2 - ch + oy)
        tr = (image.shape[0] // 2 + cw, image.shape[1] // 2 + ch)
        return image[bl[0] : tr[0], bl[1] : tr[1]]

    def _downsample(self, image: np.ndarray) -> np.ndarray:
        """Downsample the image to the resolution specified in the config."""
        return cv2.resize(image, self.resolution)

    @property
    def observation_space(self) -> spaces.Box:
        """The observation space is just the rgb image.

        NOTE:
        - The input resolution in the yaml file is (W, H) but the eye output is
            (H, W, 3) so we flip the order here.
        """

        observation_space = spaces.Box(
            0.0, 1.0, shape=(*self.resolution[::-1], 3), dtype=np.float32
        )
        return observation_space

    @property
    def num_pixels(self) -> int:
        """The number of pixels in the image."""
        return np.prod(self.resolution)

    @property
    def prev_obs(self) -> np.ndarray:
        """The last observation returned by `self.render()`."""
        return self._prev_obs


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
                    fov=[45, 20],
                    psf_filter_size=[0, 0],
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
    composite_image = np.vstack(
        [np.hstack(image_row) for image_row in reversed(images)]
    )

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
