from typing import Tuple, Dict, Optional, Any, Iterable
import time
import numpy as np

import cv2
import mujoco as mj
from gymnasium import spaces

from cambrian.renderer import MjCambrianRenderer
from cambrian.optics import MjCambrianNonDifferentiableOptics
from cambrian.utils.base_config import config_wrapper, MjCambrianBaseConfig
from cambrian.utils.cambrian_xml import MjCambrianXML


@config_wrapper
class MjCambrianEyeConfig(MjCambrianBaseConfig):
    """Defines the config for an eye. Used for type hinting.

    Attributes:
        name (Optional[str]): Placeholder for the name of the eye. If set, used
            directly. If unset, the name is set to `{animal.name}_eye_{eye_index}`.

        mode (str): The mode of the camera. Should always be "fixed". See the mujoco
            documentation for more info.
        resolution (Tuple[int, int]): The width and height of the rendered image.
            Fmt: width height.
        fov (Tuple[float, float]): Independent of the `fovy` field in the MJCF
            xml. Used to calculate the sensorsize field. Specified in degrees. Mutually
            exclusive with `fovy`. If `focal` is unset, it is set to 1, 1. Will override
            `sensorsize`, if set. Fmt: fovx fovy.

        enable_optics (bool): Whether to enable optics or not.
        enable_aperture (bool): Whether to enable the aperture or not.
        enable_lens (bool): Whether to enable the lens or not.
        enable_phase_mask (bool): Whether to enable the phase mask or not.

        scene_angular_resolution: The angular resolution of the scene. This is used to
            determine the field of view of the scene. Specified in degrees.
        pixel_size: The pixel size of the sensor in meters.
        sensor_resolution (Tuple[int, int]): TODO
        add_noise (bool): TODO
        noise_std (float): TODO
        aperture_open (float): The aperture open value. This is the radius of the
            aperture. The aperture is a circle that is used to determine which light
            rays to let through. Only used if `enable_aperture` is True. Must be
            between 0 and 1.
        aperture_radius (float): The aperture radius value.
        wavelengths (Tuple[float, float, float]): The wavelengths to use for the
            intensity sensor. Fmt: wavelength_1 wavelength_2 wavelength_3
        depth_bins (int): The number of depth bins to use for the depth dependent psf.

        load_height_mask_from_file (bool): Whether to load the height mask from file or
            not. If True, the height mask will be loaded from the file specified in
            `height_mask_from_file`. If False, the psf wil be randomized or set to zeros
            using `randomize_psf_init`.
        height_mask_from_file (Optional[str]): The path to the height mask file to load.
        randomize_psf_init (bool): Whether to randomize the psf or not. If True, the psf
            will be randomized. If False, the psf will be set to zeros. Only used if
            `load_height_mask_from_file` is False.
        zernike_basis_path (Optional[str]): The path to the zernike basis file to load.
        psf_filter_size (Tuple[int, int]): The psf filter size. This is
            convolved across the image, so the actual resolution of the image is plus
            psf_filter_size / 2. Only used if `load_height_mask_from_file` is False.
            Otherwise the psf filter size is determined by the height mask.
        refractive_index (float): The refractive index of the eye.
        min_phi_defocus (float): TODO
        max_phi_defocus (float): TODO

        load_height_mask_from_file (bool): Whether to load the height mask from file or
            not. If True, the height mask will be loaded from the file specified in
            `height_mask_from_file`. If False, the psf wil be randomized or set to zeros
            using `randomize_psf_init`.
        height_mask_from_file (Optional[str]): The path to the height mask file to load.
        randomize_psf_init (bool): Whether to randomize the psf or not. If True, the psf
            will be randomized. If False, the psf will be set to zeros. Only used if
            `load_height_mask_from_file` is False.
        zernike_basis_path (Optional[str]): The path to the zernike basis file to load.

        psf_filter_size (Tuple[int, int]): The psf filter size. This is
            convolved across the image, so the actual resolution of the image is plus
            psf_filter_size / 2. Only used if `load_height_mask_from_file` is False.
            Otherwise the psf filter size is determined by the height mask.
        refractive_index (float): The refractive index of the eye.
        depth_bins (int): The number of depth bins to use for the depth sensor.
        min_phi_defocus (float): The minimum depth to use for the depth sensor.
        max_phi_defocus (float): The maximum depth to use for the depth sensor.
        wavelengths (Tuple[float, float, float]): The wavelengths to use for the
            intensity sensor. Fmt: wavelength_1 wavelength_2 wavelength_3
        #### Optics Params

        pos (Optional[Tuple[float, float, float]]): The initial position of the camera.
            Fmt: xyz
        quat (Optional[Tuple[float, float, float, float]]): The initial rotation of the
            camera. Fmt: wxyz.
        fovy (Optional[float]): The vertical field of view of the camera.
        focal (Optional[Tuple[float, float]]): The focal length of the camera.
            Fmt: focal_x focal_y.
        sensorsize (Optional[Tuple[float, float]]): The sensor size of the camera.
            Fmt: sensor_x sensor_y.

        coord (Tuple[float, float]): The x and y coordinates of the eye.
            This is used to determine the placement of the eye on the animal.
            Specified in degrees. Mutually exclusive with `pos` and `quat`. This attr
            isn't actually used by eye, but by the animal. The eye has no knowledge
            of the geometry it's trying to be placed on. Fmt: lat lon

        renderer (MjCambrianRendererConfig): The renderer config to use for the
            underlying renderer. The width and height of the renderer will be set to the
            padded resolution (resolution + int(psf_filter_size/2)) of the eye.
    """

    name: Optional[str] = None

    mode: str
    resolution: Tuple[int, int]
    fov: Tuple[float, float]

    enable_optics: bool
    enable_aperture: bool
    enable_lens: bool
    enable_phase_mask: bool

    scene_resolution: Tuple[int, int]
    scene_angular_resolution: float
    pixel_size: float
    sensor_resolution: Tuple[int, int]
    add_noise: bool
    noise_std: float
    aperture_open: float
    aperture_radius: float
    wavelengths: Tuple[float, float, float]
    depth_bins: int

    load_height_mask_from_file: bool
    height_mask_from_file: Optional[str] = None
    randomize_psf_init: bool
    zernike_basis_path: Optional[str] = None
    psf_filter_size: Tuple[int, int]
    refractive_index: float
    min_phi_defocus: float
    max_phi_defocus: float

    pos: Optional[Tuple[float, float, float]] = None
    quat: Optional[Tuple[float, float, float, float]] = None
    fovy: Optional[float] = None
    focal: Optional[Tuple[float, float]] = None
    sensorsize: Optional[Tuple[float, float]] = None

    coord: Tuple[float, float]

    renderer: MjCambrianRendererConfig

    def to_xml_kwargs(self) -> Dict[str, Any]:
        kwargs = dict()

        def set_if_not_none(key: str, val: Any):
            if val is not None:
                if isinstance(val, Iterable) and not isinstance(val, str):
                    val = " ".join(map(str, val))
                kwargs[key] = val

        set_if_not_none("name", self.name)
        set_if_not_none("mode", self.mode)
        set_if_not_none("pos", self.pos)
        set_if_not_none("quat", self.quat)
        set_if_not_none("resolution", self.resolution)
        set_if_not_none("fovy", self.fovy)
        set_if_not_none("focal", self.focal)
        set_if_not_none("sensorsize", self.sensorsize)

        return kwargs



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

        self._optics: MjCambrianNonDifferentiableOptics = None
        if self.config.enable_optics:
            self._optics = MjCambrianNonDifferentiableOptics(self.config)

        self._renderer = MjCambrianRenderer(config.renderer_config)

    def _check_config(self, config: MjCambrianEyeConfig) -> MjCambrianEyeConfig:
        """This will automatically set some of the config values if they are not
        specified.

        psf_filter_size: set to [0, 0] if not specified
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
            "rgb_array" or "depth_array" in config.renderer_config.render_modes
        ), "Must specify 'rgb_array' or 'depth_array' in the render modes for the renderer config."

        self._render_depth = "depth_array" in config.renderer_config.render_modes
        if config.enable_optics:
            assert self._render_depth, "Must render depth if optics is enabled"

        if config.fovy is not None:
            assert 0 < config.fovy < 180, f"Invalid fovy: {config.fovy=}"
            assert config.focal is None, "Cannot set both fovy and focal"
            assert config.sensorsize is None, "Cannot set both fovy and sensorsize"
            assert config.fov is None, "Cannot set both fovy and fov"

        if config.fov is not None:
            assert (
                0 < config.fov[0] < 180 and 0 < config.fov[1] < 180
            ), f"Invalid fov: {config.fov=}."
            assert config.fovy is None, "Cannot set both fov and fovy"
            assert config.sensorsize is None, "Cannot set both fov and sensorsize"

            # the rendering resolution must be set wrt the scene resoution not the eye resolution
            if self._render_depth:
                # if optics is enabled, then render at the scene resolution
                sensor_fov = [
                    config.scene_resolution[1] * config.scene_angular_resolution,
                    config.scene_resolution[0] * config.scene_angular_resolution,
                ]
                config.setdefault("focal", [0.1, 0.1])
                fovx, fovy = sensor_fov
                focalx, focaly = config.focal
                config.sensorsize = [
                    float(2 * focalx * np.tan(np.radians(fovx) / 2)),
                    float(2 * focaly * np.tan(np.radians(fovy) / 2)),
                ]
                config.pixel_size = config.sensorsize[0] / config.scene_resolution[0]
                # if config.pixel_size > 1e-3:
                #     print(f"Warning: Pixel size {config.pixel_size} m > 0.001m. Required Scene Resolution: {config.sensorsize[0]/1e-3} for input fov and  sensorsize.")

                config.renderer_config.width = config.scene_resolution[1]
                config.renderer_config.height = config.scene_resolution[0]
                config.fov = [fovx, fovy]
            else:
                # if optics is not enabled, then render at the eye resolution
                config.setdefault("focal", [0.1, 0.1])
                fovx, fovy = config.fov
                focalx, focaly = config.focal
                config.sensorsize = [
                    float(2 * focalx * np.tan(np.radians(fovx) / 2)),
                    float(2 * focaly * np.tan(np.radians(fovy) / 2)),
                ]
                config.renderer_config.width = config.resolution[0]
                config.renderer_config.height = config.resolution[1]

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

        if self._render_depth:
            self.config.renderer_config.width = self.config.scene_resolution[1]
            self.config.renderer_config.height = self.config.scene_resolution[0]
        else:
            self.config.renderer_config.width = self.config.resolution[0]
            self.config.renderer_config.height = self.config.resolution[1]

        self._renderer.reset(model, data)
        if self.config.enable_optics:
            self._optics.reset(config=self.config)

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

        NOTE: The actual rendered resolution is `self.sensor_resolution`. This is so
        that the convolution with the psf filter doesn't cut off any of the image. After
        the psf is applied, the image will be cropped to `self.resolution`.
        """

        rgb = self._renderer.render()

        if self.config.enable_optics:
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
        if self._render_depth:
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
    def name(self) -> str:
        return self.config.name

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

    @property
    def num_pixels(self) -> int:
        """The number of pixels in the image."""
        return self.resolution[0] * self.resolution[1]

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
