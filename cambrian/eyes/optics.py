"""This is an optics-enabled eye, which implements a height map and a PSF on top
of the existing eye."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Self, Tuple

import torch
from hydra_config import HydraContainerConfig, config_wrapper

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.utils import make_odd, device, RenderFrame
from cambrian.renderer.render_utils import resize_with_aspect_fill, add_text


@config_wrapper
class MjCambrianApertureConfig(HydraContainerConfig, ABC):
    @abstractmethod
    def calculate_aperture_mask(
        self, X1_Y1: torch.Tensor, Lx: float, Ly: float
    ) -> torch.Tensor:
        """This method calculates the aperture mask.

        Args:
            X1_Y1 (torch.Tensor): Squared distance from the center of the aperture.
            Lx (float): Width of the aperture.
            Ly (float): Height of the aperture.

        Returns:
            torch.Tensor: Aperture mask.
        """
        pass


@config_wrapper
class MjCambrianCircularApertureConfig(MjCambrianApertureConfig):
    """This defines the config for the circular aperture. This extends the base aperture
    config and adds additional parameters for the circular aperture.

    Attributes:
        radius (float): Radius of the circular aperture.
    """

    radius: float

    def calculate_aperture_mask(
        self, X1_Y1: torch.Tensor, Lx: float, Ly: float
    ) -> torch.Tensor:
        aperture_radius = min(Lx / 2, Ly / 2) * self.radius + 1e-7
        return torch.nan_to_num(torch.sqrt(X1_Y1) / aperture_radius) <= 1.0


@config_wrapper
class MjCambrianMaskApertureConfig(MjCambrianApertureConfig):
    """This defines the config for the custom aperture. This extends the base aperture
    config and adds additional parameters for the custom aperture.

    Attributes:
        mask (Optional[List[List[int]]]): Aperture mask. This is a 2D array that defines
            the aperture mask. The aperture mask is a binary mask that defines the
            aperture of the lens. It's a binary mask where 1 lets light through and 0
            blocks it. The mask can only be None if randomize is True or if size is
            not None. Defaults to None.
        randomize (bool): Randomize the aperture mask. If True, the aperture mask is
            randomized.
        random_prob (Optional[float]): Probability of the aperture mask being 1. If
            None, the probability is 0.5. Defaults to None.
        size (Optional[Tuple[int, int]]): Size of the aperture mask. If None, the size 
            is the same as the pupil resolution. Defaults to None.
    """

    mask: Optional[List[List[int]]] = None
    randomize: bool
    random_prob: Optional[float] = None
    size: Optional[Tuple[int, int]] = None

    def calculate_aperture_mask(self, X1_Y1: torch.Tensor, *_) -> torch.Tensor:
        size = self.size if self.size is not None else X1_Y1.shape
        if self.mask is None:
            assert self.randomize or self.size is not None, "Mask or size must be set."
            mask = torch.randint(0, 2, size, dtype=torch.float32)
            if self.random_prob is not None:
                mask = mask > self.random_prob
        else:
            mask = torch.tensor(self.mask, dtype=torch.float32)
        assert mask.shape[0] == mask.shape[1]
        if self.randomize:
            mask = torch.randint(0, 2, mask.shape, dtype=torch.float32)

        mask = (
            torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=size,
                mode="bicubic",
            )
            .squeeze(0)
            .squeeze(0)
        )
        return mask > 0.5


@config_wrapper
class MjCambrianOpticsEyeConfig(MjCambrianEyeConfig):
    """This defines the config for the optics module. This extends the base eye config
    and adds additional parameters for the optics module.

    Attributes:
        pupil_resolution (Tuple[int, int]): Resolution of the pupil plane. This
            is used to calculate the PSF.

        noise_std (float): Standard deviation of the Gaussian noise to be
            added to the image. If 0.0, no noise is added.
        wavelengths (Tuple[float, float, float]): Wavelengths of the RGB channels.

        f_stop (float): F-stop of the lens. This is used to calculate the PSF.
        refractive_index (float): Refractive index of the lens material.
        height_map (List[float]): Height map of the lens. This is used to
            calculate the phase shift of the light passing through the lens. Uses a
            radially symmetric approximation.
        scale_intensity (bool): Whether to scale the intensity of the PSF by the
            overall throughput of the aperture.

        aperture (MjCambrianApertureConfig): Aperture config. This defines the
            aperture of the lens. The aperture can be circular or custom.

        depths (List[float]): Depths at which the PSF is calculated. If empty, the psf
            is calculated for each render call; otherwise, the PSFs are precomputed.
    """

    instance: Callable[[Self, str], "MjCambrianOpticsEye"]

    pupil_resolution: Tuple[int, int]

    noise_std: float
    wavelengths: Tuple[float, float, float]

    f_stop: float
    refractive_index: float
    height_map: List[float]
    scale_intensity: bool

    aperture: MjCambrianApertureConfig

    depths: List[float]


class MjCambrianOpticsEye(MjCambrianEye):
    """This class applies the depth invariant PSF to the image.

    Args:
        config (MjCambrianOpticsConfig): Config for the optics module.
    """

    def __init__(self, config: MjCambrianOpticsEyeConfig, name: str):
        super().__init__(config, name)
        self._config: MjCambrianOpticsEyeConfig

        self._renders_depth = "depth_array" in self._config.renderer.render_modes
        assert self._renders_depth, "Eye: 'depth_array' must be a render mode."

        self._psfs: Dict[torch.Tensor, torch.Tensor] = {}
        self._depths = torch.tensor(self._config.depths).to(device)
        self.initialize()

    def initialize(self):
        """This will initialize the parameters used during the PSF calculation."""
        # pupil_Mx,pupil_My defines the number of pixels in x,y direction
        # (i.e. width, height) of the pupil
        pupil_Mx, pupil_My = torch.tensor(self._config.pupil_resolution)
        assert (
            pupil_Mx > 2 and pupil_My > 2
        ), f"Pupil resolution must be > 2: {pupil_Mx=}, {pupil_My=}"
        assert (
            pupil_Mx % 2 and pupil_My % 2
        ), f"Pupil resolution must be odd: {pupil_Mx=}, {pupil_My=}"

        # pupil_dx/pupil_dy defines the pixel pitch (m) (i.e. distance between the
        # centers of adjacent pixels) of the pupil and Lx/Ly defines the size of the
        # pupil plane
        fx, fy = self._config.focal
        Lx, Ly = fx / self._config.f_stop, fy / self._config.f_stop
        pupil_dx, pupil_dy = Lx / pupil_Mx, Ly / pupil_My

        # Image plane coords
        # TODO: fragile to floating point errors, must use double here. okay to convert
        # to float after psf operations
        x1 = torch.linspace(-Lx / 2.0, Lx / 2.0, pupil_Mx).double()
        y1 = torch.linspace(-Ly / 2.0, Ly / 2.0, pupil_My).double()
        X1, Y1 = torch.meshgrid(x1, y1, indexing="ij")
        X1_Y1 = X1.square() + Y1.square()

        # Frequency coords
        freqx = torch.linspace(
            -1.0 / (2.0 * pupil_dx), 1.0 / (2.0 * pupil_dx), pupil_Mx
        )
        freqy = torch.linspace(
            -1.0 / (2.0 * pupil_dy), 1.0 / (2.0 * pupil_dy), pupil_My
        )
        FX, FY = torch.meshgrid(freqx, freqy, indexing="ij")

        # Aperture mask
        A = self._config.aperture.calculate_aperture_mask(X1_Y1, Lx, Ly)

        # Going to scale the intensity by the overall throughput of the aperture
        self._scaling_intensity = (A.sum() / (max(pupil_Mx * pupil_My, 1))) ** 2

        # Calculate the wave number
        wavelengths = torch.tensor(self._config.wavelengths).reshape(-1, 1, 1)
        k = 1j * 2 * torch.pi / wavelengths

        # Calculate the pupil from the height map
        # NOTE: Have to convert to numpy then to tensor to avoid issues with
        # MjCambrianConfigContainer
        maxr = torch.sqrt((pupil_Mx / 2).square() + (pupil_My / 2).square())
        h_r = torch.zeros(torch.ceil(maxr).to(int)) + 0.5
        x, y = torch.meshgrid(
            torch.arange(pupil_Mx), torch.arange(pupil_My), indexing="ij"
        )
        r = torch.sqrt((x - pupil_Mx / 2).square() + (y - pupil_My / 2).square())
        height_map: torch.Tensor = h_r[r.to(torch.int64)]  # (n, n)
        height_map *= torch.max(wavelengths / (self._config.refractive_index - 1.0))
        phi_m = k * (self._config.refractive_index - 1.0) * height_map
        pupil = A * torch.exp(phi_m)

        # Determine the scaled down psf size. Will resample the psf such that the conv
        # is faster
        sx, sy = self._config.sensorsize
        scene_dx = sx / self._config.renderer.width
        scene_dy = sy / self._config.renderer.height
        psf_resolution = (make_odd(Lx / scene_dx), make_odd(Ly / scene_dy))

        # Pre-compute some values that are reused in the PSF calculation
        H_valid = torch.sqrt(FX.square() + FY.square()) < (1.0 / wavelengths)
        # TODO: should this be fx or fy? could be different?
        FX_FY = torch.exp(
            fx
            * k
            * torch.sqrt(1 - (wavelengths * FX).square() - (wavelengths * FY).square())
        )
        H = H_valid * FX_FY

        # Now store all as class attributes
        self._X1, self._Y1 = X1.to(device), Y1.to(device)
        self._X1_Y1 = X1_Y1.to(device)
        self._H_valid = H_valid.to(device)
        self._H = H.to(device)
        self._FX, self._FY = FX.to(device), FY.to(device)
        self._FX_FY = FX_FY.to(device)
        self._k = k.to(device)
        self._A = A.to(device)
        self._pupil = pupil.to(device)
        self._height_map = height_map.to(device)
        self._psf_resolution = psf_resolution

        # Precompute the PSFs, if necessary
        if self._config.depths:
            self._precompute_psfs()

    def _precompute_psfs(self):
        """This will precompute the PSFs for all depths. This is done to avoid
        recomputing the PSF for each render call."""
        for depth in self._depths:
            self._psfs[depth.item()] = self._calculate_psf(depth).to(device)

    def _calculate_psf(self, depth: torch.Tensor):
        # electric field originating from point source
        u1 = torch.exp(self._k * torch.sqrt(self._X1_Y1 + depth.square()))

        # electric field at the aperture
        u2 = torch.mul(u1, self._pupil)

        # electric field at the sensor plane
        # Calculate the sqrt of the PSF
        u2_fft = torch.fft.fft2(torch.fft.fftshift(u2))
        H_u2_fft = torch.mul(torch.fft.fftshift(self._H), u2_fft)
        u3: torch.Tensor = torch.fft.ifftshift(torch.fft.ifft2(H_u2_fft))

        # Normalize the PSF by channel
        psf: torch.Tensor = u3.abs().square()
        psf = self._resize(psf)
        psf /= psf.sum(axis=(1, 2)).reshape(-1, 1, 1)

        # TODO: we have to do this post-calculations otherwise there are differences
        # between previous algo
        psf = psf.float()

        return psf

    def step(
        self, obs: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Overwrites the default render method to apply the depth invariant PSF to the
        image."""
        if obs is not None:
            image, depth = obs
        else:
            image, depth = self._renderer.render()

        # Calculate the depth. Remove the sky depth, which is capped at the extent
        # of the configured environment and apply a far field approximation assumption.
        depth = depth[depth < torch.max(depth)]
        depth = torch.clamp(depth, 5 * max(self.config.focal), None)
        mean_depth = torch.mean(depth)

        # Add noise to the image
        image = self._apply_noise(image, self._config.noise_std)

        # Apply the depth invariant PSF
        psf = self._get_psf(mean_depth)

        # Image may be batched in the form
        image = image.permute(2, 0, 1).unsqueeze(0)
        psf = psf.unsqueeze(1)
        image = torch.nn.functional.conv2d(image, psf, padding="same", groups=3)

        # Apply the scaling intensity ratio
        if self._config.scale_intensity:
            image *= self._scaling_intensity

        # Post-process the image
        image = image.squeeze(0).permute(1, 2, 0)
        image = self._crop(image)
        image = torch.clip(image, 0, 1)

        return super().step(obs=image)

    def _apply_noise(self, image: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise to the image."""
        if std == 0.0:
            return image

        noise = torch.normal(mean=0.0, std=std, size=image.shape, device=device)
        return torch.clamp(image + noise, 0, 1)

    def _get_psf(self, depth: torch.Tensor) -> torch.Tensor:
        """This will retrieve the psf with the closest depth to the specified depth.
        If the psfs are precomputed, this will be a simple lookup. Otherwise, the psf
        will be calculated on the fly."""
        if self._psfs:
            closest_depth = self._depths[torch.argmin(torch.abs(depth - self._depths))]
            return self._psfs[closest_depth.item()]
        else:
            return self._calculate_psf(depth)

    def _crop(self, image: torch.Tensor) -> torch.Tensor:
        """Crop the image to the resolution specified in the config. This method
        supports input shape [W, H, 3]. It crops the center part of the image.
        """
        width, height, _ = image.shape
        target_width, target_height = self._config.resolution
        top = (height - target_height) // 2
        left = (width - target_width) // 2
        return image[left : left + target_width, top : top + target_height, :]

    def _resize(self, psf: torch.Tensor) -> torch.Tensor:
        """Resize the PSF to the psf_resolution."""
        return torch.nn.functional.interpolate(
            psf.unsqueeze(0),
            size=self._psf_resolution,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
