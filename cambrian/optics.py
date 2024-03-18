from typing import Tuple, Optional

import torch
import mujoco as mj

from cambrian.utils import get_camera_id
from cambrian.utils.base_config import MjCambrianBaseConfig, config_wrapper


@config_wrapper
class MjCambrianOpticsConfig(MjCambrianBaseConfig):
    """This defines the config for the optics module.

    Attributes:
        padded_resolution (Tuple[int, int]): Resolution of the rendered image. This is
            different from the eye's resolution as it's padded such that the PSF
            can be convolved with the image and then cropped to the eye's resolution.

        aperture (float): Aperture size of the lens. This defines the radius of the
            aperture as a percentage of the sensor size.
        noise_std (Optional[float]): Standard deviation of the Gaussian noise to be
            added to the image. If None, no noise is added. Default: None.
        wavelengths (Tuple[float, float, float]): Wavelengths of the RGB channels.
    """

    padded_resolution: Tuple[int, int]

    aperture: float
    noise_std: Optional[float] = None
    wavelengths: Tuple[float, float, float]


class MjCambrianOptics(torch.nn.Module):
    """This class applies the depth invariant PSF to the image.

    Args:
        config (MjCambrianOpticsConfig): Config for the optics module.

        name (str): Name of the camera in the MJCF file. Also the name of the eye.
        resolution (Tuple[int, int]): Resolution of the eye's sensor.
    """

    def __init__(
        self, config: MjCambrianOpticsConfig, name: str, resolution: Tuple[int, int]
    ):
        self.config = config

        self._name = name
        self._resolution = resolution

    def reset(self, model: mj.MjModel):
        self._fixedcamid = get_camera_id(model, self._name)
        self._reset_psf(model)

    def _reset_psf(self, model: mj.MjModel):
        """Define a simple point spread function (PSF) for the eye."""

        focal, _ = model.cam_intrinsic[self._fixedcamid]

        # Mx,My defines the number of pixels in x,y direction (i.e. width, height)
        Mx, My = model.cam_resolution[self._fixedcamid]
        assert Mx > 2 and My > 2, f"Sensor resolution must be > 2: {Mx=}, {My=}"
        assert Mx % 2 and My % 2, f"Sensor resolution must be odd: {Mx=}, {My=}"

        # dx/dy defines the pixel pitch (m) (i.e. distance between the centers of
        # adjacent pixels) of the sensor
        sx, sy = model.cam_sensorsize[self._fixedcamid]
        dx, dy = sx / Mx, sy / My

        # Lx/Ly defines the length of the sensor plane (m)
        Lx, Ly = dx * Mx, dy * My

        # Image plane coords
        x1 = torch.linspace(-Lx / 2.0, Lx / 2.0, Mx)
        y1 = torch.linspace(-Ly / 2.0, Ly / 2.0, My)
        X1, Y1 = torch.meshgrid(x1, y1)

        # Frequency coords
        fx = torch.linspace(-1.0 / (2.0 * dx), 1.0 / (2.0 * dx), Mx)
        fy = torch.linspace(-1.0 / (2.0 * dx), 1.0 / (2.0 * dx), My)
        FX, FY = torch.meshgrid(fx, fy)

        # Aperture
        # TODO: aperture > 1.0
        aperture_radius = min(Lx / 2, Ly / 2) * self.config.aperture  # (m)
        A = (torch.sqrt(X1**2 + Y1**2) / (aperture_radius + 1.0e-7) <= 1.0).float()

        # Update the class variables
        self.A = A
        self.X1 = X1
        self.Y1 = Y1
        self.FX = FX
        self.FY = FY
        self.focal = focal

    def forward(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Apply the depth invariant PSF to the image."""
        if self.config.noise_std is not None:
            image = self._apply_noise(image, self.config.noise_std)

        psf = self._calc_depth_invariant_psf(torch.mean(depth))

        # Apply the PSF to the image
        image = torch.nn.functional.conv2d(image, psf, padding="same")

        # Post-process the image
        image = self._crop(image)
        return torch.clip(image, 0, 1)

    def _apply_noise(self, image: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise to the image."""
        noise = torch.normal(mean=0.0, std=std, size=image.shape)
        return torch.clamp(image + noise, 0, 1)

    def _calc_depth_invariant_psf(self, mean_depth: torch.Tensor) -> torch.Tensor:
        """Calculate the depth invariant PSF.

        Preforms batched calculation of the PSF for each wavelength.
        """
        wavelengths = torch.tensor(self.config.wavelengths)

        k = 2 * torch.pi / wavelengths

        # electric field originating from point source
        u1 = torch.exp(
            1j * k * torch.sqrt(self.X1**2 + self.Y1**2 + mean_depth**2)
        )

        # electric field at the aperture
        u2 = u1 * self.A

        # electric field at the sensor plane
        H_valid = (
            torch.sqrt(self.FX**2 + self.FY**2) < (1.0 / wavelengths)
        ).float()
        H = H_valid * torch.nan_to_num(
            torch.exp(
                1j
                * k
                * mean_depth
                * torch.sqrt(
                    1.0 - (wavelengths * self.FX) ** 2 - (wavelengths * self.FY) ** 2
                )
            )
        )

        u3 = torch.fft.ifftshift(
            torch.fft.ifft2(
                torch.fft.fftshift(H) @ torch.fft.fft2(torch.fft.fftshift(u2))
            )
        )

        # psf should sum to 1 because of energy
        psf = torch.abs(u3) ** 2
        psf /= torch.sum(psf) + 1.0e-7

        return psf

    def _crop(self, image: torch.Tensor) -> torch.Tensor:
        """Crop the image to the resolution specified in the config. This crops the
        center part of the image."""
        width, height = image.shape[0], image.shape[1]
        target_width, target_height = self._resolution
        top = (height - target_height) // 2
        left = (width - target_width) // 2
        return image[top : top + target_height, left : left + target_width]
