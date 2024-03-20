from typing import Tuple, Optional

import torch
import numpy as np
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
        noise_std (float): Standard deviation of the Gaussian noise to be
            added to the image. If 0.0, no noise is added.
        wavelengths (Tuple[float, float, float]): Wavelengths of the RGB channels.
    """

    padded_resolution: Tuple[int, int]

    aperture: float
    noise_std: float
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

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self, model: mj.MjModel):
        self._fixedcamid = get_camera_id(model, self._name)
        self._reset_psf(model)

    def _reset_psf(self, model: mj.MjModel):
        """Define a simple point spread function (PSF) for the eye."""

        focal = model.cam_intrinsic[self._fixedcamid][:2]

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
        X1, Y1 = torch.meshgrid(x1, y1, indexing="ij")

        # Frequency coords
        fx = torch.linspace(-1.0 / (2.0 * dx), 1.0 / (2.0 * dx), Mx)
        fy = torch.linspace(-1.0 / (2.0 * dx), 1.0 / (2.0 * dx), My)
        FX, FY = torch.meshgrid(fx, fy, indexing="ij")

        # Aperture
        # Add small epsilon to avoid division by zero
        # TODO: aperture > 1.0
        aperture_radius = min(Lx / 2, Ly / 2) * self.config.aperture + 1.0e-7  # (m)
        A: torch.Tensor = (
            torch.sqrt(X1**2 + Y1**2) / aperture_radius <= 1.0
        ).float()

        # Update the class variables
        self.A = A.to(self._device)
        self.X1 = X1.to(self._device)
        self.Y1 = Y1.to(self._device)
        self.FX = FX.to(self._device)
        self.FY = FY.to(self._device)
        self.focal = focal

        self.wavelengths = torch.tensor(self.config.wavelengths).to(self._device)
        self.wavelengths = self.wavelengths.unsqueeze(-1).unsqueeze(-1)

        # Precompute some values used in the PSF calculation
        self.X1_Y1 = self.X1.square() + self.Y1.square()
        self.H_valid = (
            torch.sqrt(self.FX.square() + self.FY.square()) < (1.0 / self.wavelengths)
        ).float()
        self.FX_FY = torch.sqrt(
            1
            - (self.wavelengths * self.FX).square()
            - (self.wavelengths * self.FY).square()
        )
        self.k = 1j * 2 * torch.pi / self.wavelengths

        # Create buffers for later use in the psf calculation
        shape = (len(self.wavelengths), *self.X1.shape)
        self.u1 = torch.empty(shape, dtype=torch.complex64).to(self._device)
        self.u2 = torch.empty_like(self.u1)
        self.H = torch.empty_like(self.u1)
        self.H_temp = torch.empty_like(self.u1)
        self.H_fft = torch.empty_like(self.u1)
        self.u2_fft = torch.empty_like(self.u1)
        self.H_u2_fft = torch.empty_like(self.u1)
        self.u3 = torch.empty_like(self.u1)
        self.psf = torch.empty_like(self.u1)

    def forward(
        self, image: torch.Tensor | np.ndarray, depth: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Apply the depth invariant PSF to the image."""
        return_np = False
        if isinstance(image, np.ndarray):
            # pytorch doesn't support negative strides, so copy if there are any
            image = image.copy() if any(s < 0 for s in image.strides) else image
            depth = depth.copy() if any(s < 0 for s in depth.strides) else depth

            image = torch.from_numpy(image).to(self._device)
            depth = torch.from_numpy(depth).to(self._device)
            return_np = True
        else:
            assert (
                image.device == self._device
            ), f"Device mismatch: {image.device=}, {self._device=}"

        # Add noise to the image if specified
        if self.config.noise_std != 0.0:
            image = self._apply_noise(image, self.config.noise_std)

        # Apply the depth invariant PSF
        psf = self._calc_depth_invariant_psf(torch.mean(depth))
        image = image.permute(2, 0, 1).unsqueeze(0)
        psf = psf.unsqueeze(1)
        image = torch.nn.functional.conv2d(image, psf, padding="same", groups=3)

        # Post-process the image
        image = image.squeeze(0).permute(1, 2, 0)
        image = self._crop(image)
        image = torch.clip(image, 0, 1)

        if return_np:
            return image.cpu().numpy()
        return image

    def _apply_noise(self, image: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise to the image."""
        noise = torch.normal(mean=0.0, std=std, size=image.shape, device=self._device)
        return torch.clamp(image + noise, 0, 1)

    def _calc_depth_invariant_psf(self, mean_depth: torch.Tensor) -> torch.Tensor:
        """Calculate the depth invariant PSF.

        Preforms batched calculation of the PSF for each wavelength.
        """
        # electric field originating from point source
        torch.exp(self.k * torch.sqrt(self.X1_Y1 + mean_depth.square()), out=self.u1)

        # electric field at the aperture
        torch.mul(self.u1, self.A, out=self.u2)

        # electric field at the sensor plane
        torch.exp(self.k * mean_depth * self.FX_FY, out=self.H_temp)
        torch.nan_to_num(self.H_temp, out=self.H_temp)
        torch.mul(self.H_valid, self.H_temp, out=self.H)

        # NOTE: not using out=... because it was found to be slower
        self.H_fft = torch.fft.fft2(self.H)
        self.u2_fft = torch.fft.fft2(self.u2)
        torch.mul(self.H_fft, self.u2_fft, out=self.H_u2_fft)
        self.u3: torch.Tensor = torch.fft.ifft2(self.H_u2_fft)

        # psf should sum to 1 because of energy
        torch.square(self.u3, out=self.psf)
        self.psf /= self.psf.sum() + 1.0e-7

        return self.psf.real

    def _crop(self, image: torch.Tensor) -> torch.Tensor:
        """Crop the image to the resolution specified in the config. This crops the
        center part of the image."""
        width, height = image.shape[0], image.shape[1]
        target_width, target_height = self._resolution
        top = (height - target_height) // 2
        left = (width - target_width) // 2
        return image[top : top + target_height, left : left + target_width]
