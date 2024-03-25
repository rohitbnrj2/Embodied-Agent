from typing import Tuple

import torch
import numpy as np

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

    # These should be copies of the attributes from the eye config
    eye_resolution: Tuple[int, int]
    eye_sensorsize: Tuple[float, float]


class MjCambrianOptics:
    """This class applies the depth invariant PSF to the image.

    Args:
        config (MjCambrianOpticsConfig): Config for the optics module.
    """

    def __init__(self, config: MjCambrianOpticsConfig):
        self.config = config

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._reset_psf()

    def _reset_psf(self):
        """Define a simple point spread function (PSF) for the eye."""

        # Mx,My defines the number of pixels in x,y direction (i.e. width, height)
        Mx, My = self.config.padded_resolution
        assert Mx > 2 and My > 2, f"Sensor resolution must be > 2: {Mx=}, {My=}"
        assert Mx % 2 and My % 2, f"Sensor resolution must be odd: {Mx=}, {My=}"

        # dx/dy defines the pixel pitch (m) (i.e. distance between the centers of
        # adjacent pixels) of the sensor
        sx, sy = self.config.eye_sensorsize
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

    def step(
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
                image.device.type == self._device.type
            ), f"Device mismatch: {image.device=}, {self._device=}"

        # Add noise to the image if specified
        if self.config.noise_std != 0.0:
            image = self._apply_noise(image, self.config.noise_std)

        # Apply the depth invariant PSF
        psf = self._calc_depth_invariant_psf(torch.mean(depth))
        # Image may be batched in the form
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
        psf = psf.unsqueeze(1)
        image = torch.nn.functional.conv2d(image, psf, padding="same", groups=3)

        # Post-process the image
        if len(image.shape) == 3:
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
        """Crop the image to the resolution specified in the config. This method supports
        input shapes [W, H, 3] and [B, 3, W, H]. It crops the center part of the image.
        """
        if image.dim() == 4:  # [B, 3, W, H]
            _, _, width, height = image.shape
            target_width, target_height = self.config.eye_resolution
            top = (height - target_height) // 2
            left = (width - target_width) // 2
            return image[:, :, top : top + target_height, left : left + target_width]
        elif image.dim() == 3:  # [W, H, 3]
            width, height, _ = image.shape
            target_width, target_height = self.config.eye_resolution
            top = (height - target_height) // 2
            left = (width - target_width) // 2
            return image[top : top + target_height, left : left + target_width, :]
        else:
            raise ValueError("Unsupported image shape.")
