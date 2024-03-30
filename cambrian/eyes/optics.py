from typing import Tuple, List, Dict

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

        depths (List[float]): Depths at which the PSF is calculated.

        eye_resolution (Tuple[int, int]): Resolution of the eye sensor.
        eye_sensorsize (Tuple[float, float]): Size of the eye sensor.
    """

    padded_resolution: Tuple[int, int]

    aperture: float
    noise_std: float
    wavelengths: Tuple[float, float, float]

    depths: List[float]

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

        self._psfs: Dict[torch.Tensor, torch.Tensor] = {}
        self._depths = torch.tensor(self.config.depths).to(self._device)
        self._calculate_psfs()

    def _calculate_psfs(self):
        # Initialize parameters used during the PSF calculation

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
        fy = torch.linspace(-1.0 / (2.0 * dy), 1.0 / (2.0 * dy), My)
        FX, FY = torch.meshgrid(fx, fy, indexing="xy")

        # Aperture
        # Add small epsilon to avoid division by zero
        # TODO: aperture > 1.0
        aperture_radius = min(Lx / 2, Ly / 2) * self.config.aperture + 1.0e-7  # (m)
        A: torch.Tensor = (
            torch.sqrt(X1.square() + Y1.square()) / aperture_radius <= 1.0
        ).float()

        # Pre-compute some values that are reused in the PSF calculation
        wavelengths = torch.tensor(self.config.wavelengths).reshape(-1, 1, 1)
        X1_Y1 = X1.square() + Y1.square()
        H_valid = (torch.sqrt(FX.square() + FY.square()) < (1.0 / wavelengths)).float()
        FX_FY = torch.sqrt(
            1 - (wavelengths * FX).square() - (wavelengths * FY).square()
        )
        k = 1j * 2 * torch.pi / wavelengths

        def _calculate_psf(depth: torch.Tensor):
            # electric field originating from point source
            u1 = torch.exp(k * torch.sqrt(X1_Y1 + depth.square()))

            # electric field at the aperture
            u2 = torch.mul(u1, A)

            # electric field at the sensor plane
            H = torch.mul(H_valid, torch.nan_to_num(torch.exp(k * depth * FX_FY)))

            # Calculate the sqrt of the PSF
            u2_fft = torch.fft.fft2(torch.fft.fftshift(u2))
            H_u2_fft = torch.mul(torch.fft.fftshift(H), u2_fft)
            u3: torch.Tensor = torch.fft.ifftshift(torch.fft.ifft2(H_u2_fft))

            # psf should sum to 1 because of energy
            psf = u3.abs().square()
            psf /= psf.sum() + 1.0e-7

            return psf

        for depth in self._depths:
            self._psfs[depth.item()] = _calculate_psf(depth.cpu()).to(self._device)

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

        # Add noise to the image
        image = self._apply_noise(image, self.config.noise_std)

        # Apply the depth invariant PSF
        psf = self._get_psf(torch.mean(depth))

        # Image may be batched in the form
        unbatched = len(image.shape) == 3
        if unbatched:
            image = image.permute(2, 0, 1).unsqueeze(0)
        psf = psf.unsqueeze(1)
        image = torch.nn.functional.conv2d(image, psf, padding="same", groups=3)

        # Post-process the image
        if unbatched:
            image = image.squeeze(0).permute(1, 2, 0)

        image = self._crop(image)
        image = torch.clip(image, 0, 1)

        if return_np:
            return image.cpu().numpy()
        return image

    def _apply_noise(self, image: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise to the image."""
        if std == 0.0:
            return image

        noise = torch.normal(mean=0.0, std=std, size=image.shape, device=self._device)
        return torch.clamp(image + noise, 0, 1)

    def _get_psf(self, depth: torch.Tensor) -> torch.Tensor:
        """This will retrieve the psf with the closest depth to the specified depth."""
        closest_depth = self._depths[torch.argmin(torch.abs(depth - self._depths))]
        return self._psfs[closest_depth.item()]

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
