from typing import Tuple, List, Dict

import torch
import numpy as np

from cambrian.eyes import MjCambrianEyeConfig, MjCambrianEye
from cambrian.utils import make_odd
from cambrian.utils.config import config_wrapper


@config_wrapper
class MjCambrianOpticsEyeConfig(MjCambrianEyeConfig):
    """This defines the config for the optics module. This extends the base eye config
    and adds additional parameters for the optics module.

    Attributes:
        aperture (float): Aperture size of the lens. This defines the radius of the
            aperture as a percentage of the sensor size.
        noise_std (float): Standard deviation of the Gaussian noise to be
            added to the image. If 0.0, no noise is added.
        wavelengths (Tuple[float, float, float]): Wavelengths of the RGB channels.

        f_stop (float): F-stop of the lens. This is used to calculate the PSF.
        refractive_index (float): Refractive index of the lens material.
        height_map (List[float]): Height map of the lens. This is used to
            calculate the phase shift of the light passing through the lens. Uses a
            radially symmetric approximation.

        pupil_resolution (Tuple[int, int]): Resolution of the pupil plane. This
            is used to calculate the PSF.

        depths (List[float]): Depths at which the PSF is calculated. If empty, the psf
            is calculated for each render call; otherwise, the PSFs are precomputed.
    """

    pupil_resolution: Tuple[int, int]

    aperture: float
    noise_std: float
    wavelengths: Tuple[float, float, float]

    f_stop: float
    refractive_index: float
    height_map: List[float]

    depths: List[float]


class MjCambrianOpticsEye(MjCambrianEye[MjCambrianOpticsEyeConfig]):
    """This class applies the depth invariant PSF to the image.

    Args:
        config (MjCambrianOpticsConfig): Config for the optics module.
    """

    def __init__(self, config: MjCambrianOpticsEyeConfig, name: str):
        super().__init__(config, name)

        self._renders_depth = "depth_array" in self._config.renderer.render_modes
        assert self._renders_depth, "Eye: 'depth_array' must be a render mode."

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._psfs: Dict[torch.Tensor, torch.Tensor] = {}
        self._depths = torch.tensor(self._config.depths).to(self._device)
        self._initialize()
        if self._config.depths:
            self._precompute_psfs()

    def _initialize(self):
        """This will initialize the parameters used during the PSF calculation."""
        # pupil_Mx,pupil_My defines the number of pixels in x,y direction (i.e. width, height) of
        # the pupil
        pupil_Mx, pupil_My = torch.tensor(self._config.pupil_resolution)
        assert (
            pupil_Mx > 2 and pupil_My > 2
        ), f"Pupil resolution must be > 2: {pupil_Mx=}, {pupil_My=}"
        assert (
            pupil_Mx % 2 and pupil_My % 2
        ), f"Pupil resolution must be odd: {pupil_Mx=}, {pupil_My=}"

        # pupil_dx/pupil_dy defines the pixel pitch (m) (i.e. distance between the centers of
        # adjacent pixels) of the pupil and Lx/Ly defines the size of the pupil plane
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
        assert 0.0 <= self._config.aperture <= 1.0
        aperture_radius = min(Lx / 2, Ly / 2) * self._config.aperture + 1e-7  # (m)
        A = torch.nan_to_num(torch.sqrt(X1_Y1) / aperture_radius) <= 1.0

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
        height_map = h_r[r.to(torch.int64)]
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
        self._X1, self._Y1 = X1.to(self._device), Y1.to(self._device)
        self._X1_Y1 = X1_Y1.to(self._device)
        self._H_valid = H_valid.to(self._device)
        self._H = H.to(self._device)
        self._FX, self._FY = FX.to(self._device), FY.to(self._device)
        self._FX_FY = FX_FY.to(self._device)
        self._k = k.to(self._device)
        self._A = A.to(self._device)
        self._pupil = pupil.to(self._device)
        self._height_map = height_map.to(self._device)
        self._psf_resolution = psf_resolution

    def _precompute_psfs(self):
        """This will precompute the PSFs for all depths. This is done to avoid
        recomputing the PSF for each render call."""
        for depth in self._depths:
            self._psfs[depth.item()] = self._calculate_psf(depth).to(self._device)

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

    def render(self) -> np.ndarray:
        """Overwrites the default render method to apply the depth invariant PSF to the
        image."""
        image, depth = self._renderer.render()

        # pytorch doesn't support negative strides, so copy if there are any
        image = image.copy() if any(s < 0 for s in image.strides) else image
        image = torch.from_numpy(image).to(self._device)

        # Calculate the depth. Remove the sky depth, which is capped at the extent
        # of the configured environment and apply a far field approximation assumption.
        depth = depth[depth < np.max(depth)]
        depth = np.clip(depth, 5 * max(self.config.focal), np.inf)
        mean_depth = torch.tensor(np.mean(depth), device=self._device)

        # Add noise to the image
        image = self._apply_noise(image, self._config.noise_std)

        # Apply the depth invariant PSF
        psf = self._get_psf(mean_depth)

        # Image may be batched in the form
        image = image.permute(2, 0, 1).unsqueeze(0)
        psf = psf.unsqueeze(1)
        image = torch.nn.functional.conv2d(image, psf, padding="same", groups=3)

        # Post-process the image
        image = image.squeeze(0).permute(1, 2, 0)
        image = self._crop(image)
        image = torch.clip(image, 0, 1)

        return image.cpu().numpy()

    def _apply_noise(self, image: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise to the image."""
        if std == 0.0:
            return image

        noise = torch.normal(mean=0.0, std=std, size=image.shape, device=self._device)
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


if __name__ == "__main__":
    import mujoco as mj
    import matplotlib.pyplot as plt

    from cambrian.utils import setattrs_temporary
    from cambrian.utils.cambrian_xml import MjCambrianXML
    from cambrian.utils.config import run_hydra, MjCambrianConfig

    def run(config: MjCambrianConfig, aperture: float = None):
        if aperture is not None:
            animal_config = next(iter(config.env.animals.values()))
            eye_name, eye_config = next(iter(animal_config.eyes.items()))
            eye_config1 = eye_config.copy()
            eye_config1.set_readonly(False)
            eye_config1.aperture = aperture
            animal_config.eyes[eye_name] = eye_config1

        # xml = MjCambrianXML.from_string(config.env.xml)
        xml = MjCambrianXML("models/test.xml")

        # NOTE: Only uses the first animal
        animal_config = next(iter(config.env.animals.values()))
        animal = animal_config.instance(animal_config, "animal_0", 0)
        xml += animal.generate_xml()

        # Load the model and data
        model = mj.MjModel.from_xml_string(xml.to_string())
        data = mj.MjData(model)
        mj.mj_step(model, data)

        # Reset the animal
        animal.reset(model, data)

        # Set initial state
        animal.quat = [np.cos(np.pi / 2), 0, 0, np.sin(np.pi / 2)]
        mj.mj_step(model, data)

        # Get the first eye
        eye: MjCambrianOpticsEye = next(iter(animal.eyes.values()))
        eye._renderer.viewer._scene.flags[mj.mjtRndFlag.mjRND_SKYBOX] = 1
        rgb, depth = eye._renderer.render()
        obs = eye.render()

        # Get the PSFs
        filtered_depth = depth[depth < np.max(depth)]
        filtered_depth = np.clip(filtered_depth, 5 * max(eye.config.focal), np.inf)
        mean_depth = torch.tensor(filtered_depth.mean(), device=eye._device)

        with eye.config.set_readonly_temporarily(False), setattrs_temporary(
            (eye.config, dict(refractive_index=1))
        ):
            aperture_only_psf: np.ndarray = eye._get_psf(mean_depth).cpu().numpy()
            aperture_only_psf = (aperture_only_psf - aperture_only_psf.min()) / (
                aperture_only_psf.max() - aperture_only_psf.min()
            )
        psf: np.ndarray = eye._get_psf(mean_depth).cpu().numpy()
        psf = (psf - psf.min()) / (psf.max() - psf.min())

        # Get the height map and pupil
        height_map = eye._height_map.cpu().numpy()
        aperture_img: np.ndarray = eye._A.cpu().numpy()

        # Plot the image and depth
        def imshow(ax, image: np.ndarray, title: str, **kwargs):
            ax.imshow(image, **kwargs)
            ax.set_title(title)
            ax.axis("off")

        fig, ax = plt.subplots(4, 2, figsize=(20, 30))  # r, c
        imshow(ax[0, 0], rgb.transpose(1, 0, 2), "Image")
        imshow(ax[0, 1], depth.transpose(1, 0), "Depth", cmap="gray")
        imshow(ax[1, 0], obs.transpose(1, 0, 2), "Observation")
        imshow(ax[1, 1], aperture_img, f"Aperture: {eye.config.aperture}", cmap="gray")
        imshow(ax[2, 0], aperture_only_psf.transpose(1, 2, 0), "Aperture Only PSF")
        imshow(ax[2, 1], psf.transpose(1, 2, 0), "PSF")
        imshow(ax[3, 0], height_map, "Height Map")

        # Save the file to a filename with the config
        aperture = f"aperture_{str(eye.config.aperture).replace('.', 'p')}"
        sp_res = f"sp_{eye.config.renderer.width}x{eye.config.renderer.height}"
        pp_res = f"pp_{eye.config.pupil_resolution[0]}x{eye.config.pupil_resolution[1]}"
        filename = f"{aperture}_{sp_res}_{pp_res}.png"

        plt.suptitle(filename.replace(".png", ""))
        plt.tight_layout()
        plt.savefig(config.logdir / filename)

    def main(config: MjCambrianConfig):
        config.set_readonly(False)
        config.logdir.mkdir(parents=True, exist_ok=True)

        run(config)
        run(config, 0.5)
        run(config, 0.1)
        run(config, 0.0)
        run(config, 0.9)
        run(config, 0.99)

    run_hydra(main)
