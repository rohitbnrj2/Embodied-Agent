import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import torch
from PIL import Image

from typing import Tuple

from cambrian.utils.config import MjCambrianEyeConfig

def electric_field(k, z1, X1, Y1):
    return np.exp(1j*k*np.sqrt(X1**2+Y1**2+z1**2)) 

def prep_buffer(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.tensor(x)
    return x

def rs_prop (u1,z,FX,FY,lmbda):
    
    k = 2*np.pi/lmbda
    H_valid = (np.sqrt(FX**2+FY**2) < 1./lmbda).astype(np.float32)
    H = H_valid * np.nan_to_num(np.exp(1j*k*z*np.sqrt(1.-(lmbda*FX)**2-(lmbda*FY)**2)))
     
    U1 = fft2(fftshift(u1))
    U2 = fftshift(H) * U1
    u2 = ifftshift(ifft2(U2))
    return u2

def _crop( image: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
    """Crop the image to the resolution specified in the config."""
    cw, ch = int(np.ceil(resolution[0] / 2)), int(np.ceil(resolution[1] / 2))
    ox, oy = 1 if resolution[0] == 1 else 0, 1 if resolution[1] == 1 else 0
    bl = (image.shape[0] // 2 - cw + ox, image.shape[1] // 2 - ch + oy)
    tr = (image.shape[0] // 2 + cw, image.shape[1] // 2 + ch)
    return image[bl[0] : tr[0], bl[1] : tr[1]]

def add_gaussian_noise(images: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    # add noise to the images with mean 0.5 and standard deviation std
    noise = torch.normal(mean=0.0, std=std, size=images.shape)
    return torch.clamp(images + noise, 0, 1)

class MjCambrianNonDifferentiableOptics(torch.nn.Module):

    def __init__(self, config: MjCambrianEyeConfig):
        super(MjCambrianNonDifferentiableOptics).__init__()
        self._debug = False
        self.reset(config)

    def reset(self, config: MjCambrianEyeConfig):
        self.config = config
        self.A, self.X1, self.Y1, self.FX, self.FY, self.focal = self.define_simple_psf(config)

    def forward(self, image: np.ndarray, depth: np.ndarray, return_psf: bool = False) -> np.ndarray:
        """Apply the depth invariant PSF to the image.
        """
        if self.config.add_noise:
            image = add_gaussian_noise(image, self.config.noise_std)

        psf = self.depth_invariant_psf(np.mean(depth), self.A, self.X1, self.Y1, self.FX, 
                              self.FY, self.focal, self.config.wavelengths)

        # Image.fromarray((image * 255).astype(np.uint8)).save('image_80x80.png')
        # np.save(f"depth_80x80.npy", depth)
            
        image = torch.tensor(image)
        if torch.cuda.is_available():
            # send to gpu
            image = image.cuda()
            psf = psf.cuda()
        
        image = image.unsqueeze(0).permute(0, 3, 1, 2) # [H, W, C] -> [1, C, H, W]
        psf = psf.permute(2, 0, 1).unsqueeze(0) # [H, W, C] -> [1, C, H, W]
        img = []
        img.append(torch.nn.functional.conv2d(image[:, 0, :,:], psf[:, 0, None, :,:], padding='same'))
        img.append(torch.nn.functional.conv2d(image[:, 1, :,:], psf[:, 1, None, :,:], padding='same'))
        img.append(torch.nn.functional.conv2d(image[:, 2, :,:], psf[:, 2, None, :,:], padding='same'))
        img = torch.cat(img, dim=0).permute(1, 2, 0).cpu().numpy()
        # img = torch.nn.functional.conv2d(image, psf, padding='same').cpu().numpy() # otherwise output is [1, 1, H, W]
        if return_psf:
            psf = psf.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return img, psf 
        
        if self._debug:
            # save image and psf to file for debugging
            psf = psf.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # make a new director inside logs to save the images
            import os
            import matplotlib.pyplot as plt
            os.makedirs('logs/psfs/', exist_ok=True)
            psf[:,:,0] = (psf[:,:,0] - np.min(psf[:,:,0])) / (np.max(psf[:,:,0]) - np.min(psf[:,:,0]))
            psf[:,:,1] = (psf[:,:,1] - np.min(psf[:,:,1])) / (np.max(psf[:,:,1]) - np.min(psf[:,:,1]))
            psf[:,:,2] = (psf[:,:,2] - np.min(psf[:,:,2])) / (np.max(psf[:,:,2]) - np.min(psf[:,:,2]))
            fig, axs = plt.subplots(2, 2)
            # set title of the whole plot 
            axs[0, 0].imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy()); axs[0, 0].set_title(f'GT Image') # axs[0].axis('off')``
            axs[0, 1].imshow(psf); axs[0, 1].set_title('PSF') # axs[1].axis('off')
            axs[1, 0].imshow(img); axs[1, 0].set_title(f'Simulated Image with Aperture: {self.config.aperture_open:2f}') # axs[0].axis('off')
            # downample img to self.config.resoltion
            img = _crop(img, (self.config.resolution[1], self.config.resolution[0]))
            img = Image.fromarray((img * 255).astype(np.uint8))
            img = np.array(img).astype(np.float32) / 255.0
            axs[1, 1].imshow(img); axs[1, 1].set_title(f'Eye Image') # axs[0].axis('off')
            title = f'SceneRes-{image.shape[2]}x{image.shape[3]}-PSF-{psf.shape[0]}x{psf.shape[1]}-EyeRes-{img.shape[0]}x{img.shape[1]}-aperture-{self.config.aperture_open:2f}'
            fig.suptitle(f'PSF Debugging: {title}')
            filename = f'./logs/psfs/{title}.png'
            fig.savefig(filename)
            self._debug = False
            raise Exception(f"{filename} saved for debugging")
        
        return np.clip(img, 0, 1), None

    def define_simple_psf(self, config: MjCambrianEyeConfig) -> torch.Tensor:
        """Define a simple point spread function (PSF) for the eye.
        """

        dx = config.pixel_size              # pixel pitch of sensor (m)            
        Mx = config.sensor_resolution[1]    # number of pixels in x direction
        My = config.sensor_resolution[0]    # number of pixels in y direction
        assert (Mx > 2 or My > 2), \
            f"Minimum resolution for sensor plane should be greater than 2 in x/y direction.: ({Mx}, {My})"
        assert (Mx % 2 != 0 and My % 2 != 0), \
            "Sensor resolution should be odd in both x and y direction. odd length is better for performance"
        Lx = dx * Mx                        # length of simulation plane (m)
        Ly = dx * My                        # length of simulation plane (m)
        if dx > 1e-3:  
            print(f"Warning: Pixel size {dx} m > 0.001m. Required SENSOR resolution: {Lx/1e-3} for input fov and  sensorsize.")
        focal = config.focal

        # Image plane coords                              
        x1 = np.linspace(-Lx/2.,Lx/2.,Mx) 
        y1 = np.linspace(-Ly/2.,Ly/2.,My) 
        X1,Y1 = np.meshgrid(x1,y1)

        # Frequency coords
        fx = np.linspace(-1./(2.*dx),1./(2.*dx),Mx)
        fy = np.linspace(-1./(2.*dx),1./(2.*dx),My)
        FX,FY = np.meshgrid(fx,fy)
        
        # Aperture
        max_aperture_size = dx * int(Mx / 2) # (m)
        aperture_radius = np.interp(np.clip(config.aperture_open, 0, 1), [0, 1], [0, max_aperture_size])
        A = (np.sqrt(X1**2+Y1**2)/(aperture_radius + 1.0e-7) <= 1.).astype(np.float32)
        return A, X1, Y1, FX, FY, focal

    def depth_invariant_psf(self, mean_depth, A, X1, Y1, FX, FY, focal, wavelengths) -> torch.Tensor:
        """
        mean_depth: float, mean depth of the point source
        """
        z1 = mean_depth # z1 is average distance of point source
        psfs = []
        for _lambda in wavelengths:
            k = 2*np.pi/_lambda
            # electric field originating from point source
            u1 = electric_field(k, z1, X1, Y1)
            # electric field at the apertur
            u2 = u1 * A 
            # electric field at the sensor plane
            u3 = rs_prop(u2, focal[0], FX, FY, _lambda)
            psf = np.abs(u3)**2
            # psf should sum to 1 because of energy 
            psf /= (np.sum(psf) + 1.0e-7) 
            psfs.append(torch.tensor(psf).unsqueeze(-1))

        return torch.cat(psfs, dim=-1).float()

    
if __name__ == "__main__":
    
    import os
    import matplotlib.pyplot as plt
    import imageio
    from cambrian.utils.utils import MjCambrianArgumentParser
    from cambrian.utils.config import MjCambrianConfig

    # get config from argparse
    parser = MjCambrianArgumentParser()
    args = parser.parse_args()
    config: MjCambrianEyeConfig = MjCambrianEyeConfig.load(
        args.config, overrides=args.overrides
    )

    img_path = 'misc/psf1/animal_0_eye_1_GT_im_4.png'
    depth_path = 'misc/psf1/animal_0_eye_1_GT_depth_4.npy'

    # img_path = 'misc/psf1/animal_0_eye_0_GT_im_0.png'
    # img_path = 'misc/psf1/animal_0_eye_1_GT_im_8.png'
    # img_path = 'misc/res20/animal_0_eye_0_GT_im_10.png'
    # img_path = 'misc/res20/animal_0_eye_0_GT_im_2.png'
    # img_path = 'misc/res5/animal_0_eye_0_GT_im_2.png'

    # depth_path = 'misc/psf1/animal_0_eye_0_GT_depth_0.npy'
    # depth_path = 'misc/psf1/animal_0_eye_1_GT_depth_8.npy'
    # depth_path = 'misc/res20/animal_0_eye_0_GT_depth_10.npy'
    # depth_path = 'misc/res20/animal_0_eye_0_GT_depth_2.npy'
    # depth_path = 'misc/res5/animal_0_eye_0_GT_depth_2.npy'

    img_path = 'misc/flatland/animal_0_eye_0_GT_im_4.png'
    depth_path = 'misc/flatland/animal_0_eye_0_GT_depth_4.npy'
    # img_path = 'image_80x80.png'
    # depth_path = 'depth_80x80.npy'
    # depth_path = 'depth_80x1.npy'
    
    res = [[5, 5], [10, 10], [20, 20]]
    # res = [[25, 25], [40, 40]] #, [60, 60]]
    res = [[25, 2]] #, [60, 60]]
    res = [[60, 2], [25, 25], [200, 200]] #, [60, 60]]
    optics = MjCambrianNonDifferentiableOptics(config)
    for i in range(len(res)):
        img = Image.open(img_path) #.resize((100,100))
        img = np.array(img).astype(np.float32)[...,0:3]
        print(f"img: {img.shape}")
        img = img / 255.0
        depth = np.load(depth_path)
        _save_path = f'aps-debug-flatland-SceneRes{img.shape[0]}x{img.shape[1]}-ApRes{res[i][0]}x{res[i][1]}'
        os.makedirs(_save_path, exist_ok=True)
        config.sensor_resolution = res[i]
        config.aperture_open = 0 
        # create a plot and save depth and image
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[0].set_title('Image')
        # axs[0].axis('off')
        axs[1].imshow(depth)
        plt.colorbar(axs[1].imshow(depth), ax=axs[1])
        axs[1].set_title(f'Depth min: {np.min(depth):.2f} max: {np.max(depth):.2f}')
        # axs[1].axis('off')
        fig.savefig(f'./{_save_path}//gt.png')

        images = []
        SSNAME = 'dx1em3'
        print(f"Saving images to: {_save_path}")
        apertures = [0., 0.01, 0.1, 0.25, 0.5, 1.0]
        for aperture in apertures:
            config.aperture_open = aperture
            optics.reset(config)
            img, psf = optics.forward(img, depth, return_psf=True)
            print(f"Rendered img: {img.shape}")
            #normalize psf to 0 and 1 for each channel
            psf[:,:,0] = (psf[:,:,0] - np.min(psf[:,:,0])) / (np.max(psf[:,:,0]) - np.min(psf[:,:,0]))
            psf[:,:,1] = (psf[:,:,1] - np.min(psf[:,:,1])) / (np.max(psf[:,:,1]) - np.min(psf[:,:,1]))
            psf[:,:,2] = (psf[:,:,2] - np.min(psf[:,:,2])) / (np.max(psf[:,:,2]) - np.min(psf[:,:,2]))

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(img)
            axs[0].set_title(f'Image After PSF Applied (Aperture: {config.aperture_open:2f})')
            # axs[0].axis('off')
            axs[1].imshow(psf)
            axs[1].set_title('PSF')
            # axs[1].axis('off')
            filename = f'./{_save_path}//psf-im-{config.aperture_open:2f}-{SSNAME}.png'
            fig.savefig(filename)
            images.append(imageio.imread(filename))

        imageio.mimsave(f'./{_save_path}/psf-ims-{SSNAME}.gif', images, loop=4, duration = 1)