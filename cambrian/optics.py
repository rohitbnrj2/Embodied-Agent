import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import torch

from scipy import signal

# from cambrian.eye import MjCambrianEye
from cambrian.utils.config import MjCambrianEyeConfig


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

class MjCambrianNonDifferentiableOptics(torch.nn.Module):

    def __init__(self):
        super(MjCambrianNonDifferentiableOptics).__init__()

        # aperture_radius: float = 1.0 
        # self.register_buffer('input_field', prep_buffer(aperture_radius))


    def render_aperture_only(self, image: np.ndarray, depth: np.ndarray, config: MjCambrianEyeConfig) -> np.ndarray:
        """
        image: (H, W, 3) should be dtype float32
        """
        psf = self.simple_psf(depth, config)
        
        TORCH_EXECUTION=True
        if TORCH_EXECUTION:
            # switch channels to first dimension
            image = torch.tensor(image).unsqueeze(0)
            if torch.cuda.is_available():
                image = image.cuda()
                psf = psf.cuda()
            # send to gpu
            image = image.permute(0, 3, 1, 2)
            psf = psf.permute(2, 0, 1).unsqueeze(0)
            img = []
            img.append(torch.nn.functional.conv2d(image[:, 0, :,:], psf[:, 0, None, :,:], padding='same'))
            img.append(torch.nn.functional.conv2d(image[:, 1, :,:], psf[:, 1, None, :,:], padding='same'))
            img.append(torch.nn.functional.conv2d(image[:, 2, :,:], psf[:, 2, None, :,:], padding='same'))
            img = torch.cat(img, dim=0).permute(1, 2, 0).cpu().numpy()
            psf = psf.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            psf = psf.numpy()
            image[..., 0] = signal.convolve2d(image[..., 0], psf[..., 0], boundary='symm', mode='same')
            image[..., 1] = signal.convolve2d(image[..., 1], psf[..., 1], boundary='symm', mode='same')
            image[..., 2] = signal.convolve2d(image[..., 2], psf[..., 2], boundary='symm', mode='same')
            return image, psf
            
        return img, psf 

    def simple_psf(self, depth, config: MjCambrianEyeConfig) -> np.ndarray:
        """
        p -> z1 -> a/l/phase -> z2/focal -> s

        choose where to place the lens such that focus is at z_focus
        z_focus = 1.               
        z2 = z_focus*f/(z_focus-f)   # distance between exit pupil and sensor plane
        """

        def electric_field(k, z1, X1, Y1):
            return np.exp(1j*k*np.sqrt(X1**2+Y1**2+z1**2)) 
                
        # Create Sensor Plane
        Mx = config.resolution[0] * 2 # 40 works well  
        My = config.resolution[1] * 2 # 40 works well  
        # Mx = 40 
        # My = 1 #40 
        # id mx and my are even, then change to odd
        # odd length is better for performance
        if Mx % 2 == 0:
            Mx += 1
        if My % 2 == 0:
            My += 1
        
        Lx = config.sensorsize[0] #7.1e-3          # length of simulation plane (m) 
        Ly = config.sensorsize[1] #7.1e-3          # length of simulation plane (m) 
        dx = Lx/Mx #3.45e-6                        # pixel pitch of sensor (m)      
        dy = Ly/My #3.45e-6                        # pixel pitch of sensor (m)

        # Image plane coords                              
        x1 = np.linspace(-Lx/2.,Lx/2.,Mx) 
        y1 = np.linspace(-Ly/2.,Ly/2.,My) 
        X1,Y1 = np.meshgrid(x1,y1)

        # Frequency coords
        fx = np.linspace(-1./(2.*dx),1./(2.*dx),Mx)
        fy = np.linspace(-1./(2.*dy),1./(2.*dy),My)
        FX,FY = np.meshgrid(fx,fy)
        
        # Aperture
        # import pdb; pdb.set_trace()
        max_aperture_size = dx * int(Mx/2) # (m)
        config.aperture_radius = np.clip(config.aperture_open, 0, 1) * max_aperture_size
        # complete aperture size: aperture_size = dx * num_pixels
        # aperture radius: aperture_radius = dx * {radius_in_pixels /in [0, num_pixels/2]}
        self.A = (np.sqrt(X1**2+Y1**2)/(config.aperture_radius + 1.0e-7) <= 1.).astype(np.float32)
        # self.A = (np.sqrt(X1**2+Y1**2) - config.aperture_radius).astype(np.float32)

        # average distance of point source
        z1 = np.mean(depth)
        psfs = []

        for _lambda in config.wavelengths:
            k = 2*np.pi/_lambda
            # electric field originating from point source
            u = electric_field(k, z1, X1, Y1)
            
            # electric field at the aperture
            u = u*self.A #*t_lens*t_mask
            
            # electric field at the sensor plane
            u = rs_prop(u, config.focal[0], FX, FY, _lambda)
            
            psf = np.abs(u)**2
            # psf should sum to 1 because of energy 
            # we dont divide by sum we are giving it more energy 
            psf /= (np.sum(psf) + 1.0e-7) 

            # psf = psf / (np.linalg.norm(psf) + 1.0e-7) # (np.sum(psf) + 1.0e-7)
            psfs.append(torch.tensor(psf).unsqueeze(-1))

        return torch.cat(psfs, dim=-1).float()