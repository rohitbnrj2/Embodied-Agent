import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio
import torch 

"""
Optics for the Rendering. Simulates a DOE on an animal. 
See Training/Implementaion details in https://drive.google.com/file/d/1ISWnM1NhrcNpu5vBtejTQdS9GNuiQyqW/view?pli=1 
code: https://github.com/YichengWu/PhaseCam3D/blob/master/depth_estimation.py#L96
"""

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def fft2dshift(input):
    dim = input.shape[1]  # dimension of the data
    if dim % 2 == 0:
        raise ValueError('Please make the size of kernel odd')
    # pdb.set_trace()c
    
    channel1 = input.shape[0]  # channels for the first dimension
    # shift up and down
    # u = torch.slice(input, [0, 0, 0], [channel1, int((dim + 1) / 2), dim])
    u = input[0:channel1, 0:int((dim + 1) / 2), 0:dim]
    
    # d = torch.slice(input, [0, int((dim + 1) / 2), 0], [channel1, int((dim - 1) / 2), dim])
    d = input[0:channel1, int((dim + 1) / 2):int((dim + 1) / 2) + int((dim - 1) / 2), 0:dim]
    du = torch.concat([d, u], axis=1)
    
    # shift left and right
    # l = torch.slice(du, [0, 0, 0], [channel1, dim, int((dim + 1) / 2)])
    l = du[0:channel1, 0:dim, 0:int((dim + 1) / 2)]
    # r = torch.slice(du, [0, 0, int((dim + 1) / 2)], [channel1, dim, int((dim - 1) / 2)])
    r = du[0:channel1, 0:dim, int((dim + 1) / 2):int((dim + 1) / 2) + int((dim - 1) / 2)]
    output = torch.concat([r, l], axis=2)
    return output


class Optics():

    def __init__(self, psf_kernel_size, wvls, min_depth, max_depth, depth_bins) -> None:
        self.disc_depth_range = disc_depth_range
        self.psf_kernel_size = psf_kernel_size
        # wvls=np.array([610., 530., 470.]) * 1e-9
        self.wvls = wvls
        self.depth_bins = depth_bins
        self.min_depth = min_depth
        self.max_depth = max_depth #min_depth + depth_bins

        self.N_R = 31
        self.N_G = 27
        self.N_B = 23  # size of the blur kernel

        self.psf_kernel_size = 23 
        self.N_R = 31
        self.N_G = 27
        self.N_B = 23  # size of the blur kernel

        self.disc_depth_range = np.linspace(self.min_depth, self.max_depth, self.depth_bins, np.float32) 
        self.defocus_phase = self.generate_defocus_phase(self.disc_depth_range, self.psf_kernel_size, self.wvls)

    def render(self, height_mask, rgb, depth):
        """
        height_mask: np.array() of size [23, 23]
        rgb: np.array(H, W, 3, dtype=np.uint8)
        depth: np.array((H, W), dtype=np.float) between (self.min_depth, max_depth)
        """
        rgb_torch = torch.tensor(rgb).unsqueeze(0)/255.
        depth_torch = self.compute_disc_depths(depth, self.min_depth, self.max_depth).unsqueeze(0).permute(0,3,1,2)
        psfs = self.generate_psf_from_height_map(height_mask, self.defocus_phase, self.wvls, idx, N_R, N_G, N_B)
        rgb = self.blur_image(rgb_torch, depth_torch, psfs)
        return rgb

    def generate_defocus_phase(self, disc_depth_range, psf_kernel_size, wvls):
        """

        disc_depth_range: 
        wvls: wavelength that psf is suceptible to. 
        """
        # return (Phi_list,pixel,pixel,color)
        x0 = np.linspace(-1.1, 1.1, psf_kernel_size)
        xx, yy = np.meshgrid(x0, x0)
        defocus_phase = np.empty([len(disc_depth_range), psf_kernel_size, psf_kernel_size, len(wvls)], dtype=np.float32)
        for j in range(len(disc_depth_range)):
            phi = disc_depth_range[j]
            for k in range(len(wvls)):
                defocus_phase[j, :, :, k] = phi * (xx ** 2 + yy ** 2) * wvls[1] / wvls[k];
        return defocus_phase

    def get_height_map(self, alpha_zernike, n_coeff_zernike, u2, psf_kernel_size, wvls, init_func='random'):

        if alpha_zernike == None:
            alpha_zernike = torch.zeros((n_coeff_zernike, 1), dtype=torch.float32)

        clip_alphas = lambda x: torch.clip(x, -wvls[1] / 2, wvls[1] / 2)
        alpha_zernike = clip_alphas(alpha_zernike)
        g = torch.matmul(torch.tensor(u2), alpha_zernike)
        height_map = torch.relu(g.reshape((psf_kernel_size, psf_kernel_size)) + wvls[1])
        return height_map

    def generate_psf_from_height_map(self, height_mask, defocus_phase, wvls, idx, N_R, N_G, N_B, refactive_idx=1.5):
        idx = torch.tensor(idx)
        height_mask = torch.tensor(height_mask)
        defocus_phase = torch.tensor(defocus_phase)
        
        defocus_phase_r = defocus_phase[:, :, :, 0]
        phase_R = torch.add(2 * np.pi / wvls[0] * (refactive_idx - 1) * height_mask, defocus_phase_r)
        e_defocused_r = torch.mul(torch.complex(idx, torch.tensor(0.0)), torch.exp(torch.complex(torch.tensor(0.0), phase_R)))

        pad_r = ((N_R - N_B) // 2, (N_R - N_B) // 2, (N_R - N_B) // 2, (N_R - N_B) // 2)
        pupil_r = torch.nn.functional.pad(e_defocused_r, pad_r)
        norm_r = N_R * N_R * torch.sum(idx ** 2)
        fft_pupil_r = torch.fft.fft2(pupil_r); 
        psf_r = torch.divide(torch.square(torch.abs(fft2dshift(fft_pupil_r))), norm_r)

        defocus_phase_g = defocus_phase[:, :, :, 1]
        phase_G = torch.add(2 * np.pi / wvls[1] * (refactive_idx - 1) * height_mask, defocus_phase_g)
        e_defocused_g = torch.mul(torch.complex(idx, torch.tensor(0.0)), torch.exp(torch.complex(torch.tensor(0.0), phase_G)))
        pad_g = ((N_G - N_B) // 2, (N_G - N_B) // 2, (N_G - N_B) // 2, (N_G - N_B) // 2)
        pupil_g = torch.nn.functional.pad(e_defocused_g, pad_g)
        norm_g = N_G * N_G * torch.sum(idx ** 2)
        fft_pupil_g = torch.fft.fft2(pupil_g)
        psf_g = torch.divide(torch.square(torch.abs(fft2dshift(fft_pupil_g))), norm_g)

        defocus_phase_b = defocus_phase[:, :, :, 2]
        phase_B = torch.add(2 * np.pi / wvls[2] * (refactive_idx - 1) * height_mask, defocus_phase_b)
        pupil_b = torch.mul(torch.complex(idx, torch.tensor(0.0)), torch.exp(torch.complex(torch.tensor(0.0), phase_B)))
        norm_b = N_B * N_B * torch.sum(idx ** 2)
        fft_pupil_b = torch.fft.fft2(pupil_b)
        psf_b = torch.divide(torch.square(torch.abs(fft2dshift(fft_pupil_b))), norm_b)
        print(psf_r.shape, psf_g.shape, psf_b.shape)

        N_crop_R = int((N_R - N_B) / 2)  # Num of pixel need to cropped at each side for R
        N_crop_G = int((N_G - N_B) / 2)  # Num of pixel need to cropped at each side for G

        psfs = torch.stack(
            [psf_r[:, N_crop_R:-N_crop_R, N_crop_R:-N_crop_R], psf_g[:, N_crop_G:-N_crop_G, N_crop_G:-N_crop_G], psf_b], axis=3)
                
        return psfs

    def blur_image(self, RGBPhi, DPPhi, PSFs, apply_normalize=True):
        N_B = PSFs.shape[1]
        N_crop = np.int32((N_B - 1) / 2)
        N_Phi = PSFs.shape[0]

        sharp_R = RGBPhi[:, :, :, 0:1].permute(0,3,1,2)
        PSFs_R = torch.reshape(torch.permute(PSFs[:, :, :, 0], dims=(1, 2, 0)), [N_Phi, 1, N_B, N_B])
        blurAll_R = torch.nn.functional.conv2d(sharp_R, PSFs_R, stride=[1, 1], padding='valid')
        blur_R = torch.sum(torch.multiply(blurAll_R, DPPhi[:, :, N_crop:-N_crop, N_crop:-N_crop]), axis=1)
        
        sharp_G = RGBPhi[:, :, :, 1:2].permute(0,3,1,2)
        PSFs_G = torch.reshape(torch.permute(PSFs[:, :, :, 1], dims=[1, 2, 0]), [N_Phi, 1, N_B, N_B])
        blurAll_G = torch.nn.functional.conv2d(sharp_G, PSFs_G, stride=[1, 1], padding='valid')
        blur_G = torch.sum(torch.multiply(blurAll_G, DPPhi[:, :, N_crop:-N_crop, N_crop:-N_crop]), axis=1)


        sharp_B = RGBPhi[:, :, :, 2:3].permute(0,3,1,2)
        PSFs_B = torch.reshape(torch.permute(PSFs[:, :, :, 2], dims=[1, 2, 0]), [N_Phi, 1, N_B, N_B])
        blurAll_B = torch.nn.functional.conv2d(sharp_B, PSFs_B, stride=[1, 1], padding='valid')
        blur_B = torch.sum(torch.multiply(blurAll_B, DPPhi[:, :, N_crop:-N_crop, N_crop:-N_crop]), axis=1)

        blur = torch.stack([blur_R, blur_G, blur_B], axis=3).squeeze().numpy()
        # blur = np.clip(blur, 0., 1.)
        if apply_normalize:
            blur = normalize(blur) 
        return blur

    def compute_disc_depths(self, mj_depth):
        disc_depth = []
        # for each depth value see if it's close to self.disc_depth_range[n-1] < x < self.disc_depth_range[n]
        mj_depth = mj_depth.round().astype(np.uint8)
        for i in range(min_bin, max_bin+1, 1):
            idx = np.where(mj_depth == i)
            disc_depth_i = np.zeros_like(mj_depth ,dtype=np.float32)
            disc_depth_i[idx] = 1
            disc_depth.append(torch.tensor(disc_depth_i).unsqueeze(-1))

        disc_depth = torch.concat(disc_depth, -1)
        return disc_depth


if __name__ == "__main__": 
    import sys 
    rgb_path = sys.argv[1]
    depth_path = sys.argv[2]
    rgb = np.array(Image.open(rgb_path))
    depth = np.array(np.load(depth_path))
    print("depth (min, max): ({},{})".format(depth.min(), depth.max()))

    # config
    DEPTH_BINS = 21
    DEPTH_MIN = -10
    DEPTH_MAX = 10
    N_R = 31
    N_G = 27
    N_B = 23  # size of the blur kernel
    wvls=np.array([610., 530., 470.]) * 1e-9

    psf_kernel_size = 23 # we should probably 10 to keep the parameters small

    # uncomment to load zernekie polynomials and height mask from scratch!
    zernike = sio.loadmat('tools/zernike_basis.mat')
    u2 = zernike['u2']  # basis of zernike poly
    n_coeff_zernike = u2.shape[1]
    idx = zernike['idx']
    idx = idx.astype(np.float32)
    # alpha_zernike = None
    # height_mask = get_height_map(alpha_zernike, n_coeff_zernike, u2, psf_kernel_size, wvls)
    height_mask_fisher = np.loadtxt('tools/FisherMask_HeightMap.txt').astype(np.float32)

    optics = Optics(psf_kernel_size, wvls, DEPTH_MIN, DEPTH_MAX, DEPTH_BINS)
    img = optics.render(height_mask_fisher, rgb, depth)

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(rgb)
    ax1.set_title("Sampled RGB")
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(depth)
    ax2.set_title("Sampled Depth")
    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(height_mask_fisher)
    ax3.set_title("Fisher Height Mask")
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(img)
    ax4.set_title("RGB after PSF based Convolution")
    plt.savefig("fisher.png")