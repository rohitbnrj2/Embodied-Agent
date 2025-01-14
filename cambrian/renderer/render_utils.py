"""Rendering utilities."""

from typing import Dict, List, Tuple

import mujoco as mj
import torch
import torch.nn.functional as F

from cambrian.utils import device
from cambrian.utils.constants import C


def resize(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize the image to the specified height and width."""
    squeeze = False
    if images.ndim == 3:
        squeeze = True
        images = images.unsqueeze(0)

    resized_images = F.interpolate(
        images.permute(0, 3, 1, 2),
        size=(height, width),
        mode="nearest",
        align_corners=None,
    ).permute(0, 2, 3, 1)

    return resized_images.squeeze(0) if squeeze else resized_images


def resize_with_aspect_fill(
    images: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Resize the image while maintaining the aspect ratio and
    filling the rest with black."""
    squeeze = False
    if images.ndim == 3:
        squeeze = True
        images = images.unsqueeze(0)

    original_height, original_width = images.shape[1:3]
    ratio_original = original_width / original_height
    ratio_new = width / height

    transpose = False
    if ratio_original < ratio_new:
        # Transpose for taller images
        transpose = True
        images = images.permute(0, 2, 1, 3)
        height, width = width, height
        ratio_new = width / height
        ratio_original = original_height / original_width

    resize_height = max(1, round(width / ratio_original))
    resized_images = resize(images, resize_height, width)

    pad_top = (height - resize_height) // 2
    pad_bottom = height - resize_height - pad_top
    padded_images = F.pad(
        resized_images.permute(0, 3, 1, 2),
        (0, 0, pad_top, pad_bottom),
        mode="constant",
        value=0,
    ).permute(0, 2, 3, 1)

    if transpose:
        # Transpose back
        padded_images = padded_images.permute(0, 2, 1, 3)

    return padded_images.squeeze(0) if squeeze else padded_images


def add_border(
    images: torch.Tensor,
    border_size: int,
    color: tuple = (0, 0, 0),
) -> torch.Tensor:
    squeeze = False
    if images.ndim == 3:
        squeeze = True
        images = images.unsqueeze(0)
    color = torch.tensor(color, device=images.device).unsqueeze(-1).unsqueeze(-1)
    pad = (border_size, border_size, border_size, border_size)
    images = images.permute(0, 3, 1, 2)
    padded = F.pad(images, pad, value=0.0)
    padded[..., :border_size, :] = color
    padded[..., -border_size:, :] = color
    padded[..., :, :border_size] = color
    padded[..., :, -border_size:] = color
    return padded.squeeze(0).permute(1, 2, 0) if squeeze else padded.permute(0, 2, 3, 1)


def generate_composite(images: Dict[float, Dict[float, torch.Tensor]]) -> torch.Tensor:
    """This is a debug method which renders the images as a composite image.

    Will appear as a compound eye. For example, if we have a 3x3 grid of eyes:
        TL T TR
        ML M MR
        BL B BR

    Each eye has a red border around it.

    Note:

        This assumes that the images have the same dimensions.
    """
    composite = torch.stack(
        [
            images[lat][lon]
            for lat in sorted(images.keys())
            for lon in sorted(images[lat].keys())[::-1]
        ]
    )
    _, H, W, _ = composite.shape
    if H > W:
        h = max(H, 10)
        w = int(W * h / H)
    else:
        w = max(W, 10)
        h = int(H * w / W)
    composite = resize_with_aspect_fill(composite, h, w)
    composite = add_border(composite, 1, color=(1, 0, 0))

    # Resize the composite image while maintaining the spatial position
    _, H, W, C = composite.shape
    nrows, ncols = len(images), len(next(iter(images.values())))
    composite = (
        composite.view(nrows, ncols, H, W, C)
        .permute(0, 2, 1, 3, 4)
        .reshape(nrows * H, ncols * W, C)
    )

    return composite


class CubeToEquirectangularConverter:
    def __init__(self, size: Tuple[int, int], image_size: Tuple[int, int]):
        self.full_size = size
        full_height, full_width = size
        self.image_size = image_size
        img_height, img_width = image_size

        theta = torch.linspace(-torch.pi, torch.pi, steps=full_width, device=device)
        phi = torch.linspace(
            -torch.pi / 2, torch.pi / 2, steps=full_height, device=device
        )
        theta, phi = torch.meshgrid(theta, phi, indexing="ij")
        theta = theta.transpose(0, 1)
        phi = phi.transpose(0, 1)

        x = torch.cos(phi) * torch.sin(theta)
        y = torch.sin(phi)
        z = torch.cos(phi) * torch.cos(theta)

        cos45 = torch.cos(torch.tensor(45.0 * torch.pi / 180, device=device))
        sin45 = torch.sin(torch.tensor(45.0 * torch.pi / 180, device=device))

        x_rot = x * cos45 + z * sin45
        z_rot = -x * sin45 + z * cos45

        abs_x_rot = x_rot.abs()
        abs_y = y.abs()
        abs_z_rot = z_rot.abs()

        face_indices = torch.zeros((full_height, full_width), dtype=int, device=device)
        u = torch.zeros((full_height, full_width), dtype=torch.float32, device=device)
        v = torch.zeros((full_height, full_width), dtype=torch.float32, device=device)

        mask_left = (abs_x_rot >= abs_y) & (abs_x_rot >= abs_z_rot) & (x_rot < 0)
        face_indices[mask_left] = 0
        u[mask_left] = z_rot[mask_left] / abs_x_rot[mask_left]
        v[mask_left] = y[mask_left] / abs_x_rot[mask_left]

        mask_front = (abs_z_rot >= abs_x_rot) & (abs_z_rot >= abs_y) & (z_rot > 0)
        face_indices[mask_front] = 1
        u[mask_front] = x_rot[mask_front] / abs_z_rot[mask_front]
        v[mask_front] = y[mask_front] / abs_z_rot[mask_front]

        mask_right = (abs_x_rot >= abs_y) & (abs_x_rot >= abs_z_rot) & (x_rot > 0)
        face_indices[mask_right] = 2
        u[mask_right] = -z_rot[mask_right] / abs_x_rot[mask_right]
        v[mask_right] = y[mask_right] / abs_x_rot[mask_right]

        mask_back = (abs_z_rot >= abs_x_rot) & (abs_z_rot >= abs_y) & (z_rot < 0)
        face_indices[mask_back] = 3
        u[mask_back] = -x_rot[mask_back] / abs_z_rot[mask_back]
        v[mask_back] = y[mask_back] / abs_z_rot[mask_back]

        mask_top = (abs_y >= abs_x_rot) & (abs_y >= abs_z_rot) & (y > 0)
        face_indices[mask_top] = 4
        u[mask_top] = x_rot[mask_top] / abs_y[mask_top]
        v[mask_top] = -z_rot[mask_top] / abs_y[mask_top]

        mask_bottom = (abs_y >= abs_x_rot) & (abs_y >= abs_z_rot) & (y < 0)
        face_indices[mask_bottom] = 5
        u[mask_bottom] = x_rot[mask_bottom] / abs_y[mask_bottom]
        v[mask_bottom] = z_rot[mask_bottom] / abs_y[mask_bottom]

        u_img = ((u + 1) / 2) * (img_width - 1)
        v_img = (1 - (v + 1) / 2) * (img_height - 1)

        offsets_x = torch.zeros_like(face_indices, dtype=torch.float32)
        offsets_y = torch.zeros_like(face_indices, dtype=torch.float32)

        offsets_x[face_indices == 0] = 0
        offsets_y[face_indices == 0] = 1
        offsets_x[face_indices == 1] = 1
        offsets_y[face_indices == 1] = 1
        offsets_x[face_indices == 2] = 2
        offsets_y[face_indices == 2] = 1
        offsets_x[face_indices == 3] = 3
        offsets_y[face_indices == 3] = 1
        offsets_x[face_indices == 4] = 1
        offsets_y[face_indices == 4] = 0
        offsets_x[face_indices == 5] = 1
        offsets_y[face_indices == 5] = 2

        map_x = u_img + offsets_x * img_width
        map_y = v_img + offsets_y * img_height

        norm_x = (map_x / (4 * img_width - 1)) * 2 - 1
        norm_y = (map_y / (3 * img_height - 1)) * 2 - 1

        grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
        self.grid = grid.to(device)

        self.combined_image = torch.zeros(
            (3 * img_height, 4 * img_width, 3),
            dtype=torch.float32,
            device=device,
        )

    def convert(self, images: List[torch.Tensor]) -> torch.Tensor:
        h, w = self.image_size
        self.combined_image[h : 2 * h, 0:w] = images[0]
        self.combined_image[h : 2 * h, w : 2 * w] = images[1]
        self.combined_image[h : 2 * h, 2 * w : 3 * w] = images[2]
        self.combined_image[h : 2 * h, 3 * w : 4 * w] = images[3]
        self.combined_image[:h, w : 2 * w] = images[4]
        self.combined_image[2 * h : 3 * h, w : 2 * w] = images[5]

        combined_image = self.combined_image.unsqueeze(0).permute(0, 3, 1, 2)
        remapped = F.grid_sample(
            combined_image,
            self.grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        return remapped.squeeze(0).permute(1, 2, 0)


def convert_depth_distances(model: mj.MjModel, depth: torch.Tensor) -> torch.Tensor:
    """Converts depth values from OpenGL to metric depth values using PyTorch.

    Args:
        model (mj.MjModel): The model.
        depth (torch.Tensor): The depth values to convert.

    Returns:
        torch.Tensor: The converted depth values.

    Note:
        This function is based on
        [this code](https://github.com/google-deepmind/mujoco/blob/main/\
            python/mujoco/renderer.py).
        It is adapted to use PyTorch instead of NumPy.
    """

    # Get the distances to the near and far clipping planes.
    extent = model.stat.extent
    near = model.vis.map.znear * extent
    far = model.vis.map.zfar * extent

    # Calculate OpenGL perspective matrix values in float32 precision
    # so they are close to what glFrustum returns
    # https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml
    zfar = torch.tensor(far, dtype=torch.float32)
    znear = torch.tensor(near, dtype=torch.float32)
    c_coef = -(zfar + znear) / (zfar - znear)
    d_coef = -(torch.tensor(2.0, dtype=torch.float32) * zfar * znear) / (zfar - znear)

    # In reverse Z mode the perspective matrix is transformed by the following
    c_coef = torch.tensor(-0.5, dtype=torch.float32) * c_coef - torch.tensor(
        0.5, dtype=torch.float32
    )
    d_coef = torch.tensor(-0.5, dtype=torch.float32) * d_coef

    # We need 64 bits to convert Z from ndc to metric depth without noticeable
    # losses in precision
    out_64 = depth.to(dtype=torch.float64)

    # Undo OpenGL projection
    # Note: We do not need to take action to convert from window coordinates
    # to normalized device coordinates because in reversed Z mode the mapping
    # is identity
    out_64 = d_coef / (out_64 + c_coef)

    # Cast result back to float32 for backwards compatibility
    # This has a small accuracy cost
    return out_64.to(dtype=torch.float32)


def convert_depth_to_rgb(
    depth: torch.Tensor, znear: float | None = None, zfar: float | None = None
) -> torch.Tensor:
    """Converts depth values to RGB values.

    Args:
        model (mj.MjModel): The model.
        depth (torch.Tensor): The depth values.

    Returns:
        torch.Tensor: The RGB values.
    """
    znear = znear or depth.min()
    zfar = zfar or depth.max()
    if znear != zfar:
        depth = (depth - znear) / (zfar - znear)
        depth = torch.clamp(depth, 0.0, 1.0)
    depth = depth.repeat(3, 1, 1).permute(1, 2, 0)
    return depth


def convert_depth_to_tof(
    model: mj.MjModel, depth: torch.Tensor, timing_resolution_ns: float, num_bins: int
) -> torch.Tensor:
    """Converts depth values to time-of-flight. It will return a transient cube, where
    each image is a time-of-flight image at a specific time.

    Args:
        depth (torch.Tensor): The depth values. Shape: (height, width).
        timing_resolution_ns (float): The timing resolution in nanoseconds.
        num_bins (int): The number of bins. The output tensor will have shape
            (num_bins, height, width).

    Returns:
        torch.Tensor: The time-of-flight transient cube. Shape: (num_bins, height,
            width).
    """
    # Ensure depth is 2D (height, width)
    if depth.dim() != 2:
        raise ValueError("Depth tensor must be 2-dimensional (height, width).")

    # Calculate ToF in seconds: ToF = 2 * distance / c
    tof_sec = 2 * convert_depth_distances(model, depth) / C  # Shape: (height, width)

    # Convert ToF to nanoseconds
    tof_ns = tof_sec * 1e9  # Shape: (height, width)

    # Determine the bin index for each ToF value
    bin_indices = torch.floor(tof_ns / timing_resolution_ns).long()

    # Mask out invalid bin indices
    bin_mask = (bin_indices >= 0) & (bin_indices < num_bins)
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    # Create a one-hot encoding for bin indices
    # Shape after one_hot: (height, width, num_bins)
    transient_one_hot = torch.nn.functional.one_hot(bin_indices, num_classes=num_bins)
    transient_one_hot[~bin_mask] = 0

    # Permute to shape (num_bins, height, width)
    transient = transient_one_hot.permute(2, 0, 1).float()

    return transient
