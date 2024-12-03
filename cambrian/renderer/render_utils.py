"""Rendering utilities."""

from typing import Tuple, Dict, List

import cv2
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R


def resize_with_aspect_fill(
    image: np.ndarray,
    width: int,
    height: int,
    *,
    border_type: int = cv2.BORDER_CONSTANT,
    interp: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """Resize the image while maintaining the aspect ratio and filling the rest with
    black.

    Args:
        image (np.ndarray): The image to resize.
        width (int): The new width.
        height (int): The new height.

    Keyword Args:
        border_type (int): The type of border to add. Default is cv2.BORDER_CONSTANT.
        interp (int): The interpolation method. Default is cv2.INTER_NEAREST.

    Returns:
        np.ndarray: The resized image.
    """

    # TODO: why is it height, width and not width, height?
    # original_width, original_height = image.shape[:2]
    original_height, original_width = image.shape[:2]
    ratio_original = original_width / original_height
    ratio_new = width / height

    # Resize the image while maintaining the aspect ratio
    if ratio_original > ratio_new:
        # Original is wider relative to the new size
        resize_height = max(1, round(width / ratio_original))
        resized_image = cv2.resize(image, (width, resize_height), interpolation=interp)
        top = (height - resize_height) // 2
        bottom = height - resize_height - top
        result = cv2.copyMakeBorder(resized_image, top, bottom, 0, 0, border_type)
    else:
        # Original is taller relative to the new size
        resize_width = max(1, round(height * ratio_original))
        resized_image = cv2.resize(image, (resize_width, height), interpolation=interp)
        left = (width - resize_width) // 2
        right = width - resize_width - left
        result = cv2.copyMakeBorder(resized_image, 0, 0, left, right, border_type)

    return result


def convert_depth_distances(model: mj.MjModel, depth: np.ndarray) -> np.ndarray:
    """Converts depth values from OpenGL to metric depth values.

    Args:
        model (mj.MjModel): The model.
        depth (np.ndarray): The depth values to convert.

    Returns:
        np.ndarray: The converted depth values.

    Note:
        This function is based on
        [this code](https://github.com/google-deepmind/mujoco/blob/main/\
            python/mujoco/renderer.py).
    """

    # Get the distances to the near and far clipping planes.
    extent = model.stat.extent
    near = model.vis.map.znear * extent
    far = model.vis.map.zfar * extent

    # Calculate OpenGL perspective matrix values in float32 precision
    # so they are close to what glFrustum returns
    # https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml
    zfar = np.float32(far)
    znear = np.float32(near)
    c_coef = -(zfar + znear) / (zfar - znear)
    d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)

    # In reverse Z mode the perspective matrix is transformed by the following
    c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
    d_coef = np.float32(-0.5) * d_coef

    # We need 64 bits to convert Z from ndc to metric depth without noticeable
    # losses in precision
    out_64 = depth.astype(np.float64)

    # Undo OpenGL projection
    # Note: We do not need to take action to convert from window coordinates
    # to normalized device coordinates because in reversed Z mode the mapping
    # is identity
    out_64 = d_coef / (out_64 + c_coef)

    # Cast result back to float32 for backwards compatibility
    # This has a small accuracy cost
    depth[:] = out_64.astype(np.float32)

    return depth


def add_border(
    image: np.ndarray,
    border_size: float,
    color: Tuple[float, float, float] = (0, 0, 0),
) -> np.ndarray:
    """Add a white border around the image."""
    image = cv2.copyMakeBorder(
        image,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return image


def generate_composite(
    images: Dict[float, Dict[float, np.ndarray]],
    max_res: Tuple[int, int],
) -> np.ndarray:
    """This is a debug method which renders the images as a composite image.

    Will appear as a compound eye. For example, if we have a 3x3 grid of eyes:
        TL T TR
        ML M MR
        BL B BR

    Each eye has a red border around it.
    """
    # Construct the composite image
    # Loop through the sorted list of images based on lat/lon
    composite = []
    for lat in sorted(images.keys())[::-1]:
        row = []
        for lon in sorted(images[lat].keys())[::-1]:
            resized_obs = resize_with_aspect_fill(images[lat][lon], *max_res)
            # Add a red border around the image
            resized_obs = add_border(resized_obs, 1, color=(1, 0, 0))
            row.append(resized_obs)
        composite.append(np.vstack(row))
    composite = np.hstack(composite)

    return composite

def project_images_to_spherical_panorama(
    images: List[np.ndarray],
    yaw_angles: List[float],
    fov_x: float,
    fov_y: float,
    total_resolution: Tuple[int, int],
) -> np.ndarray:
    """
    Projects multiple camera images onto a spherical surface to create a panorama.

    Args:
        images: List of images from the cameras.
        yaw_angles: List of yaw angles (in degrees) corresponding to each camera image.
        fov_x: Horizontal field of view of each camera in degrees.
        fov_y: Vertical field of view of each camera in degrees.
        total_resolution: Resolution (width, height) of the resulting panorama.

    Returns:
        The spherical panorama image as a NumPy array.
    """
    total_width, total_height = total_resolution

    # Convert yaw angles to radians
    yaw_angles_rad = [np.deg2rad(yaw) for yaw in yaw_angles]

    # Create an empty panorama image
    panorama = np.zeros((total_width, total_height, 3), dtype=np.uint8)

    # Create meshgrid for panorama image
    phi = np.linspace(-np.pi, np.pi, total_width)
    theta = np.linspace(-np.pi / 2, np.pi / 2, total_height)
    phi, theta = np.meshgrid(phi, theta)

    # Compute the direction vectors for each pixel in the panorama
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    dirs = np.stack((x, y, z), axis=-1)  # Shape: (H, W, 3)

    # Flatten the directions for easier processing
    dirs_flat = dirs.reshape(-1, 3)  # Shape: (H*W, 3)

    # Initialize an array to keep track of which pixels have been filled
    filled_mask = np.zeros((total_width, total_height), dtype=bool)

    for idx, (image, yaw) in enumerate(zip(images, yaw_angles_rad)):
        h, w, _ = image.shape

        # Rotate the direction vectors to align with the camera orientation
        rot = R.from_euler('y', 0) * R.from_euler('z', -yaw)
        dirs_cam = rot.apply(dirs_flat)

        # Field of view in radians
        fov_x_rad = np.deg2rad(fov_x)
        fov_y_rad = np.deg2rad(fov_y)

        # Compute the angles in the camera coordinate system
        phi_cam = np.arctan2(dirs_cam[:, 1], dirs_cam[:, 0])
        theta_cam = np.arcsin(dirs_cam[:, 2] / np.linalg.norm(dirs_cam, axis=1))

        # Mask for pixels within the camera's field of view
        fov_mask = (
            (phi_cam >= -fov_x_rad / 2) & (phi_cam <= fov_x_rad / 2) &
            (theta_cam >= -fov_y_rad / 2) & (theta_cam <= fov_y_rad / 2)
        )

        # Map the spherical coordinates to pixel coordinates in the camera image
        x_cam = ((phi_cam[fov_mask] + fov_x_rad / 2) / fov_x_rad) * (w - 1)
        y_cam = ((theta_cam[fov_mask] + fov_y_rad / 2) / fov_y_rad) * (h - 1)

        # Panorama coordinates corresponding to these directions
        pano_coords = np.argwhere(fov_mask.reshape(total_width, total_height))

        # Sample the camera image at the computed coordinates
        x_cam = x_cam.astype(np.float32)
        y_cam = y_cam.astype(np.float32)

        sampled_pixels = cv2.remap(
            image,
            x_cam.reshape(-1, 1),
            y_cam.reshape(-1, 1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Place the sampled pixels into the panorama
        panorama[pano_coords[:, 0], pano_coords[:, 1]] = sampled_pixels.reshape(-1, 3)
        filled_mask[pano_coords[:, 0], pano_coords[:, 1]] = True

    return panorama