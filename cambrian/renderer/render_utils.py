"""Rendering utilities."""

from typing import Dict, List, Tuple

import cv2
import mujoco as mj
import numpy as np


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
        images: List of images from the cameras, each with shape (width, height,
            channels).
        yaw_angles: List of yaw angles (in degrees) corresponding to each camera image.
        fov_x: Horizontal field of view of each camera in degrees.
        fov_y: Vertical field of view of each camera in degrees.
        total_resolution: Resolution (width, height) of the resulting panorama.

    Returns:
        The spherical panorama image as a NumPy array with shape (width, height,
            channels).
    """
    import numpy as np

    # Unpack total resolution
    pano_width, pano_height = total_resolution
    channels = images[0].shape[2]

    # Initialize the panorama image
    pano_image = np.zeros((pano_width, pano_height, channels), dtype=images[0].dtype)

    # Compute min_lon and max_lon based on the cameras' yaw angles and FOVs
    half_fov_x = fov_x / 2.0
    min_lon = min(yaw - half_fov_x for yaw in yaw_angles)
    max_lon = max(yaw + half_fov_x for yaw in yaw_angles)

    # Normalize longitudes to be within [-180, 180]
    min_lon = ((min_lon + 180) % 360) - 180
    max_lon = ((max_lon + 180) % 360) - 180

    # Handle wrap-around if necessary
    if min_lon > max_lon:
        max_lon += 360

    # Compute min_lat and max_lat based on FOV
    half_fov_y = fov_y / 2.0
    min_lat = -half_fov_y
    max_lat = half_fov_y

    # Create coordinate grids for the panorama (note: X is width, Y is height)
    X_pano, Y_pano = np.meshgrid(
        np.arange(pano_width), np.arange(pano_height), indexing="ij"
    )

    # Compute theta (longitude) and phi (latitude) for each pixel in the panorama
    theta = (X_pano / (pano_width - 1)) * (max_lon - min_lon) + min_lon  # Degrees
    phi = (Y_pano / (pano_height - 1)) * (max_lat - min_lat) + min_lat  # Degrees

    # Ensure theta is within [-180, 180]
    theta = ((theta + 180) % 360) - 180

    # Iterate over each camera image
    for i, image_i in enumerate(images):
        yaw = yaw_angles[i]
        # Normalize yaw to [-180, 180]
        yaw = ((yaw + 180) % 360) - 180

        # Handle wrap-around for delta_theta
        delta_theta = ((theta - yaw + 180) % 360) - 180

        # Create a mask for pixels within the camera's field of view
        mask = (
            (np.abs(delta_theta) <= half_fov_x)
            & (phi >= -half_fov_y)
            & (phi <= half_fov_y)
        )

        # Skip if no pixels are within the field of view
        if not np.any(mask):
            continue

        # Normalize delta_theta and phi to [0, 1] within the camera's FOV
        u = (delta_theta + half_fov_x) / fov_x
        v = (phi + half_fov_y) / fov_y

        # Map normalized coordinates to pixel indices in the input image
        # Images are in shape (width, height, channels)
        x_img = u * (image_i.shape[0] - 1)
        y_img = v * (image_i.shape[1] - 1)

        # Convert to integer indices
        x_img_indices = x_img.astype(int)
        y_img_indices = y_img.astype(int)

        # Clip indices to valid range
        x_img_indices = np.clip(x_img_indices, 0, image_i.shape[0] - 1)
        y_img_indices = np.clip(y_img_indices, 0, image_i.shape[1] - 1)

        # Flatten indices and apply mask
        pano_indices_flat = np.where(mask)
        x_img_indices_flat = x_img_indices[mask]
        y_img_indices_flat = y_img_indices[mask]

        # Get pixel values from the input image
        pixel_values = image_i[x_img_indices_flat, y_img_indices_flat, :]

        # Assign pixel values to the panorama image
        pano_image[pano_indices_flat[0], pano_indices_flat[1], :] = pixel_values

    return pano_image
