"""Rendering utilities."""

from typing import Dict, Tuple

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


class CubeToEquirectangularConverter:
    def __init__(self, size, w_img, h_img):
        """
        Initialize the CubeToEquirectangularConverter.

        Parameters:
        - size: Tuple[int, int] -> (width, height) of the output equirectangular image
        - w_img: int -> Width of each input cube face image
        - h_img: int -> Height of each input cube face image
        """
        # Output dimensions
        self.width, self.height = size
        self.w_img = w_img
        self.h_img = h_img

        # Preallocate the output buffer as float32
        self.output = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Generate coordinate grids
        theta = (
            np.linspace(0, 2 * np.pi, self.width, endpoint=False) - np.pi
        )  # Longitude from -π to π
        phi = np.linspace(
            -np.pi / 2, np.pi / 2, self.height
        )  # Latitude from -π/2 to π/2
        self.theta, self.phi = np.meshgrid(theta, phi)

        # Convert spherical coordinates to Cartesian coordinates
        x = np.cos(self.phi) * np.sin(self.theta)
        y = np.sin(self.phi)
        z = np.cos(self.phi) * np.cos(self.theta)

        # Determine which face each pixel corresponds to
        abs_x = np.abs(x)
        abs_y = np.abs(y)
        abs_z = np.abs(z)

        # Initialize face indices and UV coordinates
        face_indices = np.zeros((self.height, self.width), dtype=np.int32)
        u = np.zeros((self.height, self.width), dtype=np.float32)
        v = np.zeros((self.height, self.width), dtype=np.float32)

        # Define face indices:
        # 0: Left, 1: Front, 2: Right, 3: Back
        # Left face (x negative)
        mask_left = (abs_x >= abs_y) & (abs_x >= abs_z) & (x < 0)
        face_indices[mask_left] = 0
        u[mask_left] = z[mask_left] / abs_x[mask_left]
        v[mask_left] = y[mask_left] / abs_x[mask_left]

        # Front face (z positive)
        mask_front = (abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0)
        face_indices[mask_front] = 1
        u[mask_front] = x[mask_front] / abs_z[mask_front]
        v[mask_front] = y[mask_front] / abs_z[mask_front]

        # Right face (x positive)
        mask_right = (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0)
        face_indices[mask_right] = 2
        u[mask_right] = -z[mask_right] / abs_x[mask_right]
        v[mask_right] = y[mask_right] / abs_x[mask_right]

        # Back face (z negative)
        mask_back = (abs_z >= abs_x) & (abs_z >= abs_y) & (z < 0)
        face_indices[mask_back] = 3
        u[mask_back] = -x[mask_back] / abs_z[mask_back]
        v[mask_back] = y[mask_back] / abs_z[mask_back]

        # Map UV coordinates to pixel indices
        u_img = ((u + 1) / 2) * (self.w_img - 1)
        v_img = ((1 - (v + 1) / 2)) * (self.h_img - 1)  # Flip v-axis

        # Precompute the adjusted map_x by accounting for face indices
        # Each face is placed horizontally in the combined image
        # Combined image width = 4 * w_img
        self.map_x = u_img + face_indices * self.w_img
        self.map_y = v_img

        # Stack per-face maps
        self.map_x = self.map_x.astype(np.float32)
        self.map_y = self.map_y.astype(np.float32)

        # Precompute the crop indices
        self.crop_start = (self.height - self.h_img) // 2
        self.crop_end = self.crop_start + self.h_img

    def convert(self, images):
        """
        Convert cube faces to an equirectangular image.

        Parameters:
        - images: list of four images [left, front, right, back]
                  Each image should be a NumPy array of shape (h_img, w_img, 3) with
                float32 values in [0, 1]

        Returns:
        - Equirectangular image as a NumPy array of shape (h_img, width, 3) with
        float32 values in [0, 1]
        """

        # Ensure all input images are float32 and in the range [0, 1]
        images = [
            image.transpose(1, 0, 2).astype(np.float32) for image in images
        ]  # Transpose if needed

        # Combine all faces into a single image arranged horizontally: left | front |
        # right | back
        combined_image = np.hstack(images)  # Shape: (h_img, 4 * w_img, 3)

        # Perform a single remapping operation
        remapped = cv2.remap(
            combined_image,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Crop the vertical center portion to match the original image height
        output_cropped = remapped[self.crop_start : self.crop_end, :, :]

        # Optionally flip the image vertically if needed
        output_final = np.flipud(output_cropped).transpose(1, 0, 2)

        return output_final
