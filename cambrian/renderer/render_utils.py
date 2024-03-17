import cv2
import numpy as np

import mujoco as mj


def resize_with_aspect_fill(image: np.ndarray, width: int, height: int):
    # original_width, original_height = image.shape[:2]
    original_height, original_width = image.shape[:2]
    ratio_original = original_width / original_height
    ratio_new = width / height

    # Resize the image while maintaining the aspect ratio
    border_type = cv2.BORDER_CONSTANT
    if ratio_original > ratio_new:
        # Original is wider relative to the new size
        resize_height = round(width / ratio_original)
        resized_image = cv2.resize(image, (width, resize_height))
        top = (height - resize_height) // 2
        bottom = height - resize_height - top
        result = cv2.copyMakeBorder(resized_image, top, bottom, 0, 0, border_type)
    else:
        # Original is taller relative to the new size
        # import pdb; pdb.set_trace()
        resize_width = round(height * ratio_original)
        resized_image = cv2.resize(image, (resize_width, height))
        left = (width - resize_width) // 2
        right = width - resize_width - left
        result = cv2.copyMakeBorder(resized_image, 0, 0, left, right, border_type)

    return result


def convert_depth_to_rgb(model: mj.MjModel, depth: np.ndarray) -> np.ndarray:
    """https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/renderer.py"""
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
