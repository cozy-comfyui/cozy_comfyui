"""Image colorpsace support."""

import cv2
import numpy as np

from . import \
    PixelType, ImageType

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def gamma2linear(image: ImageType) -> ImageType:
    """Gamma correction for old PCs/CRT monitors"""
    return np.power(image, 2.2)

def linear2gamma(image: ImageType) -> ImageType:
    """Inverse gamma correction for old PCs/CRT monitors"""
    return np.power(np.clip(image, 0., 1.), 1.0 / 2.2)

def sRGB2Linear(image: ImageType) -> ImageType:
    """Convert sRGB to linearRGB, removing the gamma correction.
    Works for grayscale, RGB, or RGBA images.
    """
    image = image.astype(float) / 255.0

    # If the image has an alpha channel, separate it
    if image.shape[-1] == 4:
        rgb = image[..., :3]
        alpha = image[..., 3]
    else:
        rgb = image
        alpha = None

    gamma = ((rgb + 0.055) / 1.055) ** 2.4
    scale = rgb / 12.92
    rgb = np.where(rgb > 0.04045, gamma, scale)

    # Recombine the alpha channel if it exists
    if alpha is not None:
        image = np.concatenate((rgb, alpha[..., np.newaxis]), axis=-1)
    else:
        image = rgb
    return (image * 255).astype(np.uint8)

def linear2sRGB(image: ImageType) -> ImageType:
    """Convert linearRGB to sRGB, applying the gamma correction.
    Works for grayscale, RGB, or RGBA images.
    """
    image = image.astype(float) / 255.0

    # If the image has an alpha channel, separate it
    if image.shape[-1] == 4:
        rgb = image[..., :3]
        alpha = image[..., 3]
    else:
        rgb = image
        alpha = None

    higher = 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055
    lower = rgb * 12.92
    rgb = np.where(rgb > 0.0031308, higher, lower)

    # Recombine the alpha channel if it exists
    if alpha is not None:
        image = np.concatenate((rgb, alpha[..., np.newaxis]), axis=-1)
    else:
        image = rgb
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)

def rgb_to_hsv(rgb_color: PixelType) -> PixelType:
    return cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0, 0]

def hsv_to_rgb(hsl_color: PixelType) -> PixelType:
    return cv2.cvtColor(np.uint8([[hsl_color]]), cv2.COLOR_HSV2RGB)[0, 0]
