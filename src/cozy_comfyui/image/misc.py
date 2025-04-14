"""Image processing support module for handling various image formats and conversions."""

from enum import Enum
from typing import List, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image, ImageOps

from .. import \
    IMAGE_SIZE_MIN, \
    RGBAMaskType

from . import \
    ImageType

from .convert import image_mask

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumInterpolation(Enum):
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4
    LINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    NEAREST_EXACT = cv2.INTER_NEAREST_EXACT

# ==============================================================================
# === SUPPPORT ===
# ==============================================================================

def image_lerp(imageA: ImageType, imageB:ImageType, mask:ImageType=None,
               alpha:float=1.) -> ImageType:

    imageA = imageA.astype(np.float32)
    imageB = imageB.astype(np.float32)

    # establish mask
    alpha = np.clip(alpha, 0, 1)
    if mask is None:
        height, width = imageA.shape[:2]
        mask = np.ones((height, width, 1), dtype=np.float32)
    else:
        # normalize the mask
        mask = mask.astype(np.float32)
        mask = (mask - mask.min()) / (mask.max() - mask.min()) * alpha

    # LERP
    imageA = cv2.multiply(1. - mask, imageA)
    imageB = cv2.multiply(mask, imageB)
    imageA = (cv2.add(imageA, imageB) / 255. - 0.5) * 2.0
    imageA = (imageA * 255).astype(imageA.dtype)
    return np.clip(imageA, 0, 255)

def image_load(url: str) -> Tuple[ImageType, ...]:
    if url.lower().startswith("http"):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        img = image_normalize(img)
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.ndim < 3:
            img = np.expand_dims(img, -1)
    else:
        try:
            img = cv2.imread(url, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"{url} could not be loaded.")

            img = image_normalize(img)
            if img.ndim == 3:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.ndim < 3:
                img = np.expand_dims(img, -1)

        except Exception:
            try:
                img = Image.open(url)
                img = ImageOps.exif_transpose(img)
                img = np.array(img)
                if img.dtype != np.uint8:
                    img = np.clip(np.array(img * 255), 0, 255).astype(dtype=np.uint8)
            except Exception as e:
                raise Exception(f"Error loading image: {e}")

    if img is None:
        raise Exception(f"No file found at {url}")

    mask = image_mask(img)
    """
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = mask / 255.0
        img[..., :3] = img[..., :3] * alpha[..., None]
        img[:,:,3] = mask
    """

    return img, mask

def image_minmax(image: List[ImageType]) -> Tuple[int, int, int, int]:
    h_min = w_min = 100000000000
    h_max = w_max = IMAGE_SIZE_MIN
    for img in image:
        if img is None:
            continue
        h, w = img.shape[:2]
        h_max = max(h, h_max)
        w_max = max(w, w_max)
        h_min = min(h, h_min)
        w_min = min(w, w_min)

    # x,y - x+width, y+height
    return w_min, h_min, w_max, h_max

def image_normalize(image: ImageType) -> ImageType:
    image = image.astype(np.float32)
    img_min = np.min(image)
    img_max = np.max(image)
    if img_min == img_max:
        return np.zeros_like(image)
    image = (image - img_min) / (img_max - img_min)
    return (image * 255).astype(np.uint8)

def image_resize(image: ImageType, width: int, height: int, sample: EnumInterpolation) -> ImageType:
    return cv2.resize(image, (width, height), interpolation=sample)

def image_stack(images: List[ImageType] ) -> RGBAMaskType:
    return [torch.stack(i) for i in zip(*images)]
