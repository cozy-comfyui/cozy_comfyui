"""Image cropping support module."""

from typing import List, Tuple

import cv2
import numpy as np

from . import \
    ImageType, Coord2D_Float

from .convert import \
    image_convert, image_mask, image_mask_add

# ==============================================================================
# === SUPPPORT ===
# ==============================================================================

def image_crop(image: ImageType, width:int=None, height:int=None, offset:Tuple[float, float]=(0, 0)) -> ImageType:
    h, w = image.shape[:2]
    width = width if width is not None else w
    height = height if height is not None else h
    x, y = offset
    x = max(0, min(width, x))
    y = max(0, min(width, y))
    x2 = max(0, min(width, x + width))
    y2 = max(0, min(height, y + height))
    points = [(x, y), (x2, y), (x2, y2), (x, y2)]
    return image_crop_polygonal(image, points)

def image_crop_center(image: ImageType, width:int=None, height:int=None) -> ImageType:
    """Helper crop function to find the "center" of the area of interest."""
    h, w = image.shape[:2]
    cx = w // 2
    cy = h // 2
    width = w if width is None else width
    height = h if height is None else height
    x1 = max(0, int(cx - width // 2))
    y1 = max(0, int(cy - height // 2))
    x2 = min(w, int(cx + width // 2)) - 1
    y2 = min(h, int(cy + height // 2)) - 1
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return image_crop_polygonal(image, points)

def image_crop_polygonal(image: ImageType, points: List[Coord2D_Float]) -> ImageType:
    cc = image.shape[2] if image.ndim == 3 else 1
    height, width = image.shape[:2]
    point_mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    point_mask = cv2.fillPoly(point_mask, [points], 255)
    x, y, w, h = cv2.boundingRect(point_mask)
    cropped_image = cv2.resize(image[y:y+h, x:x+w], (w, h)).astype(np.uint8)
    # Apply the mask to the cropped image
    point_mask_cropped = cv2.resize(point_mask[y:y+h, x:x+w], (w, h))
    if cc == 4:
        mask = image_mask(image, 0)
        alpha_channel = cv2.resize(mask[y:y+h, x:x+w], (w, h))
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2BGR)
        cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=point_mask_cropped)
        return image_mask_add(cropped_image, alpha_channel)
    elif cc == 1:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
        cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=point_mask_cropped)
        return image_convert(cropped_image, cc)
    return cv2.bitwise_and(cropped_image, cropped_image, mask=point_mask_cropped)
