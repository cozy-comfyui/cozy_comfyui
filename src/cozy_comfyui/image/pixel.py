"""Pixel processing support."""

from enum import Enum

import numpy as np

from . import \
    PixelType, \
    EnumImageType

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumGrayscaleCrunch(Enum):
    LOW = 0
    HIGH = 1
    MEAN = 2

class EnumPrecision(Enum):
    FLOAT = 0
    INT = 1

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def pixel_convert(color:PixelType, size:int=4, alpha:int=255) -> PixelType:
    """Convert X channel pixel into Y channel pixel."""
    if (cc := len(color)) == size:
        return color
    if size > 2:
        color += (0,) * (3 - cc)
        if size == 4:
            color += (alpha,)
        return color
    return color[0]

def pixel_eval(color: PixelType,
            target: EnumImageType=EnumImageType.RGBA,
            precision:EnumPrecision=EnumPrecision.INT,
            crunch:EnumGrayscaleCrunch=EnumGrayscaleCrunch.MEAN) -> tuple[PixelType] | PixelType:
    """Evaluates R(GB)(A) pixels in range (0-255) into target target pixel type."""

    def parse_single_color(c: PixelType) -> PixelType:
        if not isinstance(c, int):
            c = np.clip(c, 0, 1)
            if precision == EnumPrecision.INT:
                c = int(c * 255)
        else:
            c = np.clip(c, 0, 255)
            if precision == EnumPrecision.FLOAT:
                c /= 255
        return c

    # make sure we are an RGBA value already
    if isinstance(color, (float, int)):
        color = tuple([parse_single_color(color)])
    elif isinstance(color, (set, tuple, list)):
        color = tuple([parse_single_color(c) for c in color])

    if target == EnumImageType.GRAYSCALE:
        alpha = 1
        if len(color) > 3:
            alpha = color[3]
            if precision == EnumPrecision.INT:
                alpha /= 255
            color = color[:3]
        match crunch:
            case EnumGrayscaleCrunch.LOW:
                val = min(color) * alpha
            case EnumGrayscaleCrunch.HIGH:
                val = max(color) * alpha
            case EnumGrayscaleCrunch.MEAN:
                val = np.mean(color) * alpha
        if precision == EnumPrecision.INT:
            val = int(val)
        return val

    if len(color) < 3:
        color += (0,) * (3 - len(color))

    if target in [EnumImageType.RGB, EnumImageType.BGR]:
        color = color[:3]
        if target == EnumImageType.BGR:
            color = color[::-1]
        return color

    if len(color) < 4:
        color += (255,)

    if target == EnumImageType.BGRA:
        color = tuple(color[2::-1]) + tuple([color[-1]])
    return color

def pixel_hsv_adjust(color:PixelType, hue:int=0, saturation:int=0, value:int=0,
                     mod_color:bool=True, mod_sat:bool=False,
                     mod_value:bool=False) -> PixelType:
    """Adjust an HSV type pixel.
    OpenCV uses... H: 0-179, S: 0-255, V: 0-255"""
    hsv = [0, 0, 0]
    hsv[0] = (color[0] + hue) % 180 if mod_color else np.clip(color[0] + hue, 0, 180)
    hsv[1] = (color[1] + saturation) % 255 if mod_sat else np.clip(color[1] + saturation, 0, 255)
    hsv[2] = (color[2] + value) % 255 if mod_value else np.clip(color[2] + value, 0, 255)
    return hsv
