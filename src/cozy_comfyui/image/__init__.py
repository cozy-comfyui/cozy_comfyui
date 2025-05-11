"""Image processing support for image formats and conversions."""

from __future__ import annotations

import math
from enum import Enum
from typing import List, Tuple, TypeAlias, Union

import numpy as np
from PIL import Image

# ==============================================================================
# === TYPE ===
# ==============================================================================

# Color type definitions
RGB_Int: TypeAlias = Tuple[int, int, int]
RGBA_Int: TypeAlias = Tuple[int, int, int, int]
RGB_Float: TypeAlias = Tuple[float, float, float]
RGBA_Float: TypeAlias = Tuple[float, float, float, float]

# Coordinate type definitions
Coord2D_Int: TypeAlias = Tuple[int, int]
Coord3D_Int: TypeAlias = Tuple[int, int, int]
Coord2D_Float: TypeAlias = Tuple[float, float]
Coord3D_Float: TypeAlias = Tuple[float, float, float]

ImageType: TypeAlias = np.ndarray
PixelType: TypeAlias = Union[int, float, RGB_Int, RGBA_Int, RGB_Float, RGBA_Float]

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumImageType(Enum):
    GRAYSCALE = 0
    RGB = 10
    RGBA = 20
    BGR = 30
    BGRA = 40

# ==============================================================================
# === CONSTANT ===
# ==============================================================================

HALFPI: float = math.pi / 2
TAU: float = math.pi * 2

IMAGE_FORMATS: List[str] = [
    ext for ext, fmt in Image.registered_extensions().items()
    if fmt in Image.OPEN
]
