""" Jovimetrix - Shape Operation Support """

from enum import Enum

from PIL import Image, ImageDraw

from . import \
    PixelType

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumShapes(Enum):
    CIRCLE = 0
    SQUARE = 1
    #ELLIPSE = 2
    #RECTANGLE = 3
    POLYGON = 4

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def shape_ellipse(width: int, height: int, sizeX:float=1., sizeY:float=1.,
                  fill:PixelType=255, back:PixelType=0) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), back)
    ImageDraw.Draw(image).ellipse(xy, fill=fill)
    return image

def shape_quad(width: int, height: int, sizeX:float=1., sizeY:float=1.,
               fill:PixelType=255, back:PixelType=0) -> Image:
    sizeX = max(0.5, sizeX / 2 + 0.5)
    sizeY = max(0.5, sizeY / 2 + 0.5)
    xy = [(width * (1. - sizeX), height * (1. - sizeY)),(width * sizeX, height * sizeY)]
    image = Image.new("RGB", (width, height), back)
    ImageDraw.Draw(image).rectangle(xy, fill=fill)
    return image

def shape_polygon(width: int, height: int, size: float=1., sides: int=3,
                  fill:PixelType=255, back:PixelType=0) -> Image:
    size = max(0.00001, size)
    r = min(width, height) * size * 0.5
    xy = (width * 0.5, height * 0.5, r)
    image = Image.new("RGB", (width, height), back)
    d = ImageDraw.Draw(image)
    d.regular_polygon(xy, sides, fill=fill)
    return image
