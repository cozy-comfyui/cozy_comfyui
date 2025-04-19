"""Image processing support for format conversions."""

import urllib
from typing import List, Tuple

import cv2
import requests
import numpy as np
from PIL import Image, ImageOps

from .. import \
    logger

from . import \
    ImageType

from .convert import \
    pil_to_cv

from .misc import \
    image_normalize, image_mask

# ==============================================================================
# === SUPPPORT ===
# ==============================================================================

def image_load_exr(url: str) -> Tuple[ImageType, ...]:
    """
    exr_file     = OpenEXR.InputFile(url)
    exr_header   = exr_file.header()
    r,g,b = exr_file.channels("RGB", pixel_type=Imath.PixelType(Imath.PixelType.FLOAT) )

    dw = exr_header[ "dataWindow" ]
    w  = dw.max.x - dw.min.x + 1
    h  = dw.max.y - dw.min.y + 1

    image = np.ones( (h, w, 4), dtype = np.float32 )
    image[:, :, 0] = np.core.multiarray.frombuffer( r, dtype = np.float32 ).reshape(h, w)
    image[:, :, 1] = np.core.multiarray.frombuffer( g, dtype = np.float32 ).reshape(h, w)
    image[:, :, 2] = np.core.multiarray.frombuffer( b, dtype = np.float32 ).reshape(h, w)
    return create_optix_image_2D( w, h, image.flatten() )
    """
    pass

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
    return img, mask

def image_load_from_url(url: str, stream:bool=True) -> Tuple[ImageType, ...]:
    """Creates a CV2 BGR image from a url."""
    image = None
    mask = None
    try:
        image  = urllib.request.urlopen(url)
        image = np.asarray(bytearray(image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except:
        try:
            image = Image.open(requests.get(url, stream=stream).raw)
            image = pil_to_cv(image)
        except Exception as e:
            logger.error(str(e))

    if image is None:
        raise Exception(f"No file found at {url}")

    mask = image_mask(image)
    return image, mask

def image_save_gif(fpath:str, images: List[Image.Image], fps: int=0,
                loop:int=0, optimize:bool=False) -> None:

    fps = min(50, max(1, fps))
    images[0].save(
        fpath,
        append_images=images[1:],
        duration=3, # int(100.0 / fps),
        loop=loop,
        optimize=optimize,
        save_all=True
    )

def image_load_data(data: str) -> ImageType:
    img = ImageOps.exif_transpose(data)
    return pil_to_cv(img)
