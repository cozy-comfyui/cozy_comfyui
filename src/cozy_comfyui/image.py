"""Image Support"""

import math
import base64
from io import BytesIO
from typing import List, Tuple, Union

import cv2
import torch
import requests
import numpy as np
from PIL import Image, ImageOps

# ==============================================================================
# === TYPE ===
# ==============================================================================

COZY_TYPE_iRGB  = Tuple[int, int, int]
COZY_TYPE_iRGBA = Tuple[int, int, int, int]
COZY_TYPE_fRGB  = Tuple[float, float, float]
COZY_TYPE_fRGBA = Tuple[float, float, float, float]

COZY_TYPE_fCOORD2D = Tuple[float, float]
COZY_TYPE_fCOORD3D = Tuple[float, float, float]
COZY_TYPE_iCOORD2D = Tuple[int, int]
COZY_TYPE_iCOORD3D = Tuple[int, int, int]

COZY_TYPE_IMAGE = Union[np.ndarray, torch.Tensor]
COZY_TYPE_PIXEL = Union[int, float,
                        COZY_TYPE_iRGB, COZY_TYPE_iRGBA,
                        COZY_TYPE_fRGB, COZY_TYPE_fRGBA]

# ==============================================================================
# === CONSTANT ===
# ==============================================================================

MIN_IMAGE_SIZE: int = 32
HALFPI: float = math.pi / 2
TAU: float = math.pi * 2

IMAGE_FORMATS: List[str] = [
    ext for ext, fmt in Image.registered_extensions().items()
    if fmt in Image.OPEN
]

# ==============================================================================
# === CONVERSION ===
# ==============================================================================

def b64_to_tensor(base64str: str) -> torch.Tensor:
    img = base64.b64decode(base64str)
    img = Image.open(BytesIO(img))
    img = ImageOps.exif_transpose(img)
    return pil_to_tensor(img)

def b64_to_pil(base64_string):
    prefix, base64_data = base64_string.split(",", 1)
    image_data = base64.b64decode(base64_data)
    image_stream = BytesIO(image_data)
    return Image.open(image_stream)

def b64_to_cv(base64_string) -> COZY_TYPE_IMAGE:
    _, data = base64_string.split(",", 1)
    data = base64.b64decode(data)
    data = BytesIO(data)
    data = Image.open(data)
    data = np.array(data)
    return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

def cv_to_pil(image: COZY_TYPE_IMAGE) -> Image.Image:
    """Convert a CV2 image to a PIL Image."""
    if image.ndim > 2:
        cc = image.shape[2]
        if cc == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif cc == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image = np.squeeze(image, axis=-1)
    return Image.fromarray(image)

def cv_to_tensor(image: COZY_TYPE_IMAGE, grayscale: bool=False) -> torch.Tensor:
    """Convert a CV2 image to a torch tensor, with handling for grayscale/mask."""
    if grayscale or image.ndim < 3 or image.shape[2] == 1:
        if image.ndim < 3:
            image = np.expand_dims(image, -1)

        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = np.squeeze(image, axis=-1)

    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image)

def cv_to_tensor_full(image: COZY_TYPE_IMAGE, matte:COZY_TYPE_PIXEL=(0,0,0,255)) -> Tuple[torch.Tensor, ...]:

    rgba = image_convert(image, 4)
    rgb = rgba[...,:3]
    mask = rgba[...,3]
    rgba = torch.from_numpy(rgba.astype(np.float32) / 255.0)
    rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    mask = torch.from_numpy(mask.astype(np.float32) / 255.0)
    return rgba, rgb, mask

def pil_to_cv(image: Image.Image) -> COZY_TYPE_IMAGE:
    """Convert a PIL Image to a CV2 Matrix."""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a Torch Tensor."""
    image = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0)

def tensor_to_cv(tensor: torch.Tensor, invert_mask:bool=True) -> COZY_TYPE_IMAGE:
    """Convert a torch Tensor to a numpy ndarray."""
    if tensor.ndim > 3:
        raise Exception("Tensor is batch of tensors")

    if tensor.ndim < 3:
        tensor = tensor.unsqueeze(-1)

    if tensor.shape[2] == 1 and invert_mask:
        tensor = 1. - tensor

    tensor = tensor.cpu().numpy()
    return np.clip(255.0 * tensor, 0, 255).astype(np.uint8)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a torch Tensor to a PIL Image.
    Tensor should be HxWxC [no batch].
    """
    tensor = tensor.cpu().numpy().squeeze()
    tensor = np.clip(255. * tensor, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

# ==============================================================================
# === IMAGE ===
# ==============================================================================

def image_convert(image: COZY_TYPE_IMAGE, channels: int, width: int=None, height: int=None,
                  matte: Tuple[int, ...]=(0, 0, 0, 255)) -> COZY_TYPE_IMAGE:
    """Force image format to a specific number of channels.
    Args:
        image (COZY_TYPE_IMAGE): Input image.
        channels (int): Desired number of channels (1, 3, or 4).
        width (int): Desired width. `None` means leave unchanged.
        height (int): Desired height. `None` means leave unchanged.
        matte (tuple): RGBA color to use as background color for transparent areas.
    Returns:
        COZY_TYPE_IMAGE: Image with the specified number of channels.
    """
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    if (cc := image.shape[2]) != channels:
        if   cc == 1 and channels == 3:
            image = np.repeat(image, 3, axis=2)
        elif cc == 1 and channels == 4:
            rgb = np.repeat(image, 3, axis=2)
            alpha = np.full(image.shape[:2] + (1,), matte[3], dtype=image.dtype)
            image = np.concatenate([rgb, alpha], axis=2)
        elif cc == 3 and channels == 1:
            image = np.mean(image, axis=2, keepdims=True).astype(image.dtype)
        elif cc == 3 and channels == 4:
            alpha = np.full(image.shape[:2] + (1,), matte[3], dtype=image.dtype)
            image = np.concatenate([image, alpha], axis=2)
        elif cc == 4 and channels == 1:
            rgb = image[..., :3]
            alpha = image[..., 3:4] / 255.0
            image = (np.mean(rgb, axis=2, keepdims=True) * alpha).astype(image.dtype)
        elif cc == 4 and channels == 3:
            image = image[..., :3]

    # Resize if width or height is specified
    h, w = image.shape[:2]
    new_width = width if width is not None else w
    new_height = height if height is not None else h
    if (new_width, new_height) != (w, h):
        # Create a new image with the matte color
        new_image = np.full((new_height, new_width, channels), matte[:channels], dtype=image.dtype)
        paste_x = (new_width - w) // 2
        paste_y = (new_height - h) // 2
        new_image[paste_y:paste_y+h, paste_x:paste_x+w] = image[:h, :w]
        image = new_image

    return image

def image_lerp(imageA: COZY_TYPE_IMAGE, imageB:COZY_TYPE_IMAGE, mask:COZY_TYPE_IMAGE=None,
               alpha:float=1.) -> COZY_TYPE_IMAGE:

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

def image_load(url: str) -> Tuple[COZY_TYPE_IMAGE, COZY_TYPE_IMAGE]:
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

def image_mask(image: COZY_TYPE_IMAGE, color: COZY_TYPE_PIXEL = 255) -> COZY_TYPE_IMAGE:
    """Create a mask from the image, preserving transparency.

    Args:
        image (COZY_TYPE_IMAGE): Input image, assumed to be 2D or 3D (with or without alpha channel).
        color (COZY_TYPE_PIXEL): Value to fill the mask (default is 255).

    Returns:
        COZY_TYPE_IMAGE: Mask of the image, either the alpha channel or a full mask of the given color.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., 3]

    h, w = image.shape[:2]
    return np.ones((h, w), dtype=np.uint8) * color

def image_mask_add(image:COZY_TYPE_IMAGE, mask:COZY_TYPE_IMAGE=None, alpha:float=255) -> COZY_TYPE_IMAGE:
    """Put custom mask into an image. If there is no mask, alpha is applied.
    Images are expanded to 4 channels.
    Existing 4 channel images with no mask input just return themselves.
    """
    image = image_convert(image, 4)
    mask = image_mask(image, alpha) if mask is None else image_convert(mask, 1)
    image[..., 3] = mask if mask.ndim == 2 else mask[:, :, 0]
    return image

def image_matte(image: COZY_TYPE_IMAGE, color: COZY_TYPE_iRGBA=(0,0,0,255), width: int=None, height: int=None) -> COZY_TYPE_IMAGE:
    """
    Puts an RGBA image atop a colored matte expanding or clipping the image if requested.

    Args:
        image (COZY_TYPE_IMAGE): The input RGBA image.
        color (COZY_TYPE_iRGBA): The color of the matte as a tuple (R, G, B, A).
        width (int, optional): The width of the matte. Defaults to the image width.
        height (int, optional): The height of the matte. Defaults to the image height.

    Returns:
        COZY_TYPE_IMAGE: Composited RGBA image on a matte with original alpha channel.
    """

    #if image.ndim != 4 or image.shape[2] != 4:
    #    return image

    # Determine the dimensions of the image and the matte
    image_height, image_width = image.shape[:2]
    width = width or image_width
    height = height or image_height

    # Create a solid matte with the specified color
    matte = np.full((height, width, 4), color, dtype=np.uint8)

    # Extract the alpha channel from the image
    alpha = None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3] / 255.0

    # Calculate the center position for the image on the matte
    x_offset = (width - image_width) // 2
    y_offset = (height - image_height) // 2

    if alpha is not None:
        # Place the image onto the matte using the alpha channel for blending
        for c in range(0, 3):
            matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] = \
                (1 - alpha) * matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] + \
                alpha * image[:, :, c]

        # Set the alpha channel of the matte to the maximum of the matte's and the image's alpha
        matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, 3] = \
            np.maximum(matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, 3], image[:, :, 3])
    else:
        image = image[y_offset:y_offset + image_height, x_offset:x_offset + image_width, :]
    return matte

def image_minmax(image:List[COZY_TYPE_IMAGE]) -> Tuple[int, int, int, int]:
    h_min = w_min = 100000000000
    h_max = w_max = MIN_IMAGE_SIZE
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

def image_normalize(image: COZY_TYPE_IMAGE) -> COZY_TYPE_IMAGE:
    image = image.astype(np.float32)
    img_min = np.min(image)
    img_max = np.max(image)
    if img_min == img_max:
        return np.zeros_like(image)
    image = (image - img_min) / (img_max - img_min)
    return (image * 255).astype(np.uint8)
