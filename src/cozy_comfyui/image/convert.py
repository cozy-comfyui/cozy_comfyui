"""Image processing support for format conversions."""

import base64
from io import BytesIO
from typing import Tuple

import cv2
import torch
import numpy as np
from PIL import Image, ImageOps

from .. import \
    TensorType

from ..image import \
    PixelType, RGBA_Int, ImageType

# ==============================================================================
# === CONVERSION ===
# ==============================================================================

def image_convert(image: ImageType, channels: int,
                  width: int=None, height: int=None,
                  matte: RGBA_Int=(0, 0, 0, 255)) -> ImageType:
    """Force image format to a specific number of channels.
    Args:
        image (ImageType): Input image.
        channels (int): Desired number of channels (1, 3, or 4).
        width (int): Desired width. `None` means leave unchanged.
        height (int): Desired height. `None` means leave unchanged.
        matte (tuple): RGBA color to use as background color for transparent areas.
    Returns:
        ImageType: Image with the specified number of channels.
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

def b64_to_tensor(base64str: str) -> TensorType:
    img = base64.b64decode(base64str)
    img = Image.open(BytesIO(img))
    img = ImageOps.exif_transpose(img)
    return pil_to_tensor(img)

def b64_to_pil(base64_string):
    prefix, base64_data = base64_string.split(",", 1)
    image_data = base64.b64decode(base64_data)
    image_stream = BytesIO(image_data)
    return Image.open(image_stream)

def b64_to_cv(base64_string) -> ImageType:
    _, data = base64_string.split(",", 1)
    data = base64.b64decode(data)
    data = BytesIO(data)
    data = Image.open(data)
    data = np.array(data)
    return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

def bgr_to_hsv(bgr_color: PixelType) -> PixelType:
    return cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0, 0]

def bgr_to_image(image: ImageType, alpha: ImageType=None, gray: bool=False) -> ImageType:
    """Restore image with alpha, if any, and converting to grayscale (optional)."""
    if gray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_mask_add(image, alpha)

def hsv_to_bgr(hsl_color: PixelType) -> PixelType:
    return cv2.cvtColor(np.uint8([[hsl_color]]), cv2.COLOR_HSV2BGR)[0, 0]

def cv_to_pil(image: ImageType) -> Image.Image:
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

def cv_to_tensor(image: ImageType, grayscale: bool=False) -> TensorType:
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

def cv_to_tensor_full(image: ImageType, matte:RGBA_Int=(0,0,0,255)) -> Tuple[TensorType, ...]:

    rgba = image_convert(image, 4, matte=matte)
    rgb = rgba[...,:3]
    mask = rgba[...,3]
    rgba = torch.from_numpy(rgba.astype(np.float32) / 255.0)
    rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    mask = torch.from_numpy(mask.astype(np.float32) / 255.0)
    return rgba, rgb, mask

def image_to_bgr(image: ImageType) -> Tuple[ImageType, ImageType, int]:
    """RGB Helper function.
    Return channel count, BGR, and Alpha.
    """
    alpha = image_mask(image)
    cc = image.shape[2] if image.ndim == 3 else 1
    if cc == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif cc == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image, alpha, cc

def pil_to_cv(image: Image.Image) -> ImageType:
    """Convert a PIL Image to a CV2 Matrix."""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image

def pil_to_tensor(image: Image.Image) -> TensorType:
    """Convert a PIL Image to a Torch Tensor."""
    image = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image).unsqueeze(0)

def tensor_to_cv(tensor: TensorType, invert_mask:bool=True) -> ImageType:
    """Convert a torch Tensor to a numpy ndarray."""
    if tensor.ndim > 3:
        raise Exception("Tensor is batch of tensors")

    if tensor.ndim < 3:
        tensor = tensor.unsqueeze(-1)

    if tensor.shape[2] == 1 and invert_mask:
        tensor = 1. - tensor

    tensor = tensor.cpu().numpy()
    return np.clip(255.0 * tensor, 0, 255).astype(np.uint8)

def tensor_to_pil(tensor: TensorType) -> Image.Image:
    """Convert a torch Tensor to a PIL Image.
    Tensor should be HxWxC [no batch].
    """
    tensor = tensor.cpu().numpy().squeeze()
    tensor = np.clip(255. * tensor, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

'''
# ==============================================================================
# === CONVERSION ===
# ==============================================================================

def mixlabLayer_to_cv(layer: dict) -> torch.Tensor:
    image=layer['image']
    mask=layer['mask']
    if 'type' in layer and layer['type']=='base64' and type(image) == str:
        image = b64_2_cv(image)
        mask = b64_2_cv(mask)
    else:
        image = tensor2cv(image)
        mask = tensor2cv(mask)
    return image_mask_add(image, mask)

'''

# ==============================================================================
# === MASKING ===
# ==============================================================================

def image_mask(image: ImageType, color: int = 255) -> ImageType:
    """Create a mask from the image, preserving transparency.

    Args:
        image: Input image, assumed to be 2D or 3D (with or without alpha channel).
        color: Value to fill the mask (default is 255).

    Returns:
        COZY_TYPE_IMAGE: Mask of the image, either the alpha channel or a full mask of the given color.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., 3]

    h, w = image.shape[:2]
    return np.ones((h, w), dtype=np.uint8) * color

def image_mask_add(image:ImageType, mask:ImageType=None, alpha:float=255) -> ImageType:
    """Put custom mask into an image. If there is no mask, alpha is applied.
    Images are expanded to 4 channels.
    Existing 4 channel images with no mask input just return themselves.
    """
    image = image_convert(image, 4)
    mask = image_mask(image, alpha) if mask is None else image_convert(mask, 1)
    image[..., 3] = mask if mask.ndim == 2 else mask[:, :, 0]
    return image

def image_matte(image: ImageType, color: RGBA_Int=(0,0,0,255),
                width: int=None, height: int=None) -> ImageType:
    """
    Puts an RGBA image atop a colored matte expanding or clipping the image if requested.

    Args:
        image (ImageType): The input RGBA image.
        color (RGBA_Int): The color of the matte as a tuple (R, G, B, A).
        width (int, optional): The width of the matte. Defaults to the image width.
        height (int, optional): The height of the matte. Defaults to the image height.

    Returns:
        ImageType: Composited RGBA image on a matte with original alpha channel.
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

