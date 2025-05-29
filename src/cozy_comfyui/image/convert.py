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

from . import \
    RGBA_Int, ImageType

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def image_convert(image: ImageType, channels: int,
                  width: int=0, height: int=0,
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
            image = image_grayscale(image)
            #image = np.mean(image, axis=2, keepdims=True).astype(image.dtype)
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
    new_width = width if width > 0 else w
    new_height = height if height > 0 else h
    if (new_width, new_height) != (w, h):
        # Create a new image with the matte color
        new_image = np.full((new_height, new_width, channels), matte[:channels], dtype=image.dtype)
        paste_x = (new_width - w) // 2
        paste_y = (new_height - h) // 2
        new_image[paste_y:paste_y+h, paste_x:paste_x+w] = image[:h, :w]
        image = new_image

    return image

def image_grayscale(image: ImageType, use_alpha: bool=False) -> ImageType:
    """Convert image to grayscale, optionally using the alpha channel if present.

    Args:
        image (ImageType): Input image, potentially with multiple channels.
        use_alpha (bool): If True and the image has 4 channels, multiply the grayscale
                          values by the alpha channel. Defaults to False.

    Returns:
        ImageType: Grayscale image, optionally alpha-multiplied.
    """
    if image.ndim == 2:
        image = np.expand_dims(image, -1)

    if image.shape[2] == 1:
        return image

    if image.shape[2] == 4:
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        grayscale = np.expand_dims(grayscale, axis=-1)
        if use_alpha:
            alpha_channel = image[:,:,3] / 255.0
            grayscale = (grayscale * alpha_channel).astype(np.uint8)
        return grayscale

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.expand_dims(image, axis=-1)

def image_mask(image: ImageType, color: int=255) -> ImageType:
    """Get the alpha mask from the image, if any, otherwise return one based on color value.

    Args:
        image: Input image, assumed to be 2D or 3D (with or without alpha channel).
        color: Value to fill the mask (default is 255).

    Returns:
        ImageType: Mask of the image, either the alpha channel or a full mask of the given color.
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
    h, w, c = image.shape
    mask = cv2.resize(mask, (w, h))
    image[..., 3] = mask if mask.ndim == 2 else mask[:, :, 0]
    return image

def image_mask_binary(image: ImageType) -> ImageType:
    """Convert an image to a binary mask where non-black pixels are 1 and black pixels are 0.
    Supports BGR, single-channel grayscale, and RGBA images.

    Args:
        image (ImageType): Input image in BGR, grayscale, or RGBA format.

    Returns:
        ImageType: Binary mask with the same width and height as the input image, where
                    pixels are 1 for non-black and 0 for black.
    """
    if image.ndim == 2:
        # Grayscale image
        gray = image
    elif image.shape[2] == 3:
        # BGR image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        # RGBA image
        alpha_channel = image[..., 3]
        # Create a mask from the alpha channel where alpha > 0
        alpha_mask = alpha_channel > 0
        # Convert RGB to grayscale
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        # Apply the alpha mask to the grayscale image
        gray = cv2.bitwise_and(gray, gray, mask=alpha_mask.astype(np.uint8))
    else:
        raise ValueError("Unsupported image format")

    # Create a binary mask where any non-black pixel is set to 1
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, -1)
    return mask.astype(np.uint8)

def b64_to_tensor(base64str: str) -> TensorType:
    img = base64.b64decode(base64str)
    img = Image.open(BytesIO(img))
    img = ImageOps.exif_transpose(img)
    return pil_to_tensor(img)

def b64_to_pil(base64_string: str):
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
    if image.ndim < 3:
        image = np.expand_dims(image, -1)

    if grayscale:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)

def srgb_to_linear(img: ImageType) -> ImageType:
    img = img / 255.0
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(img: ImageType) -> ImageType:
    img = np.clip(img, 0, 1)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * (img ** (1 / 2.4)) - 0.055)

def tensor_to_cv(tensor: TensorType, invert: bool=False, chan: int=0) -> ImageType:
    """
    Convert a torch Tensor (HWC or HW, float32 in [0, 1]) to a NumPy uint8 image array.

    - Adds a channel dimension for grayscale images if missing.
    - Optionally inverts the image (1.0 becomes 0.0 and vice versa).
    - Converts values from float [0, 1] to uint8 [0, 255].
    - Optionally forces the image to have 1, 3, or 4 channels.

    Args:
        tensor (TensorType): Image tensor with shape (H, W), (H, W, 1), or (H, W, 3).
        invert (bool): If True, invert the image.
        chan (int, optional): Force the image to have 1, 3, or 4 channels.

    Returns:
        ImageType: NumPy array with shape (H, W, C) and dtype uint8.
    """
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 2:  # [H, W] ==> HWC
        tensor = tensor.unsqueeze(-1)

    if tensor.ndim > 3:
        raise Exception("Tensor is batch of tensors")

    if invert:
        if tensor.shape[2] == 4:
            tensor[:, :, :3] = 1.0 - tensor[:, :, :3]
        else:
            tensor = 1.0 - tensor

    image = np.clip(tensor.cpu().numpy() * 255, 0, 255).astype(np.uint8)
    if chan > 0:
        image = image_convert(image, chan)
    return image

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
