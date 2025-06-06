""" Image Composition Support """

from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from blendmodes.blend import \
    BlendType, \
    blendLayers

from . import \
    PixelType, Coord2D_Float, ImageType, RGBA_Int

from .convert import \
    image_convert, cv_to_pil, pil_to_cv, tensor_to_cv, image_mask, image_mask_add

from .misc import \
    image_minmax, image_by_size

from .pixel import \
    pixel_convert

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumBlendType(Enum):
    """Rename the blend type names."""
    NORMAL = BlendType.NORMAL
    ADDITIVE = BlendType.ADDITIVE
    NEGATION = BlendType.NEGATION
    DIFFERENCE = BlendType.DIFFERENCE
    MULTIPLY = BlendType.MULTIPLY
    DIVIDE = BlendType.DIVIDE
    LIGHTEN = BlendType.LIGHTEN
    DARKEN = BlendType.DARKEN
    SCREEN = BlendType.SCREEN
    BURN = BlendType.COLOURBURN
    DODGE = BlendType.COLOURDODGE
    OVERLAY = BlendType.OVERLAY
    HUE = BlendType.HUE
    SATURATION = BlendType.SATURATION
    LUMINOSITY = BlendType.LUMINOSITY
    COLOR = BlendType.COLOUR
    SOFT = BlendType.SOFTLIGHT
    HARD = BlendType.HARDLIGHT
    PIN = BlendType.PINLIGHT
    VIVID = BlendType.VIVIDLIGHT
    EXCLUSION = BlendType.EXCLUSION
    REFLECT = BlendType.REFLECT
    GLOW = BlendType.GLOW
    XOR = BlendType.XOR
    EXTRACT = BlendType.GRAINEXTRACT
    MERGE = BlendType.GRAINMERGE
    DESTIN = BlendType.DESTIN
    DESTOUT = BlendType.DESTOUT
    SRCATOP = BlendType.SRCATOP
    DESTATOP = BlendType.DESTATOP

class EnumEdge(Enum):
    CLIP = 1
    WRAP = 2
    WRAPX = 3
    WRAPY = 4

class EnumInterpolation(Enum):
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4
    LINEAR_EXACT = cv2.INTER_LINEAR_EXACT
    NEAREST_EXACT = cv2.INTER_NEAREST_EXACT
    # INTER_MAX = cv2.INTER_MAX
    # WARP_FILL_OUTLIERS = cv2.WARP_FILL_OUTLIERS
    # WARP_INVERSE_MAP = cv2.WARP_INVERSE_MAP

class EnumMirrorMode(Enum):
    NONE = -1
    X = 0
    FLIP_X = 10
    Y = 20
    FLIP_Y = 30
    XY = 40
    X_FLIP_Y = 50
    FLIP_XY = 60
    FLIP_X_FLIP_Y = 70

class EnumOrientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    GRID = 2

class EnumScaleMode(Enum):
    # NONE = 0
    MATTE = 0
    CROP = 20
    FIT = 10
    ASPECT = 30
    ASPECT_SHORT = 35
    RESIZE_MATTE = 40

class EnumScaleInputMode(Enum):
    NONE = 0
    CROP = 20
    FIT = 10
    ASPECT = 30
    ASPECT_SHORT = 35
    RESIZE_MATTE = 40

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def image_blend(background: ImageType, foreground: ImageType, mask:Optional[ImageType]=None,
                blendOp:EnumBlendType=EnumBlendType.NORMAL, alpha:float=1) -> ImageType:
    """Blending that will size to the largest input's background."""

    # prep A
    h, w = background.shape[:2]
    background = image_convert(background, 4, w, h)
    background = cv_to_pil(background)

    # prep B
    cc = foreground.shape[2] if foreground.ndim > 2 else 1
    foreground = image_convert(foreground, 4, w, h)
    old_mask = image_mask(foreground)

    if mask is None:
        mask = old_mask
    else:
        mask = image_convert(mask, 1, w, h)
        mask = mask[..., 0][:,:]
        if cc == 4:
            mask = cv2.bitwise_and(mask, old_mask)

    foreground[..., 3] = mask
    foreground = cv_to_pil(foreground)
    alpha = np.clip(alpha, 0, 1)
    image = blendLayers(background, foreground, blendOp.value, alpha)
    image = pil_to_cv(image)
    if cc != 1:
        image = image_mask_add(image, mask)
    return image

def image_crop(image: ImageType, width:int=None, height:int=None, offset:Tuple[float, float]=(0, 0)) -> ImageType:
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
        mask = image_mask(image)
        alpha_channel = cv2.resize(mask[y:y+h, x:x+w], (w, h))
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2BGR)
        cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=point_mask_cropped)
        return image_mask_add(cropped_image, alpha_channel)
    elif cc == 1:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
        cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=point_mask_cropped)
        return image_convert(cropped_image, cc)
    return cv2.bitwise_and(cropped_image, cropped_image, mask=point_mask_cropped)

def image_edge_wrap(image: ImageType, tileX: float=1., tileY: float=1.,
                    edge:EnumEdge=EnumEdge.WRAP) -> ImageType:
    """TILING."""
    height, width = image.shape[:2]
    tileX = int(width * tileX) if edge in [EnumEdge.WRAP, EnumEdge.WRAPX] else 0
    tileY = int(height * tileY) if edge in [EnumEdge.WRAP, EnumEdge.WRAPY] else 0
    return cv2.copyMakeBorder(image, tileY, tileY, tileX, tileX, cv2.BORDER_WRAP)

def image_flatten(image: List[ImageType], offsetX:int=None, offsetY:int=None,
                  width:int=None, height:int=None, mode=EnumScaleMode.MATTE,
                  sample:EnumInterpolation=EnumInterpolation.LANCZOS4) -> ImageType:

    if mode == EnumScaleMode.MATTE:
        width, height = image_minmax(image)[2:]
    else:
        h, w = image[0].shape[:2]
        width = width or w
        height = height or h

    current = np.zeros((height, width, 4), dtype=np.uint8)
    # current = image_mask_add(current, alpha=0)
    for x in image:
        if mode != EnumScaleMode.MATTE and mode != EnumScaleMode.RESIZE_MATTE:
            x = image_scalefit(x, width, height, mode, sample)
        # matte
        x = image_matte(x, (0,0,0,0), width, height)
        # Apply offset
        M = np.float32([[1, 0, offsetX], [0, 1, offsetY]])
        x = cv2.warpAffine(x, M, (width, height), borderMode=cv2.BORDER_TRANSPARENT)
        # fit
        x = image_scalefit(x, width, height, EnumScaleMode.CROP, sample)
        x = image_convert(x, 4)
        current = cv2.add(current, x)
    return current

def image_levels(image: ImageType, in_low=0.0, in_mid=0.5, in_high=1.0, out_low=0.0, out_high=1.0):
    """
    Apply levels adjustment to an image.

    Parameters:
        image (ImageType): Input RGB image in float32 format, range [0, 1].
        in_low (float): Input black point.
        in_high (float): Input white point.
        in_mid (float): Input gamma (midtone).
        out_low (float): Output black clamp.
        out_high (float): Output white clamp.

    Returns:
        ImageType: Adjusted image in float32 format, range [0, 1].
    """

    # Separate alpha channel if it exists
    has_alpha = image.ndim == 3 and image.shape[2] == 4
    if has_alpha:
        alpha = image[..., 3:]
        image = image[..., :3]

    # Ensure image is float32 and in 0–1 range
    image = image.astype(np.float32) / 255.0

    # Normalize input range
    scale = max(in_high - in_low, 1e-6)
    image = (image - in_low) / scale
    image = np.clip(image, 0.0, 1.0)

    # Apply gamma (midtone)
    in_mid = max(in_mid, 1e-6)
    gamma = 1.0 / in_mid
    image = np.power(image, gamma)

    # Scale to output range
    image = image * (out_high - out_low) + out_low
    image = np.clip(image, 0, 1)
    image = (image * 255).round().astype(np.uint8)

    # Recombine alpha if present
    if has_alpha:
        return np.concatenate([image, alpha], axis=2)
    return image

def image_matte(image: ImageType, color: RGBA_Int=(0, 0, 0, 255),
                width: int=None, height: int=None) -> ImageType:
    """Puts an RGB(A) image atop a colored matte expanding or clipping the image if requested.

    Args:
        image (TYPE_IMAGE): The input RGBA image.
        color (TYPE_iRGBA): The color of the matte as a tuple (R, G, B, A).
        width (int, optional): The width of the matte. Defaults to the image width.
        height (int, optional): The height of the matte. Defaults to the image height.

    Returns:
        TYPE_IMAGE: Composited RGBA image on a matte with original alpha channel.
    """

    # Determine the dimensions of the image and the matte
    image_height, image_width = image.shape[:2]
    width = width or image_width
    height = height or image_height

    # Create a solid matte with the specified color
    matte = np.full((height, width, 4), color, dtype=image.dtype)

    # Calculate the center position for the image on the matte
    x_offset = (width - image_width) // 2
    y_offset = (height - image_height) // 2

    # Extract the alpha channel from the image if it's RGBA
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3] / 255.0

        # Blend the RGB channels using the alpha mask
        for c in range(3):  # Iterate over RGB channels
            matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] = \
                (1 - alpha) * matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, c] + \
                alpha * image[:, :, c]

        # Set the alpha channel to the image's alpha channel
        matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, 3] = image[:, :, 3]
    else:
        # Handle non-RGBA images (just copy the image onto the matte)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)
        matte[y_offset:y_offset + image_height, x_offset:x_offset + image_width, :3] = image[:, :, :3]

    return matte

def image_mirror(image: ImageType, mode:EnumMirrorMode, x:float=0.5,
                 y:float=0.5) -> ImageType:
    cc = image.shape[2] if image.ndim == 3 else 1
    height, width = image.shape[:2]

    def mirror(img:ImageType, axis:int, reverse:bool=False) -> ImageType:
        pivot = x if axis == 1 else y
        flip = cv2.flip(img, axis)
        pivot = np.clip(pivot, 0, 1)
        if reverse:
            pivot = 1. - pivot
            flip, img = img, flip

        scalar = height if axis == 0 else width
        slice1 = int(pivot * scalar)
        slice1w = scalar - slice1
        slice2w = min(scalar - slice1w, slice1w)

        if cc >= 3:
            output = np.zeros((height, width, cc), dtype=np.uint8)
        else:
            output = np.zeros((height, width), dtype=np.uint8)

        if axis == 0:
            output[:slice1, :] = img[:slice1, :]
            output[slice1:slice1 + slice2w, :] = flip[slice1w:slice1w + slice2w, :]
        else:
            output[:, :slice1] = img[:, :slice1]
            output[:, slice1:slice1 + slice2w] = flip[:, slice1w:slice1w + slice2w]

        return output

    if mode in [EnumMirrorMode.X, EnumMirrorMode.FLIP_X, EnumMirrorMode.XY, EnumMirrorMode.FLIP_XY, EnumMirrorMode.X_FLIP_Y, EnumMirrorMode.FLIP_X_FLIP_Y]:
        reverse = mode in [EnumMirrorMode.FLIP_X, EnumMirrorMode.FLIP_XY, EnumMirrorMode.FLIP_X_FLIP_Y]
        image = mirror(image, 1, reverse)

    if mode not in [EnumMirrorMode.NONE, EnumMirrorMode.X, EnumMirrorMode.FLIP_X]:
        reverse = mode in [EnumMirrorMode.FLIP_Y, EnumMirrorMode.FLIP_X_FLIP_Y, EnumMirrorMode.X_FLIP_Y]
        image = mirror(image, 0, reverse)

    return image

def image_resize(image: ImageType, width: int, height: int, sample: EnumInterpolation) -> ImageType:
    return cv2.resize(image, (width, height), interpolation=sample)

def image_rotate(image: ImageType, angle: float, center:Coord2D_Float=(0.5, 0.5),
                 edge:EnumEdge=EnumEdge.CLIP) -> ImageType:

    h, w = image.shape[:2]
    if edge != EnumEdge.CLIP:
        image = image_edge_wrap(image, edge=edge)

    height, width = image.shape[:2]
    c = (int(width * center[0]), int(height * center[1]))
    M = cv2.getRotationMatrix2D(c, -angle, 1.0)
    image = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)
    if edge != EnumEdge.CLIP:
        image = image_crop_center(image, w, h)
    return image

def image_scale(image: ImageType, scale:Coord2D_Float=(1.0, 1.0),
                sample:EnumInterpolation=EnumInterpolation.LANCZOS4,
                edge:EnumEdge=EnumEdge.CLIP) -> ImageType:

    h, w = image.shape[:2]
    if edge != EnumEdge.CLIP:
        image = image_edge_wrap(image, edge=edge)

    height, width = image.shape[:2]
    width = int(width * scale[0])
    height = int(height * scale[1])
    image = cv2.resize(image, (width, height), interpolation=sample.value)

    if edge != EnumEdge.CLIP:
        image = image_crop_center(image, w, h)
    return image

def image_scalefit(image: ImageType, width: int, height:int,
                mode:EnumScaleMode=EnumScaleMode.MATTE,
                sample:EnumInterpolation=EnumInterpolation.LANCZOS4,
                matte:PixelType=(0,0,0,0)) -> ImageType:

    match mode:
        case EnumScaleMode.MATTE:
            image = image_matte(image, matte, width, height)

        case EnumScaleMode.RESIZE_MATTE:
            h, w = image.shape[:2]
            w2 = max(width, w)
            h2 = max(height, h)
            canvas = np.full((h2, w2, 4), matte, dtype=image.dtype)
            mask = image_mask(image)
            # mask = image_matte(mask, (255, 255, 255, 255), w2, h2)

            image = image_matte(image, (0,0,0,0), w2, h2)
            image = image_blend(canvas, image)
            #image = image_mask_add(image, mask)
            image = image_crop_center(image, width, height)

        case EnumScaleMode.ASPECT:
            h, w = image.shape[:2]
            ratio = max(width, height) / max(w, h)
            image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=sample.value)

        case EnumScaleMode.ASPECT_SHORT:
            h, w = image.shape[:2]
            ratio = min(width, height) / min(w, h)
            image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=sample.value)

        case EnumScaleMode.CROP:
            h, w = image.shape[:2]
            if h<height or w<width:
                image = image_scalefit(image, width, height, EnumScaleMode.RESIZE_MATTE, sample, matte)
            image = image_crop_center(image, width, height)

        case EnumScaleMode.FIT:
            image = cv2.resize(image, (width, height), interpolation=sample.value)

    #if image.ndim == 2:
    #    image = np.expand_dims(image, -1)
    return image

def image_split(image: ImageType) -> tuple[ImageType, ...]:
    h, w = image.shape[:2]

    # Grayscale image
    a = np.full((h, w, 1), 255, dtype=image.dtype)
    if image.ndim == 2 or image.shape[2] == 1:
        r = g = b = image
    # RGB(A) image
    else:
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        if image.shape[2] == 4:
            a = image[:, :, 3]
    return r, g, b, a

def image_stacker(image_list: List[ImageType],
                axis:EnumOrientation=EnumOrientation.HORIZONTAL,
                stride:int=0, matte:PixelType=(0,0,0,255)) -> ImageType:

    _, width, height = image_by_size(image_list)
    images = [image_matte(image_convert(i, 4), matte, width, height) for i in image_list]
    count = len(images)

    matte = pixel_convert(matte, 4)
    match axis:
        case EnumOrientation.GRID:
            if stride < 1:
                stride = np.ceil(np.sqrt(count))
                stride = int(stride)
            stride = min(stride, count)
            stride = max(stride, 1)

            rows = []
            for i in range(0, count, stride):
                row = images[i:i + stride]
                row_stacked = np.hstack(row)
                rows.append(row_stacked)

            height, width = images[0].shape[:2]
            overhang = count % stride
            if overhang != 0:
                overhang = stride - overhang
                size = (height, overhang * width, 4)
                filler = np.full(size, matte, dtype=np.uint8)
                rows[-1] = np.hstack([rows[-1], filler])
            image = np.vstack(rows)

        case EnumOrientation.HORIZONTAL:
            image = np.hstack(images)

        case EnumOrientation.VERTICAL:
            image = np.vstack(images)
    return image

def image_translate(image: ImageType, offset: Coord2D_Float=(0.0, 0.0),
                    edge: EnumEdge=EnumEdge.CLIP, border_value:int=0) -> ImageType:
    """
    Translates an image by a given offset. Supports various edge handling methods.

    Args:
        image (ImageType): Input image as a numpy array.
        offset (Coord2D_Float): Tuple (offset_x, offset_y) representing the translation offset.
        edge (EnumEdge): Enum representing edge handling method. Options are 'CLIP', 'WRAP', 'WRAPX', 'WRAPY'.

    Returns:
        ImageType: Translated image.
    """

    def translate(img: ImageType) -> ImageType:
        height, width = img.shape[:2]
        scalarX = 0.333 if edge in [EnumEdge.WRAP, EnumEdge.WRAPX] else 1.0
        scalarY = 0.333 if edge in [EnumEdge.WRAP, EnumEdge.WRAPY] else 1.0
        scalarX = 1.0
        scalarY = 1.0

        M = np.float32([[1, 0, offset[0] * width * scalarX], [0, 1, offset[1] * height * scalarY]])
        if edge == EnumEdge.CLIP:
            border_mode = cv2.BORDER_CONSTANT
        else:
            border_mode = cv2.BORDER_WRAP

        return cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=border_value)

    return translate(image)

def image_transform(image: ImageType, offset:Coord2D_Float=(0.0, 0.0),
                    angle:float=0, scale:Coord2D_Float=(1.0, 1.0),
                    sample:EnumInterpolation=EnumInterpolation.LANCZOS4,
                    edge:EnumEdge=EnumEdge.CLIP) -> ImageType:
    sX, sY = scale
    if sX < 0:
        image = cv2.flip(image, 1)
        sX = -sX
    if sY < 0:
        image = cv2.flip(image, 0)
        sY = -sY
    if sX != 1. or sY != 1.:
        image = image_scale(image, (sX, sY), sample, edge)
    if angle != 0:
        image = image_rotate(image, angle, edge=edge)
    if offset[0] != 0. or offset[1] != 0.:
        image = image_translate(image, offset, edge)
    return image

def image_tensor_alpha_mask(image: ImageType, mask: ImageType=None, color: int=255) -> tuple[ImageType, ...]:
    image = channel_solid() if image is None else tensor_to_cv(image)
    alpha = image_mask(image)
    height, width = image.shape[:2]
    mask = channel_solid(width, height, color, EnumImageType.GRAYSCALE) if mask is None else tensor_to_cv(mask, chan=1)
    return image, alpha, mask
