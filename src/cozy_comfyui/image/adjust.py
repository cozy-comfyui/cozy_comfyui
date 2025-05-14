"""Image adjustment operations."""

from enum import Enum

import cv2
import torch
import numpy as np

from .. import \
    TensorType

from . import \
    ImageType

from .convert import \
    cv_to_tensor, tensor_to_cv, image_convert, srgb_to_linear, linear_to_srgb

# ==============================================================================
# === ENUMERATION ===
# ==============================================================================

class EnumThreshold(Enum):
    BINARY = cv2.THRESH_BINARY
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO

class EnumThresholdAdapt(Enum):
    ADAPT_NONE = -1
    ADAPT_MEAN = cv2.ADAPTIVE_THRESH_MEAN_C
    ADAPT_GAUSS = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

# ==============================================================================
# === SUPPORT ===
# ==============================================================================

def image_brightness(image: ImageType, brightness: float=0):
    brightness = np.clip(brightness, -1, 1) * 255
    if brightness > 0:
        shadow = brightness
        highlight = 255
    else:
        shadow = 0
        highlight = 255 + brightness
    alpha_b = (highlight - shadow)/255
    gamma_b = shadow

    return cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

def image_contrast(image: ImageType, contrast: float) -> ImageType:
    # Map contrast from [-255, 255] to factor
    contrast = np.clip(contrast, -1, 1) * 255
    factor = (255 * (contrast + 255)) / (255 * (255 - contrast))

    def image_contrast_rgb(lab: ImageType) -> ImageType:
        """Adjust contrast in RGB image using LAB color space and standard contrast scaling."""
        lab = cv2.cvtColor(lab, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        L = L.astype(np.float32)
        L = factor * (L - 128) + 128
        L = np.clip(L, 0, 255).astype(np.uint8)
        lab = cv2.merge([L, A, B])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Grayscale
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        img = image.astype(np.float32)
        img = factor * (img - 128) + 128
        return np.clip(img, 0, 255).astype(np.uint8)
    # RGB
    elif image.shape[2] == 3:
        return image_contrast_rgb(image)
    # RGBA
    rgb = image[..., :3]
    alpha = image[..., 3:]
    rgb = image_contrast_rgb(rgb)
    return np.concatenate([rgb, alpha], axis=2)

def image_edge_detect(image: ImageType,
                    ksize: int=3,
                    low: float=0.27,
                    high:float=0.6) -> ImageType:

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ksize = max(3, ksize)
    image = cv2.GaussianBlur(src=image, ksize=(ksize, ksize+2), sigmaX=0.5)
    # Perform Canny edge detection
    return cv2.Canny(image, int(low * 255), int(high * 255))

def image_emboss(image: ImageType, amount: float=1., kernel: int=2) -> ImageType:
    kernel = max(2, kernel)
    kernel = np.array([
        [-kernel,   -kernel+1,    0],
        [-kernel+1,   kernel-1,     1],
        [kernel-2,    kernel-1,     2]
    ]) * amount
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

def image_equalize(image:ImageType) -> ImageType:
    image = image_convert(image, 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    return cv2.cvtColor(image, cv2.COLOR_YUV2RGB)

def image_exposure(image: ImageType, value: float) -> ImageType:
    linear = srgb_to_linear(image.astype(np.float32))
    exposed = linear * (2.0 ** value)
    srgb = linear_to_srgb(exposed)
    return np.clip(srgb * 255, 0, 255).astype(np.uint8)

def image_filter(image:ImageType, start:tuple[int]=(128,128,128),
                 end:tuple[int]=(128,128,128), fuzz:tuple[float]=(0.5,0.5,0.5),
                 use_range:bool=False) -> tuple[ImageType, ImageType]:
    """Filter an image based on a range threshold.
    It can use a start point with fuzziness factor and/or a start and end point with fuzziness on both points.

    Args:
        image (np.ndarray): Input image in the form of a NumPy array.
        start (tuple): The lower bound of the color range to be filtered.
        end (tuple): The upper bound of the color range to be filtered.
        fuzz (float): A factor for adding fuzziness (tolerance) to the color range.
        use_range (bool): Boolean indicating whether to use a start and end range or just the start point with fuzziness.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the filtered image and the mask.
    """
    old_alpha = None
    new_image = cv_to_tensor(image)
    cc = image.shape[2] if image.ndim > 2 else 1
    if cc == 4:
        old_alpha = new_image[..., 3]
        new_image = new_image[:, :, :3]
    elif cc == 1:
        if new_image.ndim == 2:
            new_image = new_image.unsqueeze(-1)
        new_image = torch.repeat_interleave(new_image, 3, dim=2)

    fuzz = TensorType(fuzz, dtype=torch.float64, device="cpu")
    start = TensorType(start, dtype=torch.float64, device="cpu") / 255.
    end = TensorType(end, dtype=torch.float64, device="cpu") / 255.
    if not use_range:
        end = start
    start -= fuzz
    end += fuzz
    start = torch.clamp(start, 0.0, 1.0)
    end = torch.clamp(end, 0.0, 1.0)

    mask = ((new_image[..., 0] > start[0]) & (new_image[..., 0] < end[0]))
    #mask |= ((new_image[..., 1] > start[1]) & (new_image[..., 1] < end[1]))
    #mask |= ((new_image[..., 2] > start[2]) & (new_image[..., 2] < end[2]))
    mask = ((new_image[..., 0] >= start[0]) & (new_image[..., 0] <= end[0]) &
            (new_image[..., 1] >= start[1]) & (new_image[..., 1] <= end[1]) &
            (new_image[..., 2] >= start[2]) & (new_image[..., 2] <= end[2]))

    output_image = torch.zeros_like(new_image)
    output_image[mask] = new_image[mask]

    if old_alpha is not None:
        output_image = torch.cat([output_image, old_alpha.unsqueeze(2)], dim=2)

    return tensor_to_cv(output_image), mask.cpu().numpy().astype(np.uint8) * 255

def image_gamma(image: ImageType, gamma: float) -> ImageType:
    if gamma <= 0:
        return np.zeros_like(image, dtype=np.uint8)

    gamma = 1.0 / max(1e-6, gamma)
    table = np.power(np.linspace(0, 1, 256), gamma) * 255
    lookup_table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(image, lookup_table)

def image_histogram(image:ImageType, bins=256) -> ImageType:
    bins = max(image.max(), bins) + 1
    flatImage = image.flatten()
    histogram = np.zeros(bins)
    for pixel in flatImage:
        histogram[pixel] += 1
    return histogram

def image_histogram_normalize(image:ImageType)-> ImageType:
    L = image.max()
    nonEqualizedHistogram = image_histogram(image, bins=L)
    sumPixels = np.sum(nonEqualizedHistogram)
    nonEqualizedHistogram = nonEqualizedHistogram/sumPixels
    cfdHistogram = np.cumsum(nonEqualizedHistogram)
    transformMap = np.floor((L-1) * cfdHistogram)
    flatNonEqualizedImage = list(image.flatten())
    flatEqualizedImage = [transformMap[p] for p in flatNonEqualizedImage]
    return np.reshape(flatEqualizedImage, image.shape)

def image_hsv(image: ImageType, hue: float, saturation: float, value: float) -> ImageType:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue *= 255
    image[:, :, 0] = (image[:, :, 0] + hue) % 180
    image[:, :, 1] = np.clip(image[:, :, 1] * saturation, 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * value, 0, 255)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def image_invert(image: ImageType, value: float) -> ImageType:
    """
    Invert a Grayscale, RGB, or RGBA image using a specified inversion intensity.

    Parameters:
    - image: Input image as a NumPy array (grayscale, RGB, or RGBA).
    - value: Float between 0 and 1 representing the intensity of inversion (0: no inversion, 1: full inversion).

    Returns:
    - Inverted image.
    """
    # Clip the value to be within [0, 1] and scale to [0, 255]
    value = np.clip(value, 0, 1)

    # RGBA
    if image.ndim == 3 and image.shape[2] == 4:
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]
        inverted_rgb = 255 - rgb
        blended_rgb = ((1 - value) * rgb + value * inverted_rgb).astype(np.uint8)
        return np.dstack((blended_rgb, alpha))

    # Grayscale & RGB
    inverted = 255 - image
    return ((1 - value) * image + value * inverted).astype(np.uint8)

def image_pixelate(image: ImageType, amount:float)-> ImageType:
    h, w = image.shape[:2]
    amount = max(0, min(1, amount / float(max(w, h))))
    block_size_h = max(1, (h * amount))
    block_size_w = max(1, (w * amount))
    num_blocks_h = int(np.ceil(h / block_size_h))
    num_blocks_w = int(np.ceil(w / block_size_w))
    block_size_h = h // num_blocks_h
    block_size_w = w // num_blocks_w
    pixelated_image = image.copy()

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Calculate block boundaries
            y_start = i * block_size_h
            y_end = min((i + 1) * block_size_h, h)
            x_start = j * block_size_w
            x_end = min((j + 1) * block_size_w, w)

            # Average color values within the block
            block_average = np.mean(image[y_start:y_end, x_start:x_end], axis=(0, 1))

            # Fill the block with the average color
            pixelated_image[y_start:y_end, x_start:x_end] = block_average

    return pixelated_image.astype(np.uint8)

def image_pixelscale(image: ImageType, amount:int)-> ImageType:
    height, width = image.shape[:2]
    amount = max(1, max(width, height) - amount)
    # amount = max(1, min(amount, max(width, height)))
    w, h = (amount, amount)
    temp = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

def image_posterize(image: ImageType, levels:int=256) -> ImageType:
    divisor = 256 / max(2, min(256, levels))
    return (np.floor(image / divisor) * int(divisor)).astype(np.uint8)

def image_quantize(image:ImageType, levels:int=256, iterations:int=5,
                   epsilon:float=0.2) -> ImageType:
    levels = int(max(2, min(256, levels)))
    pixels = np.float32(image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)
    _, labels, centers = cv2.kmeans(pixels, levels, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(image.shape)

def image_sharpen(image:ImageType, kernel_size=None, sigma:float=1.0,
                amount:float=1.0, threshold:float=0) -> ImageType:
    """Return a sharpened version of the image, using an unsharp mask."""

    kernel_size = (kernel_size, kernel_size) if kernel_size else (5, 5)
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def image_threshold(image:ImageType, threshold:float=0.5,
                    mode:EnumThreshold=EnumThreshold.BINARY,
                    adapt:EnumThresholdAdapt=EnumThresholdAdapt.ADAPT_NONE,
                    block:int=3, const:float=0.) -> ImageType:

    const = max(-100, min(100, const))
    block = max(3, block if block % 2 == 1 else block + 1)
    if adapt != EnumThresholdAdapt.ADAPT_NONE:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, adapt.value, cv2.THRESH_BINARY, block, const)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # gray = np.stack([gray, gray, gray], axis=-1)
        image = cv2.bitwise_and(image, gray)
    else:
        threshold = int(threshold * 255)
        _, image = cv2.threshold(image, threshold, 255, mode.value)
    return image
