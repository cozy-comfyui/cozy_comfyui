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

class EnumAdjustBlur(Enum):
    BLUR = 10
    STACK_BLUR = 20
    GAUSSIAN_BLUR = 30
    MEDIAN_BLUR = 40

class EnumAdjustEdge(Enum):
    OUTLINE = 20
    CANNY  = 30
    LAPLACIAN = 40
    SOBEL = 50
    PREWITT = 60
    SCHARR = 70

class EnumAdjustMorpho(Enum):
    DILATE = 10
    ERODE = 20
    OPEN = 30
    CLOSE = 40
    TOPHAT = 50
    BLACKHAT = 60

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

def image_blur(image: ImageType, op: EnumAdjustBlur=EnumAdjustBlur.BLUR, kernel: int=1, sigmaX: float=0, sigmaY: float=0) -> ImageType:
    if kernel % 2 == 0:
        kernel += 1

    match op:
        case EnumAdjustBlur.BLUR:
            return cv2.blur(image, (kernel, kernel))

        case EnumAdjustBlur.STACK_BLUR:
            return cv2.stackBlur(image, (kernel, kernel))

        case EnumAdjustBlur.GAUSSIAN_BLUR:
            return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigmaX, sigmaY=sigmaY)

        case EnumAdjustBlur.MEDIAN_BLUR:
            return cv2.medianBlur(image, kernel)

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

def image_edge(image: ImageType, op: EnumAdjustEdge=EnumAdjustEdge.CANNY,
               kernel: int=1, iterations: int=1,
               lo: float=0, hi: float=1.) -> ImageType:

    if kernel % 2 == 0:
        kernel += 1

    match op:
        case EnumAdjustEdge.OUTLINE:
            kernel = max(3, kernel)
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT,
                                    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel)),
                                    iterations=iterations)

        case EnumAdjustEdge.CANNY:
            image = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0.5)
            lo = int(np.clip(lo * 255, 0, 255))
            hi = int(np.clip(hi * 255, 0, 255))
            return cv2.Canny(image, threshold1=lo, threshold2=hi)

        case EnumAdjustEdge.LAPLACIAN:
            kernel = min(31, kernel)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            lap = cv2.Laplacian(gray, ddepth=cv2.CV_64F, ksize=kernel)
            return cv2.convertScaleAbs(lap)

        case EnumAdjustEdge.SOBEL:
            kernel = min(31, kernel)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
            sobel = cv2.magnitude(sobelx, sobely)
            return cv2.convertScaleAbs(sobel)

        case EnumAdjustEdge.PREWITT:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            # Custom Prewitt kernels
            kernelx = np.array([[kernel, 0, -kernel], [kernel, 0, -kernel], [kernel, 0, -kernel]])
            kernely = np.array([[kernel, kernel, kernel], [0, 0, 0], [-kernel, -kernel, -kernel]])
            prewittx = cv2.filter2D(gray, -1, kernelx)
            prewitty = cv2.filter2D(gray, -1, kernely)
            prewitt = cv2.magnitude(prewittx.astype(np.float32), prewitty.astype(np.float32))
            return cv2.convertScaleAbs(prewitt)

        case EnumAdjustEdge.SCHARR:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=kernel)
            scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=kernel)
            scharr = cv2.magnitude(scharrx, scharry)
            return cv2.convertScaleAbs(scharr)

'''
def image_emboss(image: ImageType, amount: float=1., kernel: int=0) -> ImageType:
    kernel = max(0, kernel)
    if kernel % 2 == 0:
        kernel += 1
    kernel = np.array([
        [-kernel,   -kernel+1,    0],
        [-kernel+1,   kernel-1,     1],
        [kernel-2,    kernel-1,     2]
    ]) * amount
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
'''

def image_emboss(image: ImageType, azimuth: float=-45.0, elevation: float=45.0,
                 depth: float=10.0) -> ImageType:
    """
    Apply emboss effect to an image with a specified light angle.

    Args:
        azimuth: Direction of light in degrees (0° = right, 90° = down, 180° = left, 270° = up)
        elevation: Direction of light in degrees (0° = head on, 90° = top down)
        depth: Thickness

    Returns:
        Embossed image.
    """
    image = image.astype('float')
    image = image_convert(image, 3)
    grad_x, grad_y, grad_z = np.gradient(image)

    # length of projection of ray on ground plane
    azimuth = np.radians(azimuth-90)
    elevation = np.radians(elevation)
    gd = np.cos(elevation)
    dx = gd * np.cos(azimuth)
    dy = gd * np.sin(azimuth)
    dz = np.sin(elevation)

    # depth
    grad_x = grad_x * depth / 100.
    grad_y = grad_y * depth / 100.
    #grad_z = grad_z * depth / 100.

    # finding the unit normal vectors for the image
    leng = np.sqrt(grad_x**2 + grad_y**2 + 1.)
    uni_x = grad_x / leng
    uni_y = grad_y / leng
    uni_z = 1. / leng
    image = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
    return image.clip(0, 255).astype('uint8')

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

def image_filter(image:ImageType, start:tuple[int, ...]=(128,128,128),
                 end:tuple[int, ...]=(128,128,128), fuzz:tuple[float, ...]=(0.5,0.5,0.5),
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

    fuzz = torch.Tensor(fuzz, dtype=torch.float64, device="cpu")
    start_tensor = torch.Tensor(start, dtype=torch.float64, device="cpu") / 255.
    end_tensor = torch.Tensor(end, dtype=torch.float64, device="cpu") / 255.
    if not use_range:
        end_tensor = start_tensor
    start_tensor -= fuzz
    end_tensor += fuzz
    start = torch.clamp(start_tensor, 0.0, 1.0)
    end = torch.clamp(end_tensor, 0.0, 1.0)

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

def image_morphology(image: ImageType, op: EnumAdjustMorpho=EnumAdjustMorpho.DILATE,
               kernel_size: int=3, iterations: int=1) -> ImageType:

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    match op:
        case EnumAdjustMorpho.DILATE:
            return cv2.dilate(image, kernel, iterations=iterations)

        case EnumAdjustMorpho.ERODE:
            return cv2.erode(image, kernel, iterations=iterations)

        case EnumAdjustMorpho.OPEN:
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

        case EnumAdjustMorpho.CLOSE:
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        case EnumAdjustMorpho.TOPHAT:
            return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel, iterations=iterations)

        case EnumAdjustMorpho.BLACKHAT:
            return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)

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

def image_sharpen(image:ImageType, amount:float=1., kernel: int=3,
                  sigma:float=0.5, threshold:float=0) -> ImageType:
    """Return a sharpened version of the image, using an unsharp mask."""
    if kernel and kernel % 2 == 0:
        kernel += 1
    blurred = cv2.GaussianBlur(image, (kernel, kernel), sigma)
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
