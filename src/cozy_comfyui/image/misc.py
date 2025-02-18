

'''
@TODO: this is a color function
def bgr_to_hsv(bgr_color: COZY_TYPE_PIXEL) -> COZY_TYPE_PIXEL:
    return cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0, 0]

def bgr_to_cv(image: COZY_TYPE_IMAGE, alpha: COZY_TYPE_IMAGE=None, gray: bool=False) -> COZY_TYPE_IMAGE:
    """Restore image with alpha, if any, and converting to grayscale (optional)."""
    if gray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image_mask_add(image, alpha)
'''


'''
@TODO: this is a color function
def hsv_to_bgr(hsl_color: COZY_TYPE_PIXEL) -> COZY_TYPE_PIXEL:
    return cv2.cvtColor(np.uint8([[hsl_color]]), cv2.COLOR_HSV2BGR)[0, 0]

'''

'''
def rgb_to_bgr(image: COZY_TYPE_IMAGE) -> Tuple[COZY_TYPE_IMAGE, COZY_TYPE_IMAGE, int]:
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
'''

'''
@TODO: do I need this?

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