import numpy as np

from .pipeline_state import TransformError


def annotate_image_with_mask(image: np.ndarray, mask: np.ndarray, alpha: float=0.3):
    if image.shape != mask.shape:
        raise TransformError(
            'Trying to annotate image with incompatible shaped mask. Image shape: {}, mask shape :{}'.format(
                image.shape,
                mask.shape
            ))
    return np.minimum(255, image + alpha * mask).astype(np.uint8)


def gray_to_single_color(image: np.uint8, rgb_color: (int, int, int)):
    return np.stack([np.where(image == 1, color, 0) for color in rgb_color], axis=2)

