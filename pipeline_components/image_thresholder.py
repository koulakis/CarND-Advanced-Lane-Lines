from sklearn.base import BaseEstimator
import cv2
import numpy as np

from .pipeline_state import TransformContext
from .utils import annotate_image_with_mask, gray_to_single_color


class ImageThresholder(BaseEstimator):
    def __init__(self, transform_function, overwrite_image=None):
        self.transform_function = transform_function
        self.overwrite_image = overwrite_image

    @staticmethod
    def blurring(image, ksize=3):
        return cv2.blur(image, (ksize, ksize))

    @staticmethod
    def scale_and_filter(measure, thresh):
        scaled_measure = np.uint8(255. * measure / np.max(measure))
        return np.where((thresh[0] <= scaled_measure) & (scaled_measure <= thresh[1]), 1, 0)

    @staticmethod
    def sobel_thresh_single_direction(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        orient_values = ['x', 'y']
        if orient not in orient_values:
            raise Exception('Orient can only take the values: {}'.format(', '.join(orient_values)))
        sobel = cv2.Sobel(image, cv2.CV_64F, orient == 'x', orient == 'y', ksize=sobel_kernel)
        abs_sobel = np.abs(sobel)

        return ImageThresholder.scale_and_filter(abs_sobel, thresh)

    @staticmethod
    def magnitude_thresh(image, sobel_kernel=3, thresh=(0, 255)):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel_lth = np.sqrt(sobelx ** 2 + sobely ** 2)

        return ImageThresholder.scale_and_filter(sobel_lth, thresh)

    @staticmethod
    def direction_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx, abs_sobely = list(map(np.abs, [sobelx, sobely]))
        sobel_arc = np.arctan2(abs_sobely, abs_sobelx)

        return ImageThresholder.scale_and_filter(sobel_arc, thresh)

    def fit(self):
        return self

    def transform(self, stateful_data):
        with TransformContext(self.__class__.__name__, stateful_data) as s:
            output_image = self.transform_function(s['data']).astype('float32')
            s['data'] = output_image
            if self.overwrite_image is not None:
                s['cached_image'] = (
                    annotate_image_with_mask(
                        s['cached_image'],
                        gray_to_single_color(output_image, (0, 0, 255)),
                        alpha=self.overwrite_image))

        return stateful_data
