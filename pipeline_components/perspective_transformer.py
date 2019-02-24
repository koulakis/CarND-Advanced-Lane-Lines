from sklearn.base import BaseEstimator
import cv2
import numpy as np


class PerspectiveTransformer(BaseEstimator):
    def __init__(self, image_shape, inverse=False, x_low_offset=.14, x_high_offset=.48):
        self.x_low_offset = x_low_offset
        self.x_high_offset = x_high_offset
        self.img_height, self.img_width = image_shape[:2]
        self.transform_matrix = None
        self.inverse = inverse

    def fit(self):
        x_low_offset, x_high_offset = self.x_low_offset, self.x_high_offset
        img_height, img_width = self.img_height, self.img_width

        source_points = np.array([
            (x_low_offset * img_width, img_height),
            (x_high_offset * img_width, .6 * img_height),
            ((1 - x_high_offset) * img_width, .6 * img_height),
            ((1 - x_low_offset) * img_width, img_height)], dtype='float32')

        x_left_target = (x_low_offset + x_high_offset) / 2
        x_right_target = 1 - x_left_target

        target_points = np.array([
            (x_left_target * img_width, img_height),
            (x_left_target * img_width, 0),
            (x_right_target * img_width, 0),
            (x_right_target * img_width, img_height)], dtype='float32')

        self.transform_matrix = (
            cv2.getPerspectiveTransform(target_points, source_points)
            if self.inverse
            else cv2.getPerspectiveTransform(source_points, target_points))

    def transform(self, image):
        return cv2.warpPerspective(
            image,
            self.transform_matrix,
            (self.img_width, self.img_height),
            flags=cv2.INTER_LINEAR)
