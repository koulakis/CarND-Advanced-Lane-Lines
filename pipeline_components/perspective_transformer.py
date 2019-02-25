from sklearn.base import BaseEstimator
import cv2
import numpy as np


class PerspectiveTransformer(BaseEstimator):
    def __init__(self, image_shape, inverse=False,
                 x_low_offset=.15, x_high_offset=.44, y_offset=0.65, overwrite_image=None):
        self.x_low_offset = x_low_offset
        self.x_high_offset = x_high_offset
        self.y_offset = y_offset
        self.img_height, self.img_width = image_shape[:2]
        self.transform_matrix = None
        self.source_points = None
        self.inverse = inverse
        self.overwrite_image = overwrite_image

    def fit(self):
        x_low_offset, x_high_offset = self.x_low_offset, self.x_high_offset
        img_height, img_width = self.img_height, self.img_width

        source_points = np.array([
            (x_low_offset * img_width, img_height),
            (x_high_offset * img_width, self.y_offset * img_height),
            ((1 - x_high_offset) * img_width, self.y_offset * img_height),
            ((1 - x_low_offset) * img_width, img_height)], dtype='float32')

        self.source_points = source_points

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

        return self

    def transform(self, stateful_data):
        warped_image = (cv2.warpPerspective(
            stateful_data['X'],
            self.transform_matrix,
            (self.img_width, self.img_height),
            flags=cv2.INTER_LINEAR))

        output = stateful_data.copy()
        output['X'] = warped_image
        if self.overwrite_image is not None:
            output['state']['image'] = (
                    self.overwrite_image * np.stack([
                        np.zeros(warped_image.shape),
                        np.zeros(warped_image.shape),
                        np.where(warped_image == 1, 255, 0)], axis=2)
                    + (1 - self.overwrite_image) * output['state']['image'])

        return output
