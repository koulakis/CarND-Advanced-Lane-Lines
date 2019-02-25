from sklearn.base import BaseEstimator
import cv2
import numpy as np

from .utils import annotate_image_with_mask
from .pipeline_state import TransformContext, TransformError


class LaneAnnotator(BaseEstimator):
    def __init__(self, inverse_perspective_transformer, bottom_left_corner_text=(50, 60)):
        self.inverse_perspective_transformer = inverse_perspective_transformer
        self.bottom_left_corner_text = bottom_left_corner_text

    @staticmethod
    def mask_between_lanes(image, left_fitx, right_fitx, ploty, inverse_perspective_transformer):
        left_fitx, right_fitx, ploty = [x.astype(int) for x in [left_fitx, right_fitx, ploty]]

        try:
            fit_img = np.zeros(image.shape)
            fit_img[ploty, left_fitx, :] = (255, 255, 0)
            fit_img[ploty, right_fitx, :] = (255, 255, 0)
        except IndexError as e:
            raise TransformError('x value of fitted polynomial outside image bounds', e)

        for x_l, x_r, y in zip(left_fitx, right_fitx, ploty.astype(int)):
            fit_img[y, x_l:x_r, :] = (0, 255, 0)

        return inverse_perspective_transformer.transform({'data': fit_img})['data']

    def draw_radius_and_location_text(self, image, radius, location):
        image_copy = image.copy()

        cv2.putText(
            image_copy,
            'Radius of Curvature: {:.0f}m'.format(radius),
            self.bottom_left_corner_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            thickness=3)

        cv2.putText(
            image_copy,
            'Vehicle {:.2f}m {} of center'.format(np.abs(location), 'left' if location < 0 else 'right'),
            (self.bottom_left_corner_text[0], self.bottom_left_corner_text[1] + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            thickness=3)

        return image_copy

    def fit(self):
        return self

    def transform(self, stateful_data):
        with TransformContext(self.__class__.__name__, stateful_data) as s:
            left_fitx, right_fitx, ploty, _, _, radius, location = s['data']
            image = s['cached_image']

            lines_mask = LaneAnnotator.mask_between_lanes(
                image,
                left_fitx,
                right_fitx,
                ploty,
                self.inverse_perspective_transformer)

            image_with_lanes = annotate_image_with_mask(image, lines_mask)
            annotated_image = self.draw_radius_and_location_text(image_with_lanes, radius, location)

            s['data'] = annotated_image

        return stateful_data
