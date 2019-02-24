from sklearn.base import BaseEstimator
import cv2
import numpy as np

from .utils import update_dictionary


class LaneAnnotator(BaseEstimator):
    def __init__(self, inverse_perspective_transformer, bottom_left_corner_text=(50, 60)):
        self.inverse_perspective_transformer = inverse_perspective_transformer
        self.bottom_left_corner_text = bottom_left_corner_text

    @staticmethod
    def mask_between_lanes(image, left_fitx, right_fitx, ploty, inverse_perspective_transformer):
        left_fitx, right_fitx, ploty = [x.astype(int) for x in [left_fitx, right_fitx, ploty]]

        fit_img = np.zeros(image.shape)
        fit_img[ploty, left_fitx, :] = (255, 255, 0)
        fit_img[ploty, right_fitx, :] = (255, 255, 0)

        for x_l, x_r, y in zip(left_fitx, right_fitx, ploty.astype(int)):
            fit_img[y, x_l:x_r, :] = (0, 255, 0)

        return inverse_perspective_transformer.transform({'X': fit_img, 'state': {}})['X']

    @staticmethod
    def draw_area_between_lines_in_image(image, lines_mask, alpha=0.3):
        return np.minimum(255, image + alpha * lines_mask).astype('uint8')

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
        left_fitx, right_fitx, ploty, _, _, radius, location = stateful_data['X']
        image = stateful_data['state']['image']

        lines_mask = LaneAnnotator.mask_between_lanes(
            image,
            left_fitx,
            right_fitx,
            ploty,
            self.inverse_perspective_transformer)

        image_with_lanes = LaneAnnotator.draw_area_between_lines_in_image(image, lines_mask)
        annotated_image = self.draw_radius_and_location_text(image_with_lanes, radius, location)

        return update_dictionary(
            stateful_data,
            {'X': annotated_image})
