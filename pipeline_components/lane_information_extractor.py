from sklearn.base import BaseEstimator
import numpy as np

from .pipeline_state import TransformContext


class LaneInformationExtractor(BaseEstimator):
    def __init__(self, xm_per_pix=3.7 / 700, ym_per_pix=30 / 720):
        self.xm_per_pix = xm_per_pix
        self.ym_per_pix = ym_per_pix

    def measure_curvature(self, leftx, rightx, ploty):
        left_fit = np.polyfit(self.ym_per_pix * ploty, self.xm_per_pix * leftx, 2)
        right_fit = np.polyfit(self.ym_per_pix * ploty, self.xm_per_pix * rightx, 2)

        y_eval = np.max(ploty)

        left_curverad = (1 + (2 * left_fit[0] * self.ym_per_pix * y_eval + left_fit[1]) ** 2) ** 1.5 / (
                    2 * left_fit[0])
        right_curverad = (1 + (2 * right_fit[0] * self.ym_per_pix * y_eval + right_fit[1]) ** 2) ** 1.5 / (
                    2 * right_fit[0])

        return (left_curverad + right_curverad) / 2

    def estimate_relative_vehicle_position(self, image, leftx, rightx):
        lane_center = (leftx[-1] + rightx[-1]) / 2
        vehicle_center = image.shape[1] / 2

        return self.xm_per_pix * (vehicle_center - lane_center)

    def fit(self):
        return self

    def transform(self, stateful_data):
        with TransformContext(self.__class__.__name__, stateful_data) as s:
            left_fitx, right_fitx, ploty, left_fit, right_fit = s['data']

            s['data'] = [
                left_fitx, right_fitx, ploty, left_fit, right_fit,
                self.measure_curvature(left_fitx, right_fitx, ploty),
                self.estimate_relative_vehicle_position(s['cached_image'], left_fitx, right_fitx)]

        return stateful_data
