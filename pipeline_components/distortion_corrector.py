from sklearn.base import BaseEstimator
import cv2
import matplotlib.pyplot as plt
import numpy as np


class DistortionCorrector(BaseEstimator):
    def __init__(self, corner_shape=(9, 6)):
        self.corner_shape = corner_shape
        self.matrix = None
        self.distortion_coefficients = None

    def compute_object_and_image_points(self, images, plot_corners=False):
        actual_corners = [
            cv2.findChessboardCorners(
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                self.corner_shape,
                None)
            for image in images]

        if plot_corners:
            for result_corners, image in zip(actual_corners, images):
                result, corners = result_corners
                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.drawChessboardCorners(image.copy(), self.corner_shape, corners, result))
                plt.show()

        canonical_3d_corner_positions = np.concatenate(
            [np.mgrid[0:self.corner_shape[0], 0:self.corner_shape[1]].T.reshape(-1, 2),
             np.zeros((self.corner_shape[0] * self.corner_shape[1], 1))],
            axis=1).astype('float32')

        image_points = [corners for result, corners in actual_corners if result]
        object_points = len(image_points) * [canonical_3d_corner_positions]

        return object_points, image_points

    def fit(self, images, plot_corners=False):
        object_points, image_points = self.compute_object_and_image_points(images, plot_corners=plot_corners)
        _, self.matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(
            object_points, image_points, images[0].shape[1::-1], None, None)
        return self

    def transform(self, stateful_data):
        output = stateful_data.copy()
        output['X'] = cv2.undistort(stateful_data['X'], self.matrix, self.distortion_coefficients)

        return output
