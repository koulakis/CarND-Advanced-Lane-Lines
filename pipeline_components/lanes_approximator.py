from sklearn.base import BaseEstimator
from functools import reduce
import numpy as np
import cv2
import matplotlib.pyplot as plt


class LanePixelsFinder:
    def __init__(self, image, nwindows, margin, minpix):
        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)

        midpoint = np.int(histogram.shape[0] // 2)

        self.image = image
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.leftx_current = np.argmax(histogram[:midpoint])
        self.rightx_current = np.argmax(histogram[midpoint:]) + midpoint
        self.window_height = np.int(image.shape[0] // nwindows)
        self.nonzero = image.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])
        self.window = 0

    def __next_window(self):
        image, leftx_current, rightx_current, window, window_height, margin, minpix, nonzero, nonzeroy, nonzerox = [
            self.image, self.leftx_current, self.rightx_current, self.window, self.window_height,
            self.margin, self.minpix, self.nonzero, self.nonzeroy, self.nonzerox]

        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height

        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

        good_left_inds = (
                (win_xleft_low <= nonzerox)
                & (nonzerox < win_xleft_high)
                & (win_y_low <= nonzeroy)
                & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = (
                (win_xright_low <= nonzerox)
                & (nonzerox < win_xright_high)
                & (win_y_low <= nonzeroy)
                & (nonzeroy < win_y_high)).nonzero()[0]

        self.leftx_current = (
            int(nonzerox[good_left_inds].mean())
            if len(good_left_inds) > minpix
            else leftx_current)

        self.rightx_current = (
            int(nonzerox[good_right_inds].mean())
            if len(good_right_inds) > minpix
            else rightx_current)

        boxes = {
            'left': [(win_xleft_low, win_y_low), (win_xleft_high, win_y_high)],
            'right': [(win_xright_low, win_y_low), (win_xright_high, win_y_high)]}

        self.window += 1

        return good_left_inds, good_right_inds, boxes

    @staticmethod
    def find_pixels(image, nwindows, margin, minpix):
        pixel_finder = LanePixelsFinder(image, nwindows, margin, minpix)

        pixels_in_boxes = reduce(
            lambda acc, _: acc + [pixel_finder.__next_window()],
            range(nwindows),
            [])

        left_lane_inds, right_lane_inds, boxes = zip(*pixels_in_boxes)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        boxes = boxes

        leftx, lefty = pixel_finder.nonzerox[left_lane_inds], pixel_finder.nonzeroy[left_lane_inds]
        rightx, righty = pixel_finder.nonzerox[right_lane_inds], pixel_finder.nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, boxes


class LanePixelsFinderFromPolynomial:
    @staticmethod
    def find_pixels(image, left_fit, right_fit, margin):
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (np.abs(nonzerox - np.polyval(left_fit, nonzeroy)) < margin).nonzero()
        right_lane_inds = (np.abs(nonzerox - np.polyval(right_fit, nonzeroy)) < margin).nonzero()

        return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], nonzerox[right_lane_inds], nonzeroy[right_lane_inds]


class LaneApproximator(BaseEstimator):
    def __init__(self,  nwindows=9, margin=100, minpix=50, plot_approximation=False):
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.plot_approximation = plot_approximation

    @staticmethod
    def __fit_polynomials(leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    @staticmethod
    def plot_boxes_and_fitted_polynomials(image, boxes_info, polynomial_info):
        leftx, lefty, rightx, righty, boxes = boxes_info
        left_fitx, right_fitx, ploty = polynomial_info

        out_img = np.dstack((image, image, image))
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        for box in boxes:
            cv2.rectangle(
                out_img,
                *box['left'],
                (0, 255, 0),
                2)
            cv2.rectangle(
                out_img,
                *box['right'],
                (0, 255, 0),
                2)

        plt.figure(figsize=(10, 10))
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.imshow(out_img)
        plt.show()

    @staticmethod
    def plot_around_polynomial_curve(image, curve_info, polynomial_info, margin):
        leftx, lefty, rightx, righty = curve_info
        left_fitx, right_fitx, ploty = polynomial_info

        out_img = np.dstack((image, image, image))
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        window_img = np.zeros_like(out_img)
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        plt.figure(figsize=(10, 10))
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.imshow(result)

    def fit(self):
        return self

    def transform(self, image, pipeline_state):
        left_fit_state, right_fit_state = [pipeline_state[key] for key in ['left_fit', 'right_fit']]

        boxes_info = (
            LanePixelsFinder.find_pixels(image, self.nwindows, self.margin, self.minpix)
            if left_fit_state is None
            else LanePixelsFinderFromPolynomial.find_pixels(image, left_fit_state, right_fit_state, self.margin))

        leftx, lefty, rightx, righty = boxes_info[:4]
        left_fit, right_fit = LaneApproximator.__fit_polynomials(leftx, lefty, rightx, righty)

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)
        polynomial_info = left_fitx, right_fitx, ploty

        if self.plot_approximation:
            if left_fit_state is None:
                LaneApproximator.plot_boxes_and_fitted_polynomials(image, boxes_info, polynomial_info)
            else:
                LaneApproximator.plot_around_polynomial_curve(image, boxes_info, polynomial_info, self.margin)

        return left_fitx, right_fitx, ploty, left_fit, right_fit
