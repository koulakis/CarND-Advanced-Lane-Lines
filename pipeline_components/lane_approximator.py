from sklearn.base import BaseEstimator
from functools import reduce
import numpy as np
import cv2

from .pipeline_state import TransformContext, TransformError
from .utils import annotate_image_with_mask


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
    def __init__(self, nwindows=9, margin=100, minpix=50, overwrite_image=False, smoothing=None, max_coeff_divergence=.7):
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix
        self.overwrite_image = overwrite_image
        self.smoothing = smoothing
        self.max_coeff_divergence = max_coeff_divergence

    @staticmethod
    def __fit_polynomials(leftx, lefty, rightx, righty):
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    @staticmethod
    def plot_boxes_and_fitted_polynomials(image, boxes_info):
        leftx, lefty, rightx, righty, boxes = boxes_info

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

        return out_img

    @staticmethod
    def plot_around_polynomial_curve(image, curve_info, polynomial_info, margin):
        leftx, lefty, rightx, righty = curve_info
        left_fitx, right_fitx, ploty = polynomial_info

        out_img = np.dstack((image, image, image))
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        left_line_window1 = np.array([
            np.transpose(np.vstack([np.maximum(left_fitx - margin, 0), ploty]))])
        left_line_window2 = np.array([
            np.flipud(np.transpose(np.vstack([np.minimum(left_fitx + margin, image.shape[1] - 1), ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([np.maximum(right_fitx - margin, 0), ploty]))])
        right_line_window2 = np.array([
            np.flipud(np.transpose(np.vstack([np.minimum(right_fitx + margin, image.shape[1] - 1), ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        window_img = np.zeros_like(out_img)
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        return cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    def fit(self):
        return self

    @staticmethod
    def restrict_x_in_image(x_vals, image):
        return np.where(0 > x_vals, 0, np.where(x_vals >= image.shape[1], image.shape[1] - 1, x_vals))

    def average_with_previous_steps(self, current_value, steps, get_step_value):
        decay_coefficients, values = map(np.array, zip(
            *([([smooth], get_step_value(step))
               for step, smooth in zip(steps[-len(self.smoothing):], self.smoothing)
               if get_step_value(step) is not None]
              + [([1.], current_value)] if current_value is not None else [])))

        return (decay_coefficients * values).sum(axis=0) / decay_coefficients.sum()

    @staticmethod
    def check_data_existence(leftx, lefty, rightx, righty):
        if any([len(pointset) < 5 for pointset in [leftx, lefty, rightx, righty]]):
            raise TransformError(
                'insufficient data to fit polynomials: leftx: {}, lefty: {}, rightx: {}, righty: {}'.format(
                    *[str(len(pointset)) for pointset in [leftx, lefty, rightx, righty]]))

    def bad_quality_fits(self, steps):
        left_right_steps = np.array([[step.left_fit, step.right_fit] for step in steps])
        differences = np.abs(left_right_steps[:, 0, :] - left_right_steps[:, 1, :])
        magnitude = np.abs(left_right_steps[:, 0, :]) + np.abs(left_right_steps[:, 1, :])

        return (differences / magnitude).mean(axis=0).mean() > self.max_coeff_divergence

    def transform(self, stateful_data):
        with TransformContext(self.__class__.__name__, stateful_data) as s:
            image, state = s['data'], s['steps'][-1]

            non_empty_fits = [
                state for state in s['steps'][-len(self.smoothing):]
                if (state.left_fit is not None) and (state.right_fit is not None)]

            find_lanes_from_scratch = (len(non_empty_fits) == 0) or self.bad_quality_fits(non_empty_fits)

            boxes_info = (
                LanePixelsFinder.find_pixels(image, self.nwindows, self.margin, self.minpix)
                if find_lanes_from_scratch
                else LanePixelsFinderFromPolynomial.find_pixels(
                    image, non_empty_fits[-1].left_fit, non_empty_fits[-1].right_fit, self.margin))

            leftx, lefty, rightx, righty = boxes_info[:4]
            LaneApproximator.check_data_existence(leftx, lefty, rightx, righty)

            left_fit_current, right_fit_current = LaneApproximator.__fit_polynomials(
                leftx, lefty, rightx, righty)

            left_fit = self.average_with_previous_steps(
                left_fit_current, s['steps'], lambda x: x.left_fit)

            right_fit = self.average_with_previous_steps(
                right_fit_current, s['steps'], lambda x: x.right_fit)

            ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

            left_fitx = LaneApproximator.restrict_x_in_image(np.polyval(left_fit, ploty), image)
            right_fitx = LaneApproximator.restrict_x_in_image(np.polyval(right_fit, ploty), image)

            polynomial_info = left_fitx, right_fitx, ploty

            if self.overwrite_image is not None:
                s['cached_image'] = (
                    annotate_image_with_mask(
                        s['cached_image'],
                        (LaneApproximator.plot_boxes_and_fitted_polynomials(image, boxes_info)
                         if find_lanes_from_scratch
                         else LaneApproximator.plot_around_polynomial_curve(image, boxes_info, polynomial_info,
                                                                            self.margin)),
                        alpha=self.overwrite_image))

            state.set_left_fit(left_fit)
            state.set_right_fit(right_fit)
            s['data'] = [left_fitx, right_fitx, ploty, left_fit, right_fit]

        return stateful_data
