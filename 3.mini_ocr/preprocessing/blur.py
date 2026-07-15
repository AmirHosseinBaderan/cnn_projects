import cv2
import numpy as np


def gaussian_blur(
        image: np.ndarray,
        kernel_size=(5, 5),
        sigma=0
):
    return cv2.GaussianBlur(image, kernel_size, sigma)


def median_blur(
        image: np.ndarray,
        kernel_size=5
):
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(
        image: np.ndarray,
        d=9,
        sigma_color=75,
        sigma_space=75,
):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
