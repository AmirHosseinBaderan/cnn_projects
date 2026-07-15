import cv2
import numpy as np


def erosion(
        image: np.ndarray,
        kernel_size=(3, 3),
        iterations=1,
):
    kernel = np.ones(kernel_size, np.uint8)

    return cv2.erode(
        image,
        kernel,
        iterations=iterations
    )


def dilation(
        image: np.ndarray,
        kernel_size=(3, 3),
        iterations=1,
):
    kernel = np.ones(kernel_size, np.uint8)

    return cv2.dilate(
        image,
        kernel,
        iterations=iterations
    )


def opening(
        image,
        kernel_size=(3, 3),
):
    kernel = np.ones(kernel_size, np.uint8)

    return cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel
    )

def closing(
        image,
        kernel_size=(3, 3),
):
    kernel = np.ones(kernel_size, np.uint8)

    return cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel
    )