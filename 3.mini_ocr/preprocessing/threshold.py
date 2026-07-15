import numpy as np
import cv2


# def binary_threshold(image: np.ndarray, threshold: int = 128) -> np.ndarray:
#     height, width = image.shape
#
#     output = np.zeros((height, width), dtype=np.uint8)
#
#     for y in range(height):
#         for x in range(width):
#             if image[y, x] > threshold:
#                 output[y, x] = 255
#             else:
#                 output[y, x] = 0
#
#     return output

def binary_threshold(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    return np.where(image > threshold, 255, 0).astype(np.uint8)


def adaptive_threshold(
        image: np.ndarray,
        block_size: int = 11,
        c: int = 2
):
    return cv2.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c
    )


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    _, output = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return output
