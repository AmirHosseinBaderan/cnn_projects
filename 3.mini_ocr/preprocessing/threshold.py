import numpy as np


def binary_threshold(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    height, width = image.shape

    output = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if image[y, x] > threshold:
                output[y, x] = 255
            else:
                output[y, x] = 0

    return output
