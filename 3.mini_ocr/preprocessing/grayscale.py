import numpy as np


def cvtGrayScale(image):
    height, width = image.shape[:2]

    new_image = np.zeros((height, width), np.uint8)

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]
            gray = int(
                0.299 * r +
                0.587 * g +
                0.114 * b
            )
            new_image[y, x] = gray

    return new_image
