import numpy as np


def to_grayscale(image):
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)

    gray = 0.114 * b + 0.587 * g + 0.299 * r

    return gray.astype(np.uint8)
