import cv2
import numpy as np
import torch


class FaceTransform:
    def __init__(self, size=112):
        self.size = size

    def __call__(self, img):
        img = cv2.resize(img, (self.size, self.size))

        img = img.astype("float32")
        img = img / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        img = torch.tensor(
            img,
            dtype=torch.float32
        )

        return img
