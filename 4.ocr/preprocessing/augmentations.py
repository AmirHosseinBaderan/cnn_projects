import random

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF


class RandomRotation:

    def __init__(
        self,
        degrees=5,
        p=0.5,
    ):
        self.degrees = degrees
        self.p = p

    def __call__(self, image):

        if random.random() > self.p:
            return image

        angle = random.uniform(
            -self.degrees,
            self.degrees,
        )

        return TF.rotate(image, angle)


class RandomBrightnessContrast:

    def __init__(
        self,
        brightness=0.3,
        contrast=0.3,
        p=0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, image):

        if random.random() > self.p:
            return image

        brightness = random.uniform(
            1 - self.brightness,
            1 + self.brightness,
        )

        contrast = random.uniform(
            1 - self.contrast,
            1 + self.contrast,
        )

        image = TF.adjust_brightness(
            image,
            brightness,
        )

        image = TF.adjust_contrast(
            image,
            contrast,
        )

        return image


class RandomGaussianNoise:

    def __init__(
        self,
        sigma=10,
        p=0.3,
    ):
        self.sigma = sigma
        self.p = p

    def __call__(self, image):

        if random.random() > self.p:
            return image

        image = np.array(image).astype(np.float32)

        noise = np.random.normal(
            0,
            self.sigma,
            image.shape,
        )

        image = image + noise

        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

        return Image.fromarray(image)


class RandomGaussianBlur:

    def __init__(
        self,
        p=0.3,
    ):
        self.p = p

    def __call__(self, image):

        if random.random() > self.p:
            return image

        image = np.array(image)

        image = cv2.GaussianBlur(
            image,
            (3, 3),
            0,
        )

        return Image.fromarray(image)


class RandomMotionBlur:

    def __init__(
        self,
        kernel_size=5,
        p=0.3,
    ):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, image):

        if random.random() > self.p:
            return image

        image = np.array(image)

        kernel = np.zeros(
            (
                self.kernel_size,
                self.kernel_size,
            )
        )

        kernel[
            self.kernel_size // 2,
            :
        ] = np.ones(self.kernel_size)

        kernel /= self.kernel_size

        image = cv2.filter2D(
            image,
            -1,
            kernel,
        )

        return Image.fromarray(image)


class RandomPerspective:

    def __init__(
        self,
        distortion=0.1,
        p=0.3,
    ):
        self.distortion = distortion
        self.p = p

    def __call__(self, image):

        if random.random() > self.p:
            return image

        return TF.perspective(
            image,
            startpoints=[
                [0, 0],
                [image.width, 0],
                [image.width, image.height],
                [0, image.height],
            ],
            endpoints=[
                [
                    random.randint(
                        0,
                        int(image.width * self.distortion),
                    ),
                    random.randint(
                        0,
                        int(image.height * self.distortion),
                    ),
                ],
                [
                    image.width - random.randint(
                        0,
                        int(image.width * self.distortion),
                    ),
                    random.randint(
                        0,
                        int(image.height * self.distortion),
                    ),
                ],
                [
                    image.width - random.randint(
                        0,
                        int(image.width * self.distortion),
                    ),
                    image.height - random.randint(
                        0,
                        int(image.height * self.distortion),
                    ),
                ],
                [
                    random.randint(
                        0,
                        int(image.width * self.distortion),
                    ),
                    image.height - random.randint(
                        0,
                        int(image.height * self.distortion),
                    ),
                ],
            ],
        )