from pathlib import Path

import numpy as np
from PIL import Image


class ImagePreprocessor:
    def __init__(self, transform):
        self.transform = transform

    def preprocess(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, Path):
            image = Image.open(image)

        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        elif not isinstance(image, Image.Image):
            raise TypeError(
                f"Unsupported image type: {type(image)}"
            )

        image = image.convert("L")
        image = self.transform(image)
        image = image.unsqueeze(0)

        return image