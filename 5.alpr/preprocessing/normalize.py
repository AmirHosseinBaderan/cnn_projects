from .transforms import Transform

class Normalize(Transform):

    def __init__(
        self,
        mean=None,
        std=None,
    ):
        self.mean = mean
        self.std = std

    def __call__(
        self,
        image,
        annotation,
    ):
        image /= 255.0

        if self.mean is not None and self.std is not None:
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        return image, annotation