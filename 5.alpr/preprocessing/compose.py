from .transforms import Transform
from domain import Annotation

class Compose(Transform):

    def __init__(
        self,
        transforms,
    ):
        self.transforms = transforms

    def __call__(
        self,
        image,
        annotation: Annotation,
    ):
        for transform in self.transforms:
            image, annotation = transform(
                image,
                annotation,
            )

        return image, annotation