from abc import ABC, abstractmethod

import cv2
import torch

from domain import (
    Annotation,
    ImageInfo,
)


class Transform(ABC):

    @abstractmethod
    def __call__(
        self,
        image,
        annotation: Annotation,
    ):
        raise NotImplementedError


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


class Resize(Transform):

    def __init__(
        self,
        width: int,
        height: int,
    ):
        self.width = width
        self.height = height

    def __call__(
        self,
        image,
        annotation: Annotation,
    ):
        original_height, original_width = image.shape[:2]

        scale_x = self.width / original_width
        scale_y = self.height / original_height

        resized_image = cv2.resize(
            image,
            (self.width, self.height),
        )

        resized_annotation = Annotation(
            image=ImageInfo(
                filename=annotation.image.filename,
                folder=annotation.image.folder,
                width=self.width,
                height=self.height,
                depth=annotation.image.depth,
            )
        )

        for obj in annotation.objects:

            resized_annotation.add_object(
                obj.with_bbox(
                    obj.bbox.scale(
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )
                )
            )

        return resized_image, resized_annotation
    
class ToTensor(Transform):
    def __call__(self, image, annotation):
        image = torch.from_numpy(image)
        image = image.permute(
            2,0,1
        ).float()
        
        return image,annotation
    
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