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
