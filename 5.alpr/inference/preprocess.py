import cv2
import torch

from preprocessing import Normalize, Resize, ToTensor
from domain import Annotation, ImageInfo
from preprocessing.compose import Compose


class ImagePreprocessor:

    def __init__(
        self,
        image_size: int,
    ):
        self.transforms = Compose([
            Resize(
                width=image_size,
                height=image_size,
            ),
            ToTensor(),
            Normalize(),
        ])

    def preprocess(
        self,
        image,
    ) -> torch.Tensor:

        # Create a dummy annotation for the transforms that expect one
        # We don't have the original filename/folder, so use empty strings
        # and get height, width, depth from the image
        height, width = image.shape[:2]
        depth = image.shape[2] if len(image.shape) > 2 else 1
        annotation = Annotation(
            image=ImageInfo(
                filename="",
                folder="",
                width=width,
                height=height,
                depth=depth,
            ),
            objects=[]
        )

        image, _ = self.transforms(
            image,
            annotation,
        )

        return image.unsqueeze(0)