from .transforms import Transform
from domain import Annotation,ImageInfo
import cv2

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