import torch

from domain.annotation import Annotation
from trainer.detection_target import DetectionTarget


class TargetEncoder:

    def __init__(
        self,
        image_size: int = 640,
        stride: int = 32,
        num_classes: int = 1,
    ):
        self.image_size = image_size
        self.stride = stride
        self.grid_size = image_size // stride
        self.num_classes = num_classes

    def encode(
        self,
        annotations: list[Annotation],
    ) -> DetectionTarget:

        batch_size = len(annotations)

        target = DetectionTarget(
            boxes=torch.zeros(
                batch_size,
                self.grid_size,
                self.grid_size,
                4,
                dtype=torch.float32,
            ),
            objectness=torch.zeros(
                batch_size,
                self.grid_size,
                self.grid_size,
                dtype=torch.float32,
            ),
            classes=torch.zeros(
                batch_size,
                self.grid_size,
                self.grid_size,
                dtype=torch.long,
            ),
        )

        for batch_index, annotation in enumerate(annotations):

            for plate in annotation.plates:
                self._encode_plate(
                    plate=plate,
                    target=target,
                    batch_index=batch_index,
                )

        return target

    def _encode_plate(
        self,
        plate,
        target: DetectionTarget,
        batch_index: int,
    ) -> None:

        bbox = plate.bbox

        width = bbox.width
        height = bbox.height

        if width <= 0 or height <= 0:
            return

        cx, cy = bbox.center

        grid_x = int(cx // self.stride)
        grid_y = int(cy // self.stride)

        if not (
            0 <= grid_x < self.grid_size
            and
            0 <= grid_y < self.grid_size
        ):
            return

        # Currently each grid cell supports only one object.
        # Later this can be extended to multiple anchors.

        cell_x = grid_x * self.stride
        cell_y = grid_y * self.stride

        tx = (cx - cell_x) / self.stride
        ty = (cy - cell_y) / self.stride

        tw = width / self.image_size
        th = height / self.image_size

        target.objectness[
            batch_index,
            grid_y,
            grid_x,
        ] = 1.0

        target.boxes[
            batch_index,
            grid_y,
            grid_x,
            0,
        ] = tx

        target.boxes[
            batch_index,
            grid_y,
            grid_x,
            1,
        ] = ty

        target.boxes[
            batch_index,
            grid_y,
            grid_x,
            2,
        ] = tw

        target.boxes[
            batch_index,
            grid_y,
            grid_x,
            3,
        ] = th

        # Plate class
        target.classes[
            batch_index,
            grid_y,
            grid_x,
        ] = 0