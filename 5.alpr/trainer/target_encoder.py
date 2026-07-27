import torch

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
        annotation,
    ) -> DetectionTarget:

        target = DetectionTarget(
            boxes=torch.zeros(
                self.grid_size,
                self.grid_size,
                4,
                dtype=torch.float32,
            ),
            objectness=torch.zeros(
                self.grid_size,
                self.grid_size,
                dtype=torch.float32,
            ),
            classes=torch.zeros(
                self.grid_size,
                self.grid_size,
                dtype=torch.long,
            ),
        )

        for plate in annotation.plates:
            self._encode_plate(
                plate,
                target,
            )

        return target

    def _encode_plate(
        self,
        plate,
        target: DetectionTarget,
    ) -> None:
    
        bbox = plate.bbox
    
        cx = (bbox.xmin + bbox.xmax) / 2
        cy = (bbox.ymin + bbox.ymax) / 2
    
        width = bbox.xmax - bbox.xmin
        height = bbox.ymax - bbox.ymin
    
        grid_x = int(cx // self.stride)
        grid_y = int(cy // self.stride)
    
        if not (
            0 <= grid_x < self.grid_size
            and
            0 <= grid_y < self.grid_size
        ):
            return
    
        target.objectness[grid_y, grid_x] = 1.0
    
        target.boxes[grid_y, grid_x] = torch.tensor(
            [
                cx,
                cy,
                width,
                height,
            ],
            dtype=torch.float32,
        )
    
        target.classes[grid_y, grid_x] = 0