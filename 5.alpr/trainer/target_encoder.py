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
    
        width = bbox.xmax - bbox.xmin
        height = bbox.ymax - bbox.ymin
    
        if width <= 0 or height <= 0:
            return
    
        cx = (bbox.xmin + bbox.xmax) * 0.5
        cy = (bbox.ymin + bbox.ymax) * 0.5
    
        grid_x = int(cx // self.stride)
        grid_y = int(cy // self.stride)
    
        if not (
            0 <= grid_x < self.grid_size
            and
            0 <= grid_y < self.grid_size
        ):
            return
    
        # TODO:
        # Currently each grid cell supports only one object.
        # Later this can be extended to multiple anchors.
    
        cell_x = grid_x * self.stride
        cell_y = grid_y * self.stride
    
        tx = (cx - cell_x) / self.stride
        ty = (cy - cell_y) / self.stride
    
        tw = width / self.image_size
        th = height / self.image_size
    
        target.objectness[grid_y, grid_x] = 1.0
    
        target.boxes[grid_y, grid_x, 0] = tx
        target.boxes[grid_y, grid_x, 1] = ty
        target.boxes[grid_y, grid_x, 2] = tw
        target.boxes[grid_y, grid_x, 3] = th
    
        target.classes[grid_y, grid_x] = 0