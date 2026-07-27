import torch


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

    def encode(self, annotation):

        boxes = torch.zeros(
            self.grid_size,
            self.grid_size,
            4,
            dtype=torch.float32,
        )

        objectness = torch.zeros(
            self.grid_size,
            self.grid_size,
            dtype=torch.float32,
        )

        classes = torch.zeros(
            self.grid_size,
            self.grid_size,
            dtype=torch.long,
        )

        for plate in annotation.plates:
            self._encode_plate(
                plate,
                boxes,
                objectness,
                classes,
            )

        return {
            "boxes": boxes,
            "objectness": objectness,
            "classes": classes,
        }

    def _encode_plate(
        self,
        plate,
        boxes,
        objectness,
        classes,
    ):
        pass