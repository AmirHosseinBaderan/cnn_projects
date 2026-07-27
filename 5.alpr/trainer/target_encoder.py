import torch


class TargetEncoder:

    def __init__(
        self,
        image_size=640,
        stride=32,
        num_classes=1,
    ):
        self.image_size = image_size
        self.stride = stride

        self.grid_size = image_size // stride

        self.num_classes = num_classes

    def encode(
        self,
        annotation,
    ):

        boxes = torch.zeros(
            4,
            self.grid_size,
            self.grid_size,
        )

        objectness = torch.zeros(
            1,
            self.grid_size,
            self.grid_size,
        )

        classes = torch.zeros(
            1,
            self.grid_size,
            self.grid_size,
            dtype=torch.long,
        )

        return {
            "boxes": boxes,
            "objectness": objectness,
            "classes": classes,
        }