from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DetectionTarget:
    boxes: torch.Tensor
    objectness: torch.Tensor
    classes: torch.Tensor

    def to(self, device):
        return DetectionTarget(
            boxes=self.boxes.to(device),
            objectness=self.objectness.to(device),
            classes=self.classes.to(device),
        )