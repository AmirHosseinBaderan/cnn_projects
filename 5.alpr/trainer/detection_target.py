from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DetectionTarget:
    boxes: torch.Tensor
    objectness: torch.Tensor
    classes: torch.Tensor