from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DetectionPrediction:
    boxes: torch.Tensor
    objectness: torch.Tensor
    classes: torch.Tensor