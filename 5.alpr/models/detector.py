import torch
import torch.nn as nn

from models.modules.backbone import Backbone
from models.modules.head import DetectionHead


class Detector(nn.Module):

    def __init__(
        self,
        num_classes: int = 1,
    ):
        super().__init__()

        self.backbone = Backbone()

        self.head = DetectionHead(
            in_channels=512,
            num_classes=num_classes,
        )

    def forward(self, x):

        p3, p4, p5 = self.backbone(x)

        prediction = self.head(p5)

        return prediction