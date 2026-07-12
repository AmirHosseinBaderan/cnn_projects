from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

import torch.nn as nn


def build_resnet(num_classes=10):

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model