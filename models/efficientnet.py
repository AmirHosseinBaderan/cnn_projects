from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

import torch.nn as nn


def build_efficientnet(
    num_classes=10,
    pretrained=True
):

    weights = (
        EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )

    model = efficientnet_b0(
        weights=weights
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model