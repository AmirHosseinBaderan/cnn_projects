from collections.abc import Iterable

import torch
from torch import nn

from trainer.target_encoder import TargetEncoder


def validate(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    encoder: TargetEncoder,
    device: torch.device,
) -> dict[str, float]:

    model.eval()

    total_loss = 0.0
    total_box_loss = 0.0
    total_objectness_loss = 0.0
    total_classification_loss = 0.0

    num_batches = 0

    with torch.no_grad():

        for images, annotations in dataloader:

            images = images.to(device)

            target = encoder.encode(
                annotations
            ).to(device)

            prediction = model(images)

            losses = criterion(
                prediction,
                target,
            )

            total_loss += losses["loss"].item()
            total_box_loss += losses["box_loss"].item()
            total_objectness_loss += losses["objectness_loss"].item()
            total_classification_loss += losses["classification_loss"].item()

            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "box_loss": total_box_loss / num_batches,
        "objectness_loss": total_objectness_loss / num_batches,
        "classification_loss": total_classification_loss / num_batches,
    }