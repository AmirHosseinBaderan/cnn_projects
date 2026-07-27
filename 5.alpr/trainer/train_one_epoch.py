from collections.abc import Iterable

import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer

from trainer.target_encoder import TargetEncoder


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    optimizer: Optimizer,
    encoder: TargetEncoder,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:

    model.train()

    total_loss = 0.0
    total_box_loss = 0.0
    total_objectness_loss = 0.0
    total_classification_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Train : {epoch+1}",
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )

    num_batches = 0

    for batch_idx, (images, annotations) in enumerate(progress_bar, start=1):

        images = images.to(device)

        target = encoder.encode(annotations).to(device)

        optimizer.zero_grad()

        prediction = model(images)

        losses = criterion(
            prediction,
            target,
        )

        losses["loss"].backward()

        optimizer.step()

        total_loss += losses["loss"].item()
        total_box_loss += losses["box_loss"].item()
        total_objectness_loss += losses["objectness_loss"].item()
        total_classification_loss += losses["classification_loss"].item()

        num_batches += 1

        average_loss = total_loss / num_batches
        average_box_loss = total_box_loss / num_batches
        average_objectness_loss = total_objectness_loss / num_batches
        average_classification_loss = total_classification_loss / num_batches

        progress_bar.set_postfix(
            loss=f"{average_loss:.4f}",
            box=f"{average_box_loss:.4f}",
            obj=f"{average_objectness_loss:.4f}",
            cls=f"{average_classification_loss:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    return {
        "loss": total_loss / num_batches,
        "box_loss": total_box_loss / num_batches,
        "objectness_loss": total_objectness_loss / num_batches,
        "classification_loss": total_classification_loss / num_batches,
    }
