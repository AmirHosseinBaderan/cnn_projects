import os
import torch

from .logger import logger


def save_checkpoint(
        model,
        path,
        epoch,
        optimizer,
        train_loss,
):
    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
        },
        path
    )

    logger.info(
        f"Checkpoint saved : epoch {epoch + 1}"
    )


def load_checkpoint(
        path,
        model,
        optimizer=None
):

    logger.info(
        "Loading checkpoint"
    )

    if not os.path.exists(path):
        logger.info(
            "Checkpoint not found"
        )

        return 0

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    logger.info(
        f"Checkpoint loaded from epoch {checkpoint['epoch'] + 1}"
    )
    return checkpoint["epoch"] + 1