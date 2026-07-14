import os
import torch
from .logger import logger


def save_checkpoint(
        model,
        path,
        epoch,
        optimizer,
        valid_loss,
        valid_accuracy,
):
    os.makedirs("checkpoints", exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": valid_accuracy,
        "loss": valid_loss,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    logger.info("load check point")

    if not os.path.exists(path):
        logger.info("check point not found")
        return 0

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    logger.info("check point loaded")
    return checkpoint["epoch"] + 1
