from pathlib import Path

import torch

from utils.logger import logger


class CheckpointManager:

    def __init__(self, checkpoint_dir):

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_loss = float("inf")

    def save_best(
            self,
            model,
            optimizer,
            epoch,
            loss,
    ):

        if loss >= self.best_loss:
            return False

        self.best_loss = loss

        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            self.checkpoint_dir / "best.pt",
        )

        logger.info(
            f"Best model saved "
            f"(loss={loss:.4f})",
        )

        return True

    def save_last(
            self,
            model,
            optimizer,
            epoch,
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            self.checkpoint_dir / "last.pt",
        )

    @staticmethod
    def load(
            model,
            path,
            device,
            optimizer=None,
    ):

        logger.info("Loading checkpoint")

        if not Path(path).exists():
            logger.info("Checkpoint not found")
            return 0

        checkpoint = torch.load(
            path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"],
            )

        logger.info(
            f"Checkpoint loaded from epoch {checkpoint['epoch'] + 1}",
        )

        return checkpoint["epoch"] + 1
