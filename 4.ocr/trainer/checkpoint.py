from pathlib import Path

import torch


class CheckpointManager:

    def __init__(self, checkpoint_dir):

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.best_loss = float("inf")

    def save_best(
            self,
            model,
            optimizer,
            epoch,
            loss
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
                optimizer.state_dict()

        }

        torch.save(
            checkpoint,
            self.checkpoint_dir / "best.pt"
        )

        print(
            f"Best model saved "
            f"(loss={loss:.4f})"
        )

        return True

    def save_last(
            self,
            model,
            optimizer,
            epoch,
            loss
    ):
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict()

        }

        torch.save(
            checkpoint,
            self.checkpoint_dir / "last.pt"
        )

    @staticmethod
    def load(
            model,
            optimizer,
            path,
            device
    ):
        checkpoint = torch.load(
            path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        return checkpoint["epoch"], checkpoint["loss"]