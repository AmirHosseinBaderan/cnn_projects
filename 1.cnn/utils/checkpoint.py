import os
import torch


def save_checkpoint(model, path, epoch, optimizer, valid_loss, valid_accuracy):
    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": valid_accuracy,
        "loss": valid_loss,
    }, "best_model.pth")
