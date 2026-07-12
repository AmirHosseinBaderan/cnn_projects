import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.image_dataset import get_datasets
from models.cnn import CNN

from engine.trainer import train_one_epoch
from engine.evaluator import validate
from utils.checkpoint import save_checkpoint
from utils.visualize import plot_loss,plot_accuracy
from config import Config

def main():
    config = Config()
    device = config.DEVICE

    print(f"Device : {device}")

    best_accuracy = 0

    train_dataset, valid_dataset, _ = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    history = {

        "train_loss": [],
        "valid_loss": [],

        "train_accuracy": [],
        "valid_accuracy": []

    }

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        valid_loss, valid_accuracy = validate(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
        )

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            save_checkpoint(
                model,
                "best_model.pth",
                epoch,
                optimizer,
                valid_loss,
                valid_accuracy,
            )

            print("best model saved")

        history["train_loss"].append(train_loss)

        history["valid_loss"].append(valid_loss)

        history["train_accuracy"].append(train_accuracy)

        history["valid_accuracy"].append(valid_accuracy)

        print("-" * 60)

        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS}"
        )

        print(
            f"Train Loss      : {train_loss:.4f}"
        )

        print(
            f"Validation Loss : {valid_loss:.4f}"
        )

        print(
            f"Validation Acc  : {valid_accuracy:.2f}%"
        )

    plot_loss(history)
    plot_accuracy(history)

if __name__ == "__main__":
    main()
