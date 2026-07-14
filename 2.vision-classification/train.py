from torch import optim

from datasets.intel_dataset import IntelDataset
from torch.utils.data import DataLoader
from transforms.transforms import train_transform, test_transform
from models.simple_cnn import SimpleCNN
import torch
from utils.logger import logger
import torch.nn as nn
from utils.checkpoint import save_checkpoint, load_checkpoint
from evaluator import validate

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_PATH = "checkpoints/model.pth"
PATIENCE = 5


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"running on {device}")

    train_dataset = IntelDataset(
        "data/raw/seg_train",
        transform=train_transform,
    )

    test_dataset = IntelDataset(
        "data/raw/seg_test",
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_accuracy": [],
        "valid_accuracy": [],
    }

    start_epoch = load_checkpoint(MODEL_PATH, model, optimizer)

    early_stopping_counter = 0
    best_loss = float("inf")
    best_accuracy = 0

    for epoch in range(start_epoch, EPOCHS):
        logger.info(f"start training epoch {epoch + 1}")
        model.train()

        running_loss = 0
        running_correct = 0
        running_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward
            outputs = model(images)
            # loss
            loss = criterion(outputs, labels)

            correct = calculate_correct(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += correct
            running_total += labels.size(0)

            if batch_idx % 20 == 0:
                logger.info(
                    f"epoch={epoch + 1} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        valid_loss, valid_accuracy, _, _ = validate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step(valid_loss)
        logger.info(
            f"Current LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        epoch_acc = running_correct / running_total
        epoch_loss = running_loss / running_total

        logger.info(f"epoch {epoch + 1}, loss: {epoch_loss}, acc: {epoch_acc}")

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            save_checkpoint(
                model,
                path=MODEL_PATH,
                epoch=epoch,
                valid_loss=valid_loss,
                optimizer=optimizer,
                valid_accuracy=valid_accuracy,
            )

        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_accuracy"].append(valid_accuracy)

        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= PATIENCE:
            logger.info("Early stopping")
            logger.info(
                f"Validation loss did not improve for {PATIENCE} epochs. Stopping training."
            )
            break


def calculate_correct(outputs, labels):
    predicted = outputs.argmax(dim=1)
    return (predicted == labels).sum().item()


if __name__ == "__main__":
    train()
