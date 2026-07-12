import torch
from ..utils.metrics import accuracy


def train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
):
    model.train()

    running_loss = 0.0
    running_accuracy = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        batch_accuracy = accuracy(outputs, labels)

        running_accuracy += batch_accuracy

        # Backward
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_accuracy = running_accuracy / len(dataloader)
    epoch_loss = running_loss / len(dataloader)

    return epoch_loss, epoch_accuracy
