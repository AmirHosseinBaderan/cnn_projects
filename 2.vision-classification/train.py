from torch import optim

from datasets.intel_dataset import IntelDataset
from torch.utils.data import DataLoader
from transforms.transforms import train_transform, test_transform
from models.simple_cnn import SimpleCNN
import torch
from utils.logger import logger
import torch.nn as nn


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
        batch_size=64,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()

        running_loss = 0
        running_acc = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            # loss
            loss = criterion(outputs, labels)

            batch_acc = accuracy(outputs, labels)
            running_acc += batch_acc

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_acc = running_acc / len(train_loader)
        epoch_loss = running_loss / len(train_loader)

        logger.info(f"epoch {epoch}, loss: {epoch_loss}, acc: {epoch_acc}")


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()

    total = labels.size(0)
    return correct / total

if __name__ == "__main__":
    train()