import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.image_dataset import get_train_dataset, get_test_dataset
from models.cnn import CNN


def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    # Dataset
    train_dataset = get_train_dataset()

    # data loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
    )

    # Model
    model = CNN().to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # training
    num_epochs = 10

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:
            # move to device
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)

            # loss
            loss = criterion(outputs, labels)

            # clear gradients
            optimizer.zero_grad()

            # backpropagation
            loss.backward()

            # update weights
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss : {epoch_loss:.4f}"
        )


if __name__ == "__main__":
    main()
