from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split


def get_datasets():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = random_split(
        train_dataset,
        [train_size, valid_size]
    )

    return train_dataset, valid_dataset, test_dataset