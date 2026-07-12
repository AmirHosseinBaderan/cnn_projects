from torchvision import datasets
from torchvision import transforms


def get_train_dataset():

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    return dataset

def get_test_dataset():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    return dataset