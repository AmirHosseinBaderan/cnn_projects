import torch

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class DiffusionMNISTDataset(Dataset):

    def __init__(
        self,
        root,
        train=True,
    ):

        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=ToTensor(),
        )


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):

        image, label = self.dataset[index]

        return {
            "image": image,
            "label": torch.tensor(label),
        }