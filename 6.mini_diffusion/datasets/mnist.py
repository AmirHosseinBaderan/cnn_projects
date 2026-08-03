from torchvision import datasets
from torchvision.transforms import ToTensor


class MNISTDataset:

    def __init__(
        self,
        root: str,
        train: bool = True,
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

        return image, label