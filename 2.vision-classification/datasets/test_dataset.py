from .intel_dataset import  IntelDataset
from transforms.transforms import train_transform

def test_dataset():
    dataset = IntelDataset(
        "data/raw/seg_train",
        transform=train_transform,
    )

    print(len(dataset))

    image, label = dataset[0]
    print(image.shape)
    print(label)