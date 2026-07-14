from torch.utils.data import DataLoader

from .intel_dataset import IntelDataset
from transforms.transforms import train_transform
from utils.visualization import show_image, show_batch


def test_dataset():
    dataset = IntelDataset(
        "data/raw/seg_train",
        transform=train_transform,
    )

    print(len(dataset))

    image, label = dataset[0]
    print(image.shape)
    print(label)

    label_name = dataset.get_class_name(label)
    show_image(image, label_name)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
    )

    images, labels = next(iter(data_loader))
    show_batch(images, labels,dataset.classes)

    print(images.shape)
    print(images.dtype)
    print(images.min())
    print(images.max())

    distribution = dataset.class_distribution()

    for idx,count in distribution.items():
        print(
            dataset.get_class_name(idx),
            count,
        )
