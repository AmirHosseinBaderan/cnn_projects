from dataset.detector_dataset import DetectorDataset
from dataset.parsers import IRLPRXMLParser

from preprocessing.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize
)


dataset = DetectorDataset(
    root="data/car_images/train",
    parser=IRLPRXMLParser(),
    transform=Compose(
        [
            Resize(
                width=640,
                height=640,
            ),
            ToTensor(),
            Normalize()
        ]
    ),
)

sample = dataset[0]

print(sample["image"].shape)

print(sample["annotation"].image.width)

plate = sample["annotation"].first("کل ناحیه پلاک")

print(plate.bbox)

print(sample["image"].shape)
print(sample["image"].dtype)
print(sample["image"].min())
print(sample["image"].max())