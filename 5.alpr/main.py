from dataset.detector_dataset import DetectorDataset
from dataset.parsers import IRLPRXMLParser

from preprocessing import (
    Compose,
    Resize,
    ToTensor,
    Normalize
)
from torch.utils.data import DataLoader
from dataset.collate import DetectorCollate


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



loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=DetectorCollate(),
)

batch = next(iter(loader))

print(batch["images"].shape)

print(len(batch["annotations"]))