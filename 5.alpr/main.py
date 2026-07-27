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
from models.modules.conv import ConvBlock
import torch
from models.modules.backbone import Backbone

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

model = ConvBlock(
    in_channels=3,
    out_channels=32,
)

sample = dataset[0]

image = sample["image"]

print(image.shape)
image = image.unsqueeze(0)

y = model(image)

print(y.shape)

model = Backbone()

x = torch.randn(
    1,
    3,
    640,
    640,
)

y = model(x)

print(y.shape)