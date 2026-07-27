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
from models.modules.head import DetectionHead
from models.detector import Detector
from trainer.target_encoder import TargetEncoder

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

p3, p4, p5 = model(x)

print(p3.shape)
print(p4.shape)
print(p5.shape)

head = DetectionHead(
    in_channels=512,
    num_classes=1,
)

x = torch.randn(
    1,
    512,
    20,
    20,
)

y = head(x)

print(y.shape)

model = Detector()

x = torch.randn(
    2,
    3,
    640,
    640,
)

y = model(x)

print(y.shape)

encoder = TargetEncoder()

target = encoder.encode(sample["annotation"])

print(target["boxes"].shape)
print(target["objectness"].shape)
print(target["classes"].shape)