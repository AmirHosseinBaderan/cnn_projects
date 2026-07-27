# 5.alpr Project Documentation

## Project Description

5.alpr is an Automatic License Plate Recognition (ALPR) system built with PyTorch. The project implements a deep learning pipeline for detecting and recognizing license plates from images. The architecture follows a modular design with separate components for data loading, preprocessing, model definition (backbone, detection head), training, and inference.

### Key Features:
- **Detection Module**: Detects license plate regions in images using a CNN-based detector with backbone and head modules
- **Dataset Handling**: Supports VOC-style XML annotations through custom parsers
- **Preprocessing Pipeline**: Composable transforms including resize, normalization, and tensor conversion
- **Training Infrastructure**: Target encoding, loss computation, checkpointing, and validation
- **Inference Engine**: Ready for plate detection and recognition inference
- **Domain Objects**: Clean data structures for annotations, bounding boxes, and image metadata

### Project Structure:
- `main.py` - Entry point with test scripts for model components
- `config.py` - Device configuration (CUDA/CPU)
- `models/` - Neural network architectures (detector, recognizer, backbone, head, modules)
- `dataset/` - Dataset loading, parsing, and collation
- `preprocessing/` - Image transformation pipeline
- `domain/` - Data classes and domain objects
- `trainer/` - Training loop, loss functions, target encoding, validation
- `inference/` - Prediction and preprocessing for inference

---

## Source Code

### File: main.py
```python
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

target = encoder.encode(
    sample["annotation"],
)

print(target.objectness.sum())

print(torch.nonzero(target.objectness))

print(target.boxes[target.objectness == 1])
```

### File: train.py
```python

```

### File: predict.py
```python

```

### File: config.py
```python
import torch

class Config:
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
```

### File: dataset/collate.py
```python
import torch


class DetectorCollate:

    def __call__(self, batch):

        images = torch.stack(
            [sample["image"] for sample in batch]
        )

        annotations = [
            sample["annotation"]
            for sample in batch
        ]

        return {
            "images": images,
            "annotations": annotations,
        }
```

### File: dataset/vocabulary.py
```python

```

### File: dataset/detector_dataset.py
```python
from pathlib import Path

import cv2
from torch.utils.data import Dataset

from dataset.parsers import AnnotationParser
from domain import Annotation


class DetectorDataset(Dataset):

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        root: str | Path,
        parser: AnnotationParser,
        transform=None,
    ):
        self.root = Path(root)
        self.parser = parser
        self.transform = transform

        self.samples = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):
        image_path, annotation_path = self.samples[index]

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation = self.parser.parse(annotation_path)

        if self.transform is not None:
            image, annotation = self.transform(
                image,
                annotation,
            )

        return {
            "image": image,
            "annotation": annotation,
        }

    def _load_samples(
        self,
    ) -> list[tuple[Path, Path]]:

        samples = []

        for image_path in sorted(self.root.iterdir()):

            if image_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue

            annotation_path = image_path.with_suffix(".xml")

            if not annotation_path.exists():
                continue

            samples.append(
                (
                    image_path,
                    annotation_path,
                )
            )

        return samples
```

### File: dataset/recognizer_dataset.py
```python

```

### File: dataset/parsers/ir_lpr_xml_parser.py
```python
from pathlib import Path
import xml.etree.ElementTree as ET

from domain import (
    Annotation,
    BoundingBox,
    ImageInfo,
    LabeledObject,
)

from dataset.parsers.base import AnnotationParser


class IRLPRXMLParser(AnnotationParser):

    def parse(
        self,
        annotation_path: str | Path,
    ) -> Annotation:

        root = ET.parse(Path(annotation_path)).getroot()

        annotation = Annotation(
            image=self._parse_image(root),
        )

        for object_element in root.findall("object"):
            annotation.add_object(
                self._parse_object(object_element),
            )

        return annotation

    def _parse_image(
        self,
        root: ET.Element,
    ) -> ImageInfo:

        size = root.find("size")

        return ImageInfo(
            filename=self._text(root, "filename"),
            folder=self._text(root, "folder"),
            width=self._optional_int(size, "width"),
            height=self._optional_int(size, "height"),
            depth=self._optional_int(size, "depth"),
        )

    def _parse_object(
        self,
        object_element: ET.Element,
    ) -> LabeledObject:

        bbox = object_element.find("bndbox")

        if bbox is None:
            raise ValueError("Missing <bndbox> element.")

        return LabeledObject(
            label=self._text(object_element, "name"),
            bbox=BoundingBox(
                xmin=self._number(bbox, "xmin"),
                ymin=self._number(bbox, "ymin"),
                xmax=self._number(bbox, "xmax"),
                ymax=self._number(bbox, "ymax"),
            ),
        )

    @staticmethod
    def _text(
        element: ET.Element,
        tag: str,
    ) -> str | None:

        value = element.findtext(tag)

        if value is None:
            return None

        return value.strip()

    @staticmethod
    def _number(
        element,
        tag,
    ) -> float:
        value = element.findtext(tag)

        if value is None:
            raise ValueError(f"{tag} not found")

        return float(value)

    @staticmethod
    def _optional_int(
        element: ET.Element | None,
        tag: str,
    ) -> int | None:

        if element is None:
            return None

        value = element.findtext(tag)

        if value is None:
            return None

        return int(value)
```

### File: dataset/parsers/__init__.py
```python
from .base import AnnotationParser
from .ir_lpr_xml_parser import IRLPRXMLParser

__all__ = [
    "AnnotationParser",
    "IRLPRXMLParser",
]
```

### File: dataset/parsers/base.py
```python
from abc import ABC, abstractmethod
from pathlib import Path

from domain.annotation import Annotation


class AnnotationParser(ABC):

    @abstractmethod
    def parse(
        self,
        annotation_path: str |Path,
    ) -> Annotation:
        raise NotImplementedError
```

### File: preprocessing/transforms.py
```python
from abc import ABC, abstractmethod

import cv2
import torch

from domain import (
    Annotation,
    ImageInfo,
)


class Transform(ABC):

    @abstractmethod
    def __call__(
        self,
        image,
        annotation: Annotation,
    ):
        raise NotImplementedError
```

### File: preprocessing/__init__.py
```python
from .transforms import Transform
from .compose import Compose
from .normalize import Normalize
from .resize import Resize
from .to_tensor import ToTensor

__all__ = [
    "Transform",
    "Compose",
    "Normalize",
    "Resize",
    "ToTensor"
]
```

### File: preprocessing/resize.py
```python
from .transforms import Transform
from domain import Annotation,ImageInfo
import cv2

class Resize(Transform):

    def __init__(
        self,
        width: int,
        height: int,
    ):
        self.width = width
        self.height = height

    def __call__(
        self,
        image,
        annotation: Annotation,
    ):
        original_height, original_width = image.shape[:2]

        scale_x = self.width / original_width
        scale_y = self.height / original_height

        resized_image = cv2.resize(
            image,
            (self.width, self.height),
        )

        resized_annotation = Annotation(
            image=ImageInfo(
                filename=annotation.image.filename,
                folder=annotation.image.folder,
                width=self.width,
                height=self.height,
                depth=annotation.image.depth,
            )
        )

        for obj in annotation.objects:

            resized_annotation.add_object(
                obj.with_bbox(
                    obj.bbox.scale(
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )
                )
            )

        return resized_image, resized_annotation
```

### File: preprocessing/compose.py
```python
from .transforms import Transform
from domain import Annotation

class Compose(Transform):

    def __init__(
        self,
        transforms,
    ):
        self.transforms = transforms

    def __call__(
        self,
        image,
        annotation: Annotation,
    ):
        for transform in self.transforms:
            image, annotation = transform(
                image,
                annotation,
            )

        return image, annotation
```

### File: preprocessing/normalize.py
```python
from .transforms import Transform

class Normalize(Transform):

    def __init__(
        self,
        mean=None,
        std=None,
    ):
        self.mean = mean
        self.std = std

    def __call__(
        self,
        image,
        annotation,
    ):
        image /= 255.0

        if self.mean is not None and self.std is not None:
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        return image, annotation
```

### File: preprocessing/to_tensor.py
```python
from .transforms import Transform
import torch

class ToTensor(Transform):
    def __call__(self, image, annotation):
        image = torch.from_numpy(image)
        image = image.permute(
            2,0,1
        ).float()
         
        return image,annotation
```

### File: domain/__init__.py
```python
from .annotation import Annotation
from .bounding_box import BoundingBox
from .image_info import ImageInfo
from .labeled_object import LabeledObject

__all__ = [
    "Annotation",
    "BoundingBox",
    "ImageInfo",
    "LabeledObject",
]
```

### File: domain/image_info.py
```python
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ImageInfo:
    filename: str

    folder: str | None = None

    width: float | None = None

    height: float | None = None

    depth: float | None = None
```

### File: domain/labeled_object.py
```python
from dataclasses import dataclass

from domain.bounding_box import BoundingBox


@dataclass(slots=True, frozen=True)
class LabeledObject:
    label: str
    bbox: BoundingBox

    def with_bbox(
        self,
        bbox: BoundingBox,
    ) -> "LabeledObject":
        return LabeledObject(
            label=self.label,
            bbox=bbox,
        )
```

### File: domain/labels.py
```python
class Labels:
    PLATE = "کل ناحیه پلاک"
```

### File: domain/annotation.py
```python
from dataclasses import dataclass, field

from domain.image_info import ImageInfo
from domain.labeled_object import LabeledObject
from domain.labels import Labels


@dataclass(slots=True)
class Annotation:
    image: ImageInfo
    objects: list[LabeledObject] = field(default_factory=list)

    def add_object(
        self,
        obj: LabeledObject,
    ) -> None:
        self.objects.append(obj)

    def has_label(
        self,
        label: str,
    ) -> bool:
        return any(
            obj.label == label
            for obj in self.objects
        )

    def find_all(
        self,
        label: str,
    ) -> list[LabeledObject]:
        return [
            obj
            for obj in self.objects
            if obj.label == label
        ]

    def exclude_label(
        self,
        label: str,
    ) -> list[LabeledObject]:
        return [
            obj
            for obj in self.objects
            if obj.label != label
        ]

    def first(
        self,
        label: str,
    ) -> LabeledObject | None:

        for obj in self.objects:
            if obj.label == label:
                return obj

        return None

    @property
    def plates(self) -> list[LabeledObject]:
        return self.find_all(Labels.PLATE)

    @property
    def characters(self) -> list[LabeledObject]:
        return self.exclude_label(Labels.PLATE)
     
    def count(
        self,
        label: str,
    ) -> int:
        return len(self.find_all(label))
```

### File: domain/bounding_box.py
```python
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
        )

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.xmin,
            self.ymin,
            self.xmax,
            self.ymax,
        )

    def to_xywh(self) -> tuple[float, float,float,float]:
        return (
            self.xmin,
            self.ymin,
            self.width,
            self.height,
        )

    def scale(
        self,
        scale_x: float,
        scale_y: float,
    ) -> "BoundingBox":
        return BoundingBox(
            xmin=round(self.xmin * scale_x),
            ymin=round(self.ymin * scale_y),
            xmax=round(self.xmax * scale_x),
            ymax=round(self.ymax * scale_y),
        )
```

### File: inference/predictor.py
```python

```

### File: inference/preprocess.py
```python

```

### File: inference/engine.py
```python

```

### File: models/detector.py
```python
import torch
import torch.nn as nn

from models.modules.backbone import Backbone
from models.modules.head import DetectionHead


class Detector(nn.Module):

    def __init__(
        self,
        num_classes: int = 1,
    ):
        super().__init__()

        self.backbone = Backbone()

        self.head = DetectionHead(
            in_channels=512,
            num_classes=num_classes,
        )

    def forward(self, x):

        p3, p4, p5 = self.backbone(x)

        prediction = self.head(p5)

        return prediction
```

### File: models/recognizer.py
```python

```

### File: models/modules/neck.py
```python

```

### File: models/modules/blocks.py
```python

```

### File: models/modules/conv.py
```python
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool = False,
        activation: bool = True,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.activation = (
            nn.SiLU(inplace=True)
            if activation
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x
```

### File: models/modules/head.py
```python
import torch
import torch.nn as nn

from models.modules.conv import ConvBlock


class DetectionHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.prediction_channels = 5 + num_classes

        self.features = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
        )

        self.predictor = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.prediction_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        features = self.features(x)
        prediction = self.predictor(features)

        return prediction
```

### File: models/modules/backbone.py
```python
import torch
import torch.nn as nn

from models.modules.conv import ConvBlock


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.stage1 = ConvBlock(3, 32, stride=2)

        self.stage2 = ConvBlock(32, 64, stride=2)

        self.stage3 = ConvBlock(64, 128, stride=2)

        self.stage4 = ConvBlock(128, 256, stride=2)

        self.stage5 = ConvBlock(256, 512, stride=2)

    def forward(
        self,
        x: torch.Tensor,
    ):

        x = self.stage1(x)
        x = self.stage2(x)

        p3 = self.stage3(x)

        p4 = self.stage4(p3)

        p5 = self.stage5(p4)

        return p3, p4, p5
```

### File: trainer/loss.py
```python
import torch.nn as nn


class DetectionLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.box_loss_fn = nn.SmoothL1Loss()
        self.objectness_loss_fn = nn.BCEWithLogitsLoss()
        self.classification_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        prediction,
        target,
    ):
        box_loss = self.compute_box_loss(
            prediction,
            target,
        )

        objectness_loss = self.compute_objectness_loss(
            prediction,
            target,
        )

        classification_loss = self.compute_classification_loss(
            prediction,
            target,
        )

        total_loss = (
            box_loss
            + objectness_loss
            + classification_loss
        )

        return {
            "loss": total_loss,
            "box_loss": box_loss,
            "objectness_loss": objectness_loss,
            "classification_loss": classification_loss,
        }

    def compute_box_loss(
        self,
        prediction,
        target,
    ):
        raise NotImplementedError

    def compute_objectness_loss(
        self,
        prediction,
        target,
    ):
        raise NotImplementedError

    def compute_classification_loss(
        self,
        prediction,
        target,
    ):
        raise NotImplementedError
```

### File: trainer/detection_target.py
```python
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DetectionTarget:
    boxes: torch.Tensor
    objectness: torch.Tensor
    classes: torch.Tensor
```

### File: trainer/trainer.py
```python

```

### File: trainer/logger.py
```python

```

### File: trainer/checkpoint.py
```python

```

### File: trainer/target_encoder.py
```python
import torch

from trainer.detection_target import DetectionTarget


class TargetEncoder:

    def __init__(
        self,
        image_size: int = 640,
        stride: int = 32,
        num_classes: int = 1,
    ):
        self.image_size = image_size
        self.stride = stride
        self.grid_size = image_size // stride
        self.num_classes = num_classes

    def encode(
        self,
        annotation,
    ) -> DetectionTarget:

        target = DetectionTarget(
            boxes=torch.zeros(
                self.grid_size,
                self.grid_size,
                4,
                dtype=torch.float32,
            ),
            objectness=torch.zeros(
                self.grid_size,
                self.grid_size,
                dtype=torch.float32,
            ),
            classes=torch.zeros(
                self.grid_size,
                self.grid_size,
                dtype=torch.long,
            ),
        )

        for plate in annotation.plates:
            self._encode_plate(
                plate,
                target,
            )

        return target

    def _encode_plate(
        self,
        plate,
        target: DetectionTarget,
    ) -> None:
     
        bbox = plate.bbox
     
        width = bbox.xmax - bbox.xmin
        height = bbox.ymax - bbox.ymin
     
        if width <= 0 or height <= 0:
            return
     
        cx = (bbox.xmin + bbox.xmax) * 0.5
        cy = (bbox.ymin + bbox.ymax) * 0.5
     
        grid_x = int(cx // self.stride)
        grid_y = int(cy // self.stride)
     
        if not (
            0 <= grid_x < self.grid_size
            and
            0 <= grid_y < self.grid_size
        ):
            return
     
        # TODO:
        # Currently each grid cell supports only one object.
        # Later this can be extended to multiple anchors.
     
        cell_x = grid_x * self.stride
        cell_y = grid_y * self.stride
     
        tx = (cx - cell_x) / self.stride
        ty = (cy - cell_y) / self.stride
     
        tw = width / self.image_size
        th = height / self.image_size
     
        target.objectness[grid_y, grid_x] = 1.0
     
        target.boxes[grid_y, grid_x, 0] = tx
        target.boxes[grid_y, grid_x, 1] = ty
        target.boxes[grid_y, grid_x, 2] = tw
        target.boxes[grid_y, grid_x, 3] = th
     
        target.classes[grid_y, grid_x] = 0
```

### File: trainer/validate.py
```python

```

### File: trainer/train_one_epoch.py
```python

```
