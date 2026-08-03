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