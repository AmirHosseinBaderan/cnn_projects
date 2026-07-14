import os
from collections import Counter

from PIL import Image

from torch.utils.data import Dataset


class IntelDataset(Dataset):
    def __init__(
            self,
            root_dir,
            transform=None,
    ):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(
            os.listdir(self.root_dir)
        )

        self.class_to_idx = {
            cls_name: idx
            for idx, cls_name in enumerate(self.classes)
        }

        self.images = []

        for cls_name in self.classes:
            cls_path = os.path.join(
                root_dir,
                cls_name
            )

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(
                    cls_path,
                    img_name
                )

                self.images.append(
                    (
                        img_path,
                        self.class_to_idx[cls_name]
                    )
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx):
        return self.classes[idx]

    def class_distribution(self):
        labels = [label for _, label in self.images]

        return Counter(labels)
