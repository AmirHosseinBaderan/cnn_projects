from pathlib import Path
import cv2
from torch.utils.data import Dataset
from .transforms import FaceTransform
from .logger import logger


class FaceDataset(Dataset):

    def __init__(self, root):
        self.root = Path(root)

        self.transform = FaceTransform()
        self.samples = []

        for person in self.root.iterdir():
            if person.is_dir():
                for img in person.glob("*"):
                    self.samples.append(img)

        logger.info(f"Images found : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]

        img = cv2.imread(
            str(path),
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img
