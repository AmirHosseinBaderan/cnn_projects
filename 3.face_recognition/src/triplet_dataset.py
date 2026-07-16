import random
from pathlib import Path
import cv2
from torch.utils.data import Dataset
from .transforms import FaceTransform


class TripletDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.transform = FaceTransform()

        self.people = {}

        for person in self.root.iterdir():
            if person.is_dir():
                images = list(person.glob('*'))
                if len(images) >= 2:
                    self.people[person.name] = images

        self.names = list(self.people.keys())

    def __len__(self):
        return len(self.people)

    def __getitem__(self, idx):
        anchor_person = random.choice(self.names)

        anchor_images = self.people[
            anchor_person
        ]

        anchor, positive = random.sample(
            anchor_images,
            2
        )

        negative_person = random.choice(
            [
                x for x in self.names
                if x != anchor_person
            ]
        )

        negative = random.choice(
            self.people[negative_person]
        )
        return (
            self.load(anchor),
            self.load(positive),
            self.load(negative)
        )

    def load(self, path):
        image = cv2.imread(
            str(path),
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self.transform(image)
