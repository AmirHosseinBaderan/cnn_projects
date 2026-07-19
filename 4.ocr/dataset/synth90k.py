from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

class Synth90kDataset(Dataset):

    def __init__(
        self,
        image_dir,
        annotation_file,
        vocabulary,
        transform=None
    ):

        self.image_dir = Path(image_dir)
        self.vocabulary = vocabulary
        self.transform = transform

        self.samples = []

        with open(annotation_file, "r", encoding="utf-8") as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                image_name, label = line.split(maxsplit=1)

                self.samples.append(
                    (
                        image_name,
                        label
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name, label = self.samples[index]
        image_path = self.image_dir / image_name

        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)

        target = self.vocabulary.encode(label)

        return {
            "image": image,
            "label": label,
            "target": target
        }