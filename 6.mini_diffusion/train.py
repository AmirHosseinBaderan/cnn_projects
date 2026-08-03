from torch.utils.data import DataLoader

from configs.config import Config
from datasets.mnist import MNISTDataset


dataset = MNISTDataset(
    root=Config.DATA_DIR,
)

loader = DataLoader(
    dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
)

images, labels = next(iter(loader))

print(images.shape)
print(labels.shape)
print(labels)