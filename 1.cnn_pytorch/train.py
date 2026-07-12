from datasets.image_dataset import get_train_dataset
from torch.utils.data import DataLoader

train_dataset = get_train_dataset()

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

for images, labels in train_loader:

    print(images.shape)

    print(labels.shape)

    break