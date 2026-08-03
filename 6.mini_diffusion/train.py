import torch

from torch.utils.data import DataLoader

from datasets.diffusion_mnist import DiffusionMNISTDataset
from diffusion.noise_scheduler import NoiseScheduler
from models.unet import UNet
from trainer.trainer import Trainer
from configs.config import Config

dataset = DiffusionMNISTDataset(
    "data"
)


loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)


scheduler = NoiseScheduler()

model = UNet().to(Config.DEVICE)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)


trainer = Trainer(
    model,
    scheduler,
    optimizer,
    Config.DEVICE,
)


EPOCHS = 10

for epoch in range(EPOCHS):

    total_loss = 0

    for batch in loader:

        loss = trainer.train_step(batch)

        total_loss += loss


    avg_loss = total_loss / len(loader)

    print(
        epoch,
        avg_loss
    )