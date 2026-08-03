import torch
import os

from torch.utils.data import DataLoader

from datasets.diffusion_mnist import DiffusionMNISTDataset
from diffusion.noise_scheduler import NoiseScheduler
from models.unet import UNet
from trainer.trainer import Trainer
from configs.config import Config
from utils.logger import logger
from utils.checkpoint_manager import CheckpointManager


# Dataset and DataLoader
dataset = DiffusionMNISTDataset(
    "data"
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)

# Model, scheduler, optimizer, trainer
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

# Checkpoint setup
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_manager = CheckpointManager(
    checkpoint_dir=checkpoint_dir,
    model=model,
    optimizer=optimizer,
    scheduler=None  # No PyTorch LR scheduler used
)

EPOCHS = 10
start_epoch = 0
best_loss = float('inf')

# Try to load latest checkpoint to resume training
latest_checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pt')
if os.path.exists(latest_checkpoint_path):
    loaded_epoch, loaded_loss = checkpoint_manager.load_checkpoint(latest_checkpoint_path)
    start_epoch = loaded_epoch + 1  # resume from next epoch
    best_loss = loaded_loss
    logger.info(f"Resuming training from epoch {start_epoch} with best loss {best_loss:.4f}")

# Training loop
for epoch in range(start_epoch, EPOCHS):
    total_loss = 0.0
    for batch in loader:
        loss = trainer.train_step(batch)
        total_loss += loss
    
    avg_loss = total_loss / len(loader)
    logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
    
    # Save checkpoint every epoch
    is_best = avg_loss < best_loss
    if is_best:
        best_loss = avg_loss
    checkpoint_manager.save_checkpoint(epoch, avg_loss, is_best=is_best)