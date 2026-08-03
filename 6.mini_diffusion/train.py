import torch
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets.diffusion_mnist import DiffusionMNISTDataset
from diffusion.noise_scheduler import NoiseScheduler
from models.unet import UNet
from trainer.trainer import Trainer
from configs.config import Config
from utils.logger import logger
from utils.checkpoint_manager import CheckpointManager

# Dataset and DataLoader
full_dataset = DiffusionMNISTDataset("data")

# Split dataset into train and validation (90% train, 10% validation)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
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

# Training parameters
start_epoch = 0
best_val_loss = float('inf')
epochs_without_improvement = 0

# Try to load latest checkpoint to resume training
latest_checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pt')
if os.path.exists(latest_checkpoint_path):
    loaded_epoch, loaded_val_loss, loaded_best_val_loss = checkpoint_manager.load_checkpoint(latest_checkpoint_path)
    start_epoch = loaded_epoch + 1  # resume from next epoch
    best_val_loss = loaded_best_val_loss
    epochs_without_improvement = 0  # Reset counter when resetting
    logger.info(f"Resuming training from epoch {start_epoch} with best val loss {best_val_loss:.4f}")

# Training loop
for epoch in range(start_epoch, Config.EPOCHS):
    # Training phase
    model.train()
    total_train_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in train_loop:
        loss = trainer.train_step(batch)
        total_train_loss += loss
        train_loop.set_postfix(loss=loss)
    
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    total_val_loss = 0.0
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    with torch.no_grad():
        for batch in val_loop:
            loss = trainer.validation_step(batch)
            total_val_loss += loss
            val_loop.set_postfix(loss=loss)
    
    avg_val_loss = total_val_loss / len(val_loader)
    
    logger.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Check if this is the best model so far
    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        logger.info(f"New best validation loss: {best_val_loss:.4f}")
    else:
        epochs_without_improvement += 1
        logger.info(f"No improvement for {epochs_without_improvement} epochs")
    
    # Save checkpoint every epoch
    checkpoint_manager.save_checkpoint(epoch, avg_val_loss, is_best=is_best, best_val_loss=best_val_loss)
    
    # Early stopping check
    if epochs_without_improvement >= Config.PATIENCE:
        logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
        break