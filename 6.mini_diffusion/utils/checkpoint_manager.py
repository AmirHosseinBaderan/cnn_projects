import torch
import os

class CheckpointManager:
    def __init__(self, checkpoint_dir, model, optimizer, scheduler=None):
        """
        Args:
            checkpoint_dir: directory to save checkpoints
            model: the model to save
            optimizer: the optimizer to save
            scheduler: the learning rate scheduler (optional)
        """
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, epoch, val_loss, is_best=False, best_val_loss=None):
        """
        Save checkpoint.
        Args:
            epoch: current epoch
            val_loss: current validation loss
            is_best: whether this is the best model so far
            best_val_loss: the best validation loss seen so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss if best_val_loss is not None else val_loss,
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save last checkpoint
        last_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pt')
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint if applicable
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            return 0, 0.0, 0.0

        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu'
        )

        self.model.load_state_dict(
            checkpoint['model_state_dict']
        )

        if (
            self.optimizer is not None
            and 'optimizer_state_dict' in checkpoint
        ):
            self.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )


        if (
            self.scheduler is not None
            and 'scheduler_state_dict' in checkpoint
        ):
            self.scheduler.load_state_dict(
                checkpoint['scheduler_state_dict']
            )

        epoch = checkpoint['epoch']

        val_loss = checkpoint.get(
            'val_loss',
            0.0
        )

        best_val_loss = checkpoint.get(
            'best_val_loss',
            0.0
        )

        return epoch, val_loss, best_val_loss