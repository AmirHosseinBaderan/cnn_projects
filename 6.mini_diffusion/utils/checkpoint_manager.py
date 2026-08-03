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
        
    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        Save checkpoint.
        Args:
            epoch: current epoch
            loss: current loss
            is_best: whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
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
        """
        Load checkpoint.
        Args:
            checkpoint_path: path to checkpoint file
        Returns:
            epoch: the epoch of the loaded checkpoint
        """
        if not os.path.exists(checkpoint_path):
            return 0  # start from epoch 0 if no checkpoint
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint.get('loss', 0.0)
        return epoch, loss