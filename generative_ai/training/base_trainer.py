"""Base trainer class."""

import os
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter

from ..utils import setup_logger, set_seed, save_checkpoint


class BaseTrainer(ABC):
    """Base trainer class for all generative models."""
    
    def __init__(self, config: Any, model: torch.nn.Module, 
                 device: torch.device):
        """
        Args:
            config: Training configuration
            model: Model to train
            device: Device to use for training
        """
        self.config = config
        self.model = model
        self.device = device
        
        # Set random seed
        set_seed(config.seed)
        
        # Setup logging
        os.makedirs(config.log_dir, exist_ok=True)
        log_file = os.path.join(config.log_dir, 'training.log')
        self.logger = setup_logger('trainer', log_file)
        
        # Setup tensorboard
        self.writer = SummaryWriter(config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Move model to device
        self.model.to(device)
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    @abstractmethod
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        pass
    
    @abstractmethod
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate the model."""
        pass
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            self.model.train()
            train_metrics = self.train_epoch(train_dataloader)
            
            # Log training metrics
            self.log_metrics(train_metrics, 'train', epoch)
            
            # Validate
            if val_dataloader is not None:
                self.model.eval()
                val_metrics = self.validate(val_dataloader)
                self.log_metrics(val_metrics, 'val', epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, train_metrics.get('loss', 0.0))
        
        self.logger.info("Training completed")
        self.writer.close()
    
    def log_metrics(self, metrics: Dict[str, float], split: str, epoch: int):
        """Log metrics to tensorboard and logger."""
        for key, value in metrics.items():
            self.writer.add_scalar(f'{split}/{key}', value, epoch)
        
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f'Epoch {epoch} ({split}): {metrics_str}')
    
    def save_checkpoint(self, epoch: int, loss: float, 
                       additional_info: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'
        )
        
        save_checkpoint(
            self.model, None, epoch, loss, checkpoint_path, additional_info
        )
        
        self.logger.info(f'Checkpoint saved: {checkpoint_path}')