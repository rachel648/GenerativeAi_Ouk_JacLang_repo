"""Helper utilities for training."""

import torch
import random
import numpy as np
import os
from typing import Dict, Any


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, checkpoint_path: str,
                   additional_info: Dict[str, Any] = None):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer = None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)