"""Data loader utilities."""

from torch.utils.data import DataLoader, Dataset
from typing import Optional


def get_dataloader(dataset: Dataset, batch_size: int = 32, 
                  shuffle: bool = True, num_workers: int = 4,
                  pin_memory: bool = True, drop_last: bool = True) -> DataLoader:
    """
    Create a DataLoader for the given dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop the last incomplete batch
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )