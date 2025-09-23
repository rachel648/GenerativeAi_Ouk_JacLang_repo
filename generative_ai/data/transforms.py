"""Data transformation utilities."""

import torchvision.transforms as transforms
from typing import Optional


def get_image_transforms(image_size: int = 64, normalize: bool = True,
                        augment: bool = False) -> transforms.Compose:
    """
    Get image transformations for training.
    
    Args:
        image_size: Target image size
        normalize: Whether to normalize to [-1, 1]
        augment: Whether to apply data augmentation
    
    Returns:
        Composed transforms
    """
    transform_list = []
    
    # Resize
    transform_list.append(transforms.Resize(image_size))
    transform_list.append(transforms.CenterCrop(image_size))
    
    # Data augmentation
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, 
                                 saturation=0.1, hue=0.1),
        ])
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if normalize:
        transform_list.append(transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ))
    
    return transforms.Compose(transform_list)


def get_text_transforms():
    """Get text transformations (placeholder for future implementation)."""
    # This would include tokenization, padding, etc.
    pass