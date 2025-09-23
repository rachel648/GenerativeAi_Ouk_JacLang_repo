"""Dataset classes for different data types."""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Optional, Callable, List


class ImageDataset(Dataset):
    """Dataset for image data."""
    
    def __init__(self, root_dir: str, transform: Optional[Callable] = None,
                 extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        """
        Args:
            root_dir: Root directory containing images
            transform: Optional transform to be applied on images
            extensions: Tuple of valid image extensions
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extensions = extensions
        
        # Get all image files
        self.image_paths = []
        if os.path.exists(root_dir):
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.lower().endswith(extensions):
                        self.image_paths.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


class TextDataset(Dataset):
    """Dataset for text data."""
    
    def __init__(self, texts: List[str], tokenizer: Callable,
                 max_length: int = 512):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer function
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text, max_length=self.max_length,
                               truncation=True, padding='max_length',
                               return_tensors='pt')
        return tokens


class SyntheticImageDataset(Dataset):
    """Synthetic image dataset for testing."""
    
    def __init__(self, size: int = 1000, image_size: int = 64, 
                 channels: int = 3, transform: Optional[Callable] = None):
        """
        Args:
            size: Number of synthetic images
            image_size: Size of square images
            channels: Number of channels
            transform: Optional transform
        """
        self.size = size
        self.image_size = image_size
        self.channels = channels
        self.transform = transform
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random image
        image = torch.randn(self.channels, self.image_size, self.image_size)
        
        if self.transform:
            image = self.transform(image)
        
        return image