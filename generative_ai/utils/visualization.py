"""Visualization utilities."""

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import numpy as np
import os
from typing import List, Tuple


def plot_losses(losses: List[float], title: str = "Training Loss", 
                save_path: str = None):
    """Plot training losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()


def plot_samples(real_samples: torch.Tensor, fake_samples: torch.Tensor,
                nrow: int = 8, save_path: str = None):
    """Plot real and generated samples side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Real samples
    real_grid = vutils.make_grid(real_samples[:64], nrow=nrow, normalize=True)
    ax1.imshow(np.transpose(real_grid.cpu(), (1, 2, 0)))
    ax1.set_title("Real Samples")
    ax1.axis('off')
    
    # Fake samples
    fake_grid = vutils.make_grid(fake_samples[:64], nrow=nrow, normalize=True)
    ax2.imshow(np.transpose(fake_grid.cpu(), (1, 2, 0)))
    ax2.set_title("Generated Samples")
    ax2.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()


def save_image_grid(images: torch.Tensor, save_path: str, nrow: int = 8):
    """Save a grid of images."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid = vutils.make_grid(images, nrow=nrow, normalize=True)
    vutils.save_image(grid, save_path)


def plot_gan_losses(g_losses: List[float], d_losses: List[float], 
                   save_path: str = None):
    """Plot GAN generator and discriminator losses."""
    plt.figure(figsize=(12, 6))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title('GAN Training Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()