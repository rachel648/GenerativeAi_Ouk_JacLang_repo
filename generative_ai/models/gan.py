"""GAN model architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Generator(nn.Module):
    """DCGAN Generator."""
    
    def __init__(self, latent_dim: int = 100, image_size: int = 64, 
                 channels: int = 3, features: int = 64):
        """
        Args:
            latent_dim: Dimension of latent vector
            image_size: Output image size (must be power of 2)
            channels: Number of output channels
            features: Number of generator features
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels
        
        # Calculate the initial size after first layer
        self.init_size = image_size // 16  # 4 upsampling layers
        
        self.fc = nn.Linear(latent_dim, features * 8 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(features, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = z.size(0)
        
        # Project and reshape
        out = self.fc(z)
        out = out.view(batch_size, -1, self.init_size, self.init_size)
        
        # Generate image
        img = self.conv_blocks(out)
        
        return img


class Discriminator(nn.Module):
    """DCGAN Discriminator."""
    
    def __init__(self, image_size: int = 64, channels: int = 3, features: int = 64):
        """
        Args:
            image_size: Input image size
            channels: Number of input channels
            features: Number of discriminator features
        """
        super(Discriminator, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 1x1
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        validity = self.conv_blocks(img)
        return validity.view(validity.size(0), -1)


class DCGAN(nn.Module):
    """Complete DCGAN model."""
    
    def __init__(self, latent_dim: int = 100, image_size: int = 64,
                 channels: int = 3, g_features: int = 64, d_features: int = 64):
        """
        Args:
            latent_dim: Dimension of latent vector
            image_size: Image size
            channels: Number of channels
            g_features: Generator features
            d_features: Discriminator features
        """
        super(DCGAN, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.generator = Generator(latent_dim, image_size, channels, g_features)
        self.discriminator = Discriminator(image_size, channels, d_features)
    
    def generate_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random noise vector."""
        return torch.randn(batch_size, self.latent_dim, device=device)
    
    def generate_samples(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate fake samples."""
        noise = self.generate_noise(batch_size, device)
        return self.generator(noise)