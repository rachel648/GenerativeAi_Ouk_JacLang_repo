"""VAE model architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class Encoder(nn.Module):
    """VAE Encoder."""
    
    def __init__(self, image_size: int = 64, channels: int = 3,
                 latent_dim: int = 20, hidden_dims: List[int] = None):
        """
        Args:
            image_size: Input image size
            channels: Number of input channels
            latent_dim: Latent dimension
            hidden_dims: Hidden layer dimensions
        """
        super(Encoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.latent_dim = latent_dim
        
        # Build encoder
        modules = []
        in_channels = channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate flattened dimension
        self.flatten_dim = hidden_dims[-1] * (image_size // (2 ** len(hidden_dims))) ** 2
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return mu, log_var


class Decoder(nn.Module):
    """VAE Decoder."""
    
    def __init__(self, image_size: int = 64, channels: int = 3,
                 latent_dim: int = 20, hidden_dims: List[int] = None):
        """
        Args:
            image_size: Output image size
            channels: Number of output channels
            latent_dim: Latent dimension
            hidden_dims: Hidden layer dimensions
        """
        super(Decoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.latent_dim = latent_dim
        
        # Calculate dimensions
        self.decoder_input_size = image_size // (2 ** len(hidden_dims))
        self.flatten_dim = hidden_dims[-1] * self.decoder_input_size ** 2
        
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        # Build decoder
        modules = []
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        result = self.decoder_input(z)
        result = result.view(-1, 512, self.decoder_input_size, self.decoder_input_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        
        return result


class VAE(nn.Module):
    """Variational Autoencoder."""
    
    def __init__(self, image_size: int = 64, channels: int = 3,
                 latent_dim: int = 20, hidden_dims: List[int] = None):
        """
        Args:
            image_size: Image size
            channels: Number of channels
            latent_dim: Latent dimension
            hidden_dims: Hidden layer dimensions
        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(image_size, channels, latent_dim, hidden_dims)
        self.decoder = Decoder(image_size, channels, latent_dim, hidden_dims)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        
        return recon_x, mu, log_var
    
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples from the model."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor,
                     mu: torch.Tensor, log_var: torch.Tensor,
                     kl_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE loss function."""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss