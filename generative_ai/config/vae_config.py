"""VAE-specific configuration."""

from .base_config import BaseConfig


class VAEConfig(BaseConfig):
    """Configuration for VAE training."""
    
    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)
        
        # VAE-specific parameters
        self.latent_dim = getattr(self, 'latent_dim', 20)
        self.image_size = getattr(self, 'image_size', 64)
        self.channels = getattr(self, 'channels', 3)
        self.hidden_dims = getattr(self, 'hidden_dims', [32, 64, 128, 256, 512])
        self.kl_weight = getattr(self, 'kl_weight', 1.0)
        self.reconstruction_loss = getattr(self, 'reconstruction_loss', 'mse')
        
        # Annealing parameters
        self.kl_annealing = getattr(self, 'kl_annealing', False)
        self.kl_annealing_epochs = getattr(self, 'kl_annealing_epochs', 50)
        
        # Override with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)