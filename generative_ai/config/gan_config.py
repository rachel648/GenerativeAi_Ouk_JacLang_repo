"""GAN-specific configuration."""

from .base_config import BaseConfig


class GANConfig(BaseConfig):
    """Configuration for GAN training."""
    
    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)
        
        # GAN-specific parameters
        self.generator_lr = getattr(self, 'generator_lr', 0.0002)
        self.discriminator_lr = getattr(self, 'discriminator_lr', 0.0002)
        self.beta1 = getattr(self, 'beta1', 0.5)
        self.beta2 = getattr(self, 'beta2', 0.999)
        self.latent_dim = getattr(self, 'latent_dim', 100)
        self.image_size = getattr(self, 'image_size', 64)
        self.channels = getattr(self, 'channels', 3)
        self.d_steps = getattr(self, 'd_steps', 1)
        self.g_steps = getattr(self, 'g_steps', 1)
        self.label_smoothing = getattr(self, 'label_smoothing', 0.1)
        self.feature_matching = getattr(self, 'feature_matching', False)
        
        # Architecture parameters
        self.generator_features = getattr(self, 'generator_features', 64)
        self.discriminator_features = getattr(self, 'discriminator_features', 64)
        
        # Override with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)