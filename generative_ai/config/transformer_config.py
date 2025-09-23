"""Transformer-based generative model configuration."""

from .base_config import BaseConfig


class TransformerConfig(BaseConfig):
    """Configuration for Transformer-based generative models."""
    
    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)
        
        # Transformer-specific parameters
        self.vocab_size = getattr(self, 'vocab_size', 10000)
        self.seq_length = getattr(self, 'seq_length', 512)
        self.d_model = getattr(self, 'd_model', 512)
        self.num_heads = getattr(self, 'num_heads', 8)
        self.num_layers = getattr(self, 'num_layers', 6)
        self.d_ff = getattr(self, 'd_ff', 2048)
        self.dropout = getattr(self, 'dropout', 0.1)
        
        # Training parameters
        self.warmup_steps = getattr(self, 'warmup_steps', 4000)
        self.max_lr = getattr(self, 'max_lr', 0.001)
        self.label_smoothing = getattr(self, 'label_smoothing', 0.1)
        
        # Override with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)