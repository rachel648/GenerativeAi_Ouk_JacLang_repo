"""Base configuration class for generative models."""

import yaml
import os
from typing import Dict, Any


class BaseConfig:
    """Base configuration class with common settings."""
    
    def __init__(self, config_path: str = None, **kwargs):
        # Default configuration
        self.batch_size = 32
        self.learning_rate = 0.0002
        self.num_epochs = 100
        self.device = "cuda"
        self.seed = 42
        self.save_interval = 10
        self.log_interval = 100
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"
        self.data_dir = "data"
        
        # Load from config file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # Override with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}