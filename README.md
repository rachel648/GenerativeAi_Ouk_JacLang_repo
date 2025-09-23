# Generative AI Training Framework

A comprehensive framework for training various generative AI models including GANs, VAEs, and Transformer-based models.

## Features

- **Multiple Model Architectures**: 
  - GANs (Generative Adversarial Networks) with DCGAN implementation
  - VAEs (Variational Autoencoders) with customizable encoder/decoder
  - Transformer-based models (GPT-style) for text generation

- **Flexible Configuration System**: YAML-based configuration management for all model types

- **Comprehensive Training Infrastructure**: 
  - Base trainer class with common functionality
  - Specialized trainers for each model type
  - Automatic logging and visualization with TensorBoard
  - Model checkpointing and resume functionality

- **Data Handling**: 
  - Support for image and text datasets
  - Data transformations and augmentation
  - Synthetic data generation for testing

- **Utilities**: 
  - Visualization tools for samples and training metrics
  - Evaluation metrics (FID, IS, LPIPS)
  - Helper functions for reproducibility

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rachel648/GenerativeAi_Ouk_Repo.git
cd GenerativeAi_Ouk_Repo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Training a GAN

```python
# Using the example script
python examples/train_gan.py --batch_size 64 --num_epochs 100 --image_size 64

# Or with a config file
python examples/train_gan.py --config config_examples/gan_config.yaml
```

### Training a VAE

```python
# Using the example script
python examples/train_vae.py --batch_size 32 --num_epochs 100 --latent_dim 20

# Or with a config file
python examples/train_vae.py --config config_examples/vae_config.yaml
```

### Custom Training Script

```python
import torch
from generative_ai.models.gan import DCGAN
from generative_ai.training.gan_trainer import GANTrainer
from generative_ai.config.gan_config import GANConfig

# Load configuration
config = GANConfig("config_examples/gan_config.yaml")

# Create model
model = DCGAN(
    latent_dim=config.latent_dim,
    image_size=config.image_size,
    channels=config.channels
)

# Create trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = GANTrainer(config, model, device)

# Train
trainer.train(train_dataloader, val_dataloader)
```

## Project Structure

```
GenerativeAi_Ouk_Repo/
├── generative_ai/                 # Main package
│   ├── config/                    # Configuration management
│   │   ├── base_config.py
│   │   ├── gan_config.py
│   │   ├── vae_config.py
│   │   └── transformer_config.py
│   ├── data/                      # Data handling
│   │   ├── datasets.py
│   │   ├── transforms.py
│   │   └── loaders.py
│   ├── models/                    # Model architectures
│   │   ├── gan.py                 # DCGAN implementation
│   │   ├── vae.py                 # VAE implementation
│   │   └── transformer.py        # GPT-style transformer
│   ├── training/                  # Training infrastructure
│   │   ├── base_trainer.py
│   │   ├── gan_trainer.py
│   │   ├── vae_trainer.py
│   │   └── transformer_trainer.py
│   └── utils/                     # Utility functions
│       ├── logging.py
│       ├── visualization.py
│       ├── metrics.py
│       └── helpers.py
├── examples/                      # Example training scripts
│   ├── train_gan.py
│   ├── train_vae.py
│   └── train_transformer.py
├── config_examples/               # Example configurations
│   ├── gan_config.yaml
│   ├── vae_config.yaml
│   └── transformer_config.yaml
├── notebooks/                     # Jupyter notebooks (coming soon)
├── tests/                         # Unit tests (coming soon)
├── requirements.txt
├── setup.py
└── README.md
```

## Model Details

### GAN (Generative Adversarial Network)
- **Architecture**: Deep Convolutional GAN (DCGAN)
- **Features**: Configurable generator/discriminator, label smoothing, feature matching
- **Use Case**: High-quality image generation

### VAE (Variational Autoencoder)
- **Architecture**: Convolutional encoder-decoder with reparameterization trick
- **Features**: KL annealing, customizable hidden dimensions, reconstruction loss options
- **Use Case**: Image generation with latent space interpolation

### Transformer
- **Architecture**: GPT-style decoder-only transformer
- **Features**: Multi-head attention, positional encoding, top-k sampling
- **Use Case**: Text generation and sequence modeling

## Configuration

All models use YAML configuration files for easy experimentation. Key parameters include:

- **Common**: batch_size, learning_rate, num_epochs, device, seed
- **GAN-specific**: latent_dim, generator_lr, discriminator_lr, label_smoothing
- **VAE-specific**: latent_dim, kl_weight, hidden_dims, kl_annealing
- **Transformer-specific**: vocab_size, d_model, num_heads, num_layers

## Monitoring and Visualization

The framework provides comprehensive monitoring through:
- **TensorBoard**: Real-time loss tracking and sample visualization
- **Logging**: Detailed training logs with configurable verbosity
- **Checkpointing**: Automatic model saving and resume functionality

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Research papers that inspired the model implementations
- Open source community for tools and libraries used in this project
