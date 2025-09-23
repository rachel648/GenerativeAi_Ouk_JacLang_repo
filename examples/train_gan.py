#!/usr/bin/env python3
"""
Example script for training a GAN model.
"""

import torch
import argparse
from torch.utils.data import DataLoader

from generative_ai.models.gan import DCGAN
from generative_ai.training.gan_trainer import GANTrainer
from generative_ai.config.gan_config import GANConfig
from generative_ai.data.datasets import SyntheticImageDataset
from generative_ai.data.transforms import get_image_transforms
from generative_ai.utils.helpers import set_seed


def main():
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data/images', 
                       help='Path to image data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    if args.config:
        config = GANConfig(args.config)
    else:
        config = GANConfig(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            latent_dim=args.latent_dim,
            image_size=args.image_size,
            generator_lr=args.lr,
            discriminator_lr=args.lr,
            device=str(device),
            seed=args.seed
        )
    
    # Set random seed
    set_seed(config.seed)
    
    # Create dataset and dataloader
    # For demonstration, using synthetic data
    # Replace with actual ImageDataset for real data
    transform = get_image_transforms(config.image_size, normalize=True, augment=True)
    dataset = SyntheticImageDataset(
        size=5000,
        image_size=config.image_size,
        channels=config.channels,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Create model
    model = DCGAN(
        latent_dim=config.latent_dim,
        image_size=config.image_size,
        channels=config.channels,
        g_features=config.generator_features,
        d_features=config.discriminator_features
    )
    
    print(f"Generator parameters: {sum(p.numel() for p in model.generator.parameters())}")
    print(f"Discriminator parameters: {sum(p.numel() for p in model.discriminator.parameters())}")
    
    # Create trainer
    trainer = GANTrainer(config, model, device)
    
    # Start training
    print("Starting training...")
    trainer.train(dataloader)
    
    print("Training completed!")


if __name__ == '__main__':
    main()