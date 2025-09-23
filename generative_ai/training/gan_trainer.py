"""GAN trainer implementation."""

import torch
import torch.nn as nn
from typing import Dict
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..models.gan import DCGAN
from ..config.gan_config import GANConfig
from ..utils.visualization import save_image_grid


class GANTrainer(BaseTrainer):
    """Trainer for GAN models."""
    
    def __init__(self, config: GANConfig, model: DCGAN, device: torch.device):
        super().__init__(config, model, device)
        
        # Setup optimizers
        self.optimizer_g = torch.optim.Adam(
            model.generator.parameters(),
            lr=config.generator_lr,
            betas=(config.beta1, config.beta2)
        )
        
        self.optimizer_d = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=config.discriminator_lr,
            betas=(config.beta1, config.beta2)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Fixed noise for consistent generation monitoring
        self.fixed_noise = torch.randn(64, config.latent_dim, device=device)
        
        # Loss tracking
        self.g_losses = []
        self.d_losses = []
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(tqdm(dataloader, desc=f'Epoch {self.current_epoch}')):
            real_images = data[0] if isinstance(data, (list, tuple)) else data
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # Create labels
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
            
            # Apply label smoothing
            if self.config.label_smoothing > 0:
                real_labels -= self.config.label_smoothing * torch.rand_like(real_labels)
            
            # Train Discriminator
            for _ in range(self.config.d_steps):
                self.optimizer_d.zero_grad()
                
                # Real images
                real_output = self.model.discriminator(real_images)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # Fake images
                noise = self.model.generate_noise(batch_size, self.device)
                fake_images = self.model.generator(noise).detach()
                fake_output = self.model.discriminator(fake_images)
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer_d.step()
            
            # Train Generator
            for _ in range(self.config.g_steps):
                self.optimizer_g.zero_grad()
                
                noise = self.model.generate_noise(batch_size, self.device)
                fake_images = self.model.generator(noise)
                fake_output = self.model.discriminator(fake_images)
                
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.optimizer_g.step()
            
            # Update losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log intermediate results
            if batch_idx % self.config.log_interval == 0:
                self.writer.add_scalar('batch/g_loss', g_loss.item(), self.global_step)
                self.writer.add_scalar('batch/d_loss', d_loss.item(), self.global_step)
                
                # Generate and save sample images
                if self.global_step % (self.config.log_interval * 5) == 0:
                    self.generate_samples()
        
        # Calculate average losses
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)
        
        return {
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'loss': avg_g_loss + avg_d_loss  # Combined loss for checkpoint saving
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate the model."""
        total_d_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data in dataloader:
                real_images = data[0] if isinstance(data, (list, tuple)) else data
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)
                
                # Real labels
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                
                # Discriminator loss on real images
                real_output = self.model.discriminator(real_images)
                d_loss_real = self.criterion(real_output, real_labels)
                
                # Discriminator loss on fake images
                noise = self.model.generate_noise(batch_size, self.device)
                fake_images = self.model.generator(noise)
                fake_output = self.model.discriminator(fake_images)
                d_loss_fake = self.criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                total_d_loss += d_loss.item()
                num_batches += 1
        
        return {'d_loss': total_d_loss / num_batches}
    
    def generate_samples(self):
        """Generate and save sample images."""
        self.model.eval()
        with torch.no_grad():
            fake_images = self.model.generator(self.fixed_noise)
            
            # Save images
            save_path = f"{self.config.log_dir}/samples_epoch_{self.current_epoch}_step_{self.global_step}.png"
            save_image_grid(fake_images, save_path)
        
        self.model.train()