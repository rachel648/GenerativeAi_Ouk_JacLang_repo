"""VAE trainer implementation."""

import torch
from typing import Dict
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..models.vae import VAE
from ..config.vae_config import VAEConfig
from ..utils.visualization import save_image_grid


class VAETrainer(BaseTrainer):
    """Trainer for VAE models."""
    
    def __init__(self, config: VAEConfig, model: VAE, device: torch.device):
        super().__init__(config, model, device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        # Fixed latent vectors for consistent generation monitoring
        self.fixed_z = torch.randn(64, config.latent_dim, device=device)
        
        # Loss tracking
        self.total_losses = []
        self.recon_losses = []
        self.kl_losses = []
    
    def get_kl_weight(self, epoch: int) -> float:
        """Get KL weight with annealing if enabled."""
        if not self.config.kl_annealing:
            return self.config.kl_weight
        
        # Linear annealing
        if epoch < self.config.kl_annealing_epochs:
            return self.config.kl_weight * (epoch / self.config.kl_annealing_epochs)
        else:
            return self.config.kl_weight
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        kl_weight = self.get_kl_weight(self.current_epoch)
        
        for batch_idx, data in enumerate(tqdm(dataloader, desc=f'Epoch {self.current_epoch}')):
            images = data[0] if isinstance(data, (list, tuple)) else data
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_images, mu, log_var = self.model(images)
            
            # Calculate loss
            total_loss, recon_loss, kl_loss = self.model.loss_function(
                recon_images, images, mu, log_var, kl_weight
            )
            
            # Normalize by batch size
            batch_size = images.size(0)
            total_loss = total_loss / batch_size
            recon_loss = recon_loss / batch_size
            kl_loss = kl_loss / batch_size
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Update losses
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log intermediate results
            if batch_idx % self.config.log_interval == 0:
                self.writer.add_scalar('batch/total_loss', total_loss.item(), self.global_step)
                self.writer.add_scalar('batch/recon_loss', recon_loss.item(), self.global_step)
                self.writer.add_scalar('batch/kl_loss', kl_loss.item(), self.global_step)
                self.writer.add_scalar('batch/kl_weight', kl_weight, self.global_step)
                
                # Generate and save sample images
                if self.global_step % (self.config.log_interval * 5) == 0:
                    self.generate_samples()
        
        # Calculate average losses
        avg_total_loss = epoch_total_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        
        self.total_losses.append(avg_total_loss)
        self.recon_losses.append(avg_recon_loss)
        self.kl_losses.append(avg_kl_loss)
        
        return {
            'loss': avg_total_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
            'kl_weight': kl_weight
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate the model."""
        total_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        num_batches = 0
        
        kl_weight = self.get_kl_weight(self.current_epoch)
        
        with torch.no_grad():
            for data in dataloader:
                images = data[0] if isinstance(data, (list, tuple)) else data
                images = images.to(self.device)
                
                # Forward pass
                recon_images, mu, log_var = self.model(images)
                
                # Calculate loss
                batch_total_loss, batch_recon_loss, batch_kl_loss = self.model.loss_function(
                    recon_images, images, mu, log_var, kl_weight
                )
                
                # Normalize by batch size
                batch_size = images.size(0)
                total_loss += batch_total_loss.item() / batch_size
                recon_loss += batch_recon_loss.item() / batch_size
                kl_loss += batch_kl_loss.item() / batch_size
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'recon_loss': recon_loss / num_batches,
            'kl_loss': kl_loss / num_batches
        }
    
    def generate_samples(self):
        """Generate and save sample images."""
        self.model.eval()
        with torch.no_grad():
            # Generate from fixed latent vectors
            generated_images = self.model.decoder(self.fixed_z)
            
            # Save images
            save_path = f"{self.config.log_dir}/samples_epoch_{self.current_epoch}_step_{self.global_step}.png"
            save_image_grid(generated_images, save_path)
        
        self.model.train()
    
    def reconstruct_samples(self, images: torch.Tensor):
        """Reconstruct and save sample images."""
        self.model.eval()
        with torch.no_grad():
            recon_images, _, _ = self.model(images[:64])
            
            # Save original and reconstructed images side by side
            comparison = torch.cat([images[:64], recon_images], dim=0)
            save_path = f"{self.config.log_dir}/reconstruction_epoch_{self.current_epoch}.png"
            save_image_grid(comparison, save_path, nrow=8)
        
        self.model.train()