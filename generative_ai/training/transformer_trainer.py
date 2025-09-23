"""Transformer trainer implementation."""

import torch
import torch.nn as nn
from typing import Dict
from tqdm import tqdm
import math

from .base_trainer import BaseTrainer
from ..models.transformer import GPTModel
from ..config.transformer_config import TransformerConfig


class TransformerTrainer(BaseTrainer):
    """Trainer for Transformer-based generative models."""
    
    def __init__(self, config: TransformerConfig, model: GPTModel, device: torch.device):
        super().__init__(config, model, device)
        
        # Setup optimizer with learning rate scheduling
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )
        
        # Loss tracking
        self.losses = []
        self.perplexities = []
    
    def get_lr(self, step: int) -> float:
        """Get learning rate with warmup."""
        if step < self.config.warmup_steps:
            return self.config.max_lr * step / self.config.warmup_steps
        else:
            return self.config.max_lr
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_loss = 0.0
        num_batches = 0
        num_tokens = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {self.current_epoch}')):
            # Extract input_ids and attention_mask from batch
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
            else:
                input_ids = batch.to(self.device)
                attention_mask = None
            
            # Create target (shifted input for autoregressive training)
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            
            # Update learning rate with warmup
            current_lr = self.get_lr(self.global_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            num_tokens += targets.numel()
            self.global_step += 1
            
            # Log intermediate results
            if batch_idx % self.config.log_interval == 0:
                perplexity = math.exp(loss.item())
                self.writer.add_scalar('batch/loss', loss.item(), self.global_step)
                self.writer.add_scalar('batch/perplexity', perplexity, self.global_step)
                self.writer.add_scalar('batch/lr', current_lr, self.global_step)
                
                # Generate sample text
                if self.global_step % (self.config.log_interval * 10) == 0:
                    self.generate_samples()
        
        # Calculate average loss and perplexity
        avg_loss = epoch_loss / num_batches
        avg_perplexity = math.exp(avg_loss)
        
        self.losses.append(avg_loss)
        self.perplexities.append(avg_perplexity)
        
        # Step scheduler
        self.scheduler.step()
        
        return {
            'loss': avg_loss,
            'perplexity': avg_perplexity,
            'tokens_per_second': num_tokens / (num_batches * self.config.batch_size)
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        """Validate the model."""
        total_loss = 0.0
        num_batches = 0
        num_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Extract input_ids from batch
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                else:
                    input_ids = batch.to(self.device)
                    attention_mask = None
                
                # Create target
                targets = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                num_tokens += targets.numel()
        
        avg_loss = total_loss / num_batches
        avg_perplexity = math.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': avg_perplexity
        }
    
    def generate_samples(self, prompt: str = None, max_length: int = 100):
        """Generate text samples."""
        self.model.eval()
        
        with torch.no_grad():
            if prompt is None:
                # Start with random token or special start token
                input_ids = torch.randint(0, self.config.vocab_size, (1, 1), device=self.device)
            else:
                # This would require a tokenizer - placeholder for now
                input_ids = torch.randint(0, self.config.vocab_size, (1, 1), device=self.device)
            
            # Generate sequence
            generated = self.model.generate(
                input_ids, 
                max_length=max_length, 
                temperature=0.8, 
                top_k=50
            )
            
            # Log generated sequence (would need detokenization for actual text)
            self.logger.info(f"Generated sequence: {generated.tolist()}")
            
            # Save to tensorboard as text (placeholder)
            self.writer.add_text(
                'generated_samples', 
                str(generated.tolist()), 
                self.global_step
            )
        
        self.model.train()