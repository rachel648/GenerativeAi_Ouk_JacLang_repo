"""Test model architectures."""

import unittest
import torch
from generative_ai.models.gan import DCGAN, Generator, Discriminator
from generative_ai.models.vae import VAE
from generative_ai.models.transformer import GPTModel


class TestModels(unittest.TestCase):
    """Test model architectures."""
    
    def setUp(self):
        self.device = torch.device('cpu')  # Use CPU for tests
        self.batch_size = 4
        self.image_size = 64
        self.channels = 3
        self.latent_dim = 100
    
    def test_generator(self):
        """Test GAN generator."""
        generator = Generator(
            latent_dim=self.latent_dim,
            image_size=self.image_size,
            channels=self.channels
        )
        
        # Test forward pass
        noise = torch.randn(self.batch_size, self.latent_dim)
        output = generator(noise)
        
        expected_shape = (self.batch_size, self.channels, self.image_size, self.image_size)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.all(output >= -1) and torch.all(output <= 1))  # Tanh output
    
    def test_discriminator(self):
        """Test GAN discriminator."""
        discriminator = Discriminator(
            image_size=self.image_size,
            channels=self.channels
        )
        
        # Test forward pass
        images = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size)
        output = discriminator(images)
        
        expected_shape = (self.batch_size, 1)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))  # Sigmoid output
    
    def test_dcgan(self):
        """Test complete DCGAN model."""
        model = DCGAN(
            latent_dim=self.latent_dim,
            image_size=self.image_size,
            channels=self.channels
        )
        
        # Test noise generation
        noise = model.generate_noise(self.batch_size, self.device)
        self.assertEqual(noise.shape, (self.batch_size, self.latent_dim))
        
        # Test sample generation
        samples = model.generate_samples(self.batch_size, self.device)
        expected_shape = (self.batch_size, self.channels, self.image_size, self.image_size)
        self.assertEqual(samples.shape, expected_shape)
    
    def test_vae(self):
        """Test VAE model."""
        latent_dim = 20
        model = VAE(
            image_size=self.image_size,
            channels=self.channels,
            latent_dim=latent_dim
        )
        
        # Test forward pass
        images = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size)
        recon_images, mu, log_var = model(images)
        
        # Check shapes
        self.assertEqual(recon_images.shape, images.shape)
        self.assertEqual(mu.shape, (self.batch_size, latent_dim))
        self.assertEqual(log_var.shape, (self.batch_size, latent_dim))
        
        # Test generation
        samples = model.generate(self.batch_size, self.device)
        expected_shape = (self.batch_size, self.channels, self.image_size, self.image_size)
        self.assertEqual(samples.shape, expected_shape)
        
        # Test loss function
        total_loss, recon_loss, kl_loss = model.loss_function(
            recon_images, images, mu, log_var
        )
        self.assertIsInstance(total_loss.item(), float)
        self.assertIsInstance(recon_loss.item(), float)
        self.assertIsInstance(kl_loss.item(), float)
    
    def test_gpt_model(self):
        """Test GPT-style transformer model."""
        vocab_size = 1000
        seq_length = 64
        d_model = 256
        
        model = GPTModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=4,
            num_layers=2,
            max_length=seq_length
        )
        
        # Test forward pass
        input_ids = torch.randint(0, vocab_size, (self.batch_size, seq_length))
        logits = model(input_ids)
        
        expected_shape = (self.batch_size, seq_length, vocab_size)
        self.assertEqual(logits.shape, expected_shape)
        
        # Test generation
        generated = model.generate(input_ids[:, :10], max_length=20)
        self.assertEqual(generated.shape[0], self.batch_size)
        self.assertGreaterEqual(generated.shape[1], 10)


if __name__ == '__main__':
    unittest.main()