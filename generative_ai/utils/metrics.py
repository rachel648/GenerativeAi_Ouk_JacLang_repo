"""Evaluation metrics for generative models."""

import torch
import numpy as np
from typing import Tuple
from scipy import linalg


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """
    Calculate Fréchet Inception Distance (FID) between real and fake features.
    
    Args:
        real_features: Features extracted from real images
        fake_features: Features extracted from generated images
    
    Returns:
        FID score
    """
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # Calculate sqrt of product between covariances
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check for numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid


def calculate_is(predictions: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    """
    Calculate Inception Score (IS).
    
    Args:
        predictions: Softmax outputs from Inception model
        splits: Number of splits for bootstrap estimation
    
    Returns:
        Mean and standard deviation of IS scores
    """
    N = predictions.shape[0]
    
    # Calculate the mean KL divergence
    split_scores = []
    
    for k in range(splits):
        part = predictions[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)


def entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate KL divergence between two probability distributions."""
    return np.sum(p * np.log(p / q))


def lpips_distance(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) distance.
    This is a placeholder - requires the lpips library to be installed.
    """
    # This would require: import lpips
    # loss_fn = lpips.LPIPS(net='alex')
    # return loss_fn(img1, img2).item()
    
    # For now, return MSE as a placeholder
    return torch.nn.functional.mse_loss(img1, img2).item()