"""ELBO loss with beta-weighted KL term."""
import torch
import torch.nn.functional as F


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    """Negative ELBO. Returns total, recon, kl."""
    x = x.view(x.size(0), -1)
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
    # KL divergence between N(mu, sigma^2) and N(0, 1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon, kl
