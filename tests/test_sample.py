"""Tests for sampling helpers."""
import torch

from src.model import VAE
from src.sample import sample_prior, interpolate
from src.cvae import CVAE
from src.sample import conditional_sample


def test_sample_prior_shape():
    m = VAE(latent_dim=8)
    out = sample_prior(m, n=5)
    assert out.shape == (5, 1, 28, 28)


def test_interpolate_shape():
    m = VAE(latent_dim=8)
    z1 = torch.randn(8)
    z2 = torch.randn(8)
    out = interpolate(m, z1, z2, steps=6)
    assert out.shape == (6, 1, 28, 28)


def test_conditional_sample():
    cv = CVAE(n_classes=10, latent_dim=8)
    out = conditional_sample(cv, label=3, n=4)
    assert out.shape == (4, 1, 28, 28)
