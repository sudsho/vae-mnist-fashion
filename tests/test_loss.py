"""ELBO sanity checks."""
import torch

from src.loss import vae_loss


def test_kl_zero_when_unit_gaussian():
    """KL(N(0,1) || N(0,1)) = 0 within numerical tolerance."""
    x = torch.rand(8, 784)
    x_hat = x.clone()
    mu = torch.zeros(8, 5)
    logvar = torch.zeros(8, 5)
    _, _, kl = vae_loss(x_hat, x, mu, logvar, beta=1.0)
    assert kl.abs().item() < 1e-4


def test_beta_scales_kl():
    x = torch.rand(4, 784)
    x_hat = torch.rand(4, 784)
    mu = torch.randn(4, 5)
    logvar = torch.randn(4, 5)
    t1, _, k1 = vae_loss(x_hat, x, mu, logvar, beta=1.0)
    t4, _, k4 = vae_loss(x_hat, x, mu, logvar, beta=4.0)
    assert torch.allclose(k1, k4, atol=1e-5)
    # total grew by 3 * kl
    assert torch.allclose(t4 - t1, 3 * k1, atol=1e-3)
