"""Shape and forward-pass sanity checks for the VAE."""
import torch

from src.model import VAE, reparameterize


def test_mlp_forward_shapes():
    m = VAE(latent_dim=8)
    x = torch.rand(4, 1, 28, 28)
    x_hat, mu, logvar = m(x)
    assert x_hat.shape == (4, 784)
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)


def test_cnn_forward_shapes():
    m = VAE(latent_dim=8, arch="cnn")
    x = torch.rand(2, 1, 28, 28)
    x_hat, mu, logvar = m(x)
    assert x_hat.shape == (2, 1, 28, 28)
    assert mu.shape == (2, 8)


def test_reparameterize_is_random():
    mu = torch.zeros(8, 4)
    logvar = torch.zeros(8, 4)
    z1 = reparameterize(mu, logvar)
    z2 = reparameterize(mu, logvar)
    assert not torch.allclose(z1, z2)
