"""Conditional VAE - condition on class label.

Encoder takes (x, y_onehot) and decoder takes (z, y_onehot).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CondEncoder(nn.Module):
    def __init__(self, n_classes=10, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(784 + n_classes, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y_oh):
        h = F.relu(self.fc1(torch.cat([x, y_oh], dim=1)))
        return self.fc_mu(h), self.fc_logvar(h)


class CondDecoder(nn.Module):
    def __init__(self, n_classes=10, latent_dim=20, hidden_dim=400):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 784)

    def forward(self, z, y_oh):
        h = F.relu(self.fc1(torch.cat([z, y_oh], dim=1)))
        return torch.sigmoid(self.fc2(h))


class CVAE(nn.Module):
    def __init__(self, n_classes=10, latent_dim=20, hidden_dim=400):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.encoder = CondEncoder(n_classes, hidden_dim, latent_dim)
        self.decoder = CondDecoder(n_classes, latent_dim, hidden_dim)

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        y_oh = F.one_hot(y, self.n_classes).float()
        mu, logvar = self.encoder(x, y_oh)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_hat = self.decoder(z, y_oh)
        return x_hat, mu, logvar
