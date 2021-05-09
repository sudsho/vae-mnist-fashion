"""CNN encoder/decoder variant. Easier to plug in vs MLP for 28x28."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)   # 14x14
        self.c2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 7x7
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.t1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.t2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        h = F.relu(self.fc(z)).view(-1, 64, 7, 7)
        h = F.relu(self.t1(h))
        return torch.sigmoid(self.t2(h))
