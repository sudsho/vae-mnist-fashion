"""t-SNE on encoder means + 2D latent grid traversal."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


CLASSES = ["t-shirt", "trouser", "pullover", "dress", "coat",
           "sandal", "shirt", "sneaker", "bag", "boot"]


@torch.no_grad()
def encode_dataset(model, loader, device="cpu", max_n=2000):
    mus, ys = [], []
    seen = 0
    for x, y in loader:
        x = x.to(device)
        mu, _ = model.encoder(x.view(x.size(0), -1))
        mus.append(mu.cpu().numpy())
        ys.append(y.numpy())
        seen += x.size(0)
        if seen >= max_n:
            break
    return np.concatenate(mus)[:max_n], np.concatenate(ys)[:max_n]


def tsne_plot(mu, y, out_path):
    z2 = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(mu)
    plt.figure(figsize=(7, 7))
    for c in range(10):
        m = y == c
        plt.scatter(z2[m, 0], z2[m, 1], s=4, label=CLASSES[c])
    plt.legend(fontsize=7, markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
