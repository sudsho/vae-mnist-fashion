"""Sample from prior, interpolate between two latents."""
import argparse
import os
import torch

from .model import VAE
from .utils import ensure_dir


def sample_prior(model, n=64, device="cpu"):
    z = torch.randn(n, model.latent_dim, device=device)
    with torch.no_grad():
        x = model.decoder(z)
    return x.view(-1, 1, 28, 28).cpu()


def interpolate(model, z1, z2, steps=10):
    alphas = torch.linspace(0, 1, steps)
    zs = torch.stack([(1 - a) * z1 + a * z2 for a in alphas])
    with torch.no_grad():
        x = model.decoder(zs)
    return x.view(-1, 1, 28, 28).cpu()


@torch.no_grad()
def conditional_sample(cvae, label, n=8):
    """Sample for a specific class label (CVAE only)."""
    y = torch.full((n,), int(label), dtype=torch.long)
    y_oh = torch.nn.functional.one_hot(y, cvae.n_classes).float()
    z = torch.randn(n, cvae.latent_dim)
    return cvae.decoder(z, y_oh).view(-1, 1, 28, 28).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="samples")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=20)
    p.add_argument("--arch", default="mlp")
    args = p.parse_args()
    ensure_dir(args.out)

    model = VAE(latent_dim=args.latent_dim, arch=args.arch)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    imgs = sample_prior(model, n=args.n)
    torch.save(imgs, os.path.join(args.out, "prior.pt"))


if __name__ == "__main__":
    main()
