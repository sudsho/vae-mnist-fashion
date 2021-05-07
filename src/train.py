"""Training loop for VAE on Fashion-MNIST."""
import argparse
import os
import torch
import torch.optim as optim

from .data import get_loaders
from .model import VAE
from .loss import vae_loss
from .utils import load_config, set_seed, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def train_one_epoch(model, loader, opt, beta, device):
    model.train()
    total = 0.0
    for x, _ in loader:
        x = x.to(device)
        opt.zero_grad()
        x_hat, mu, logvar = model(x)
        loss, _, _ = vae_loss(x_hat, x, mu, logvar, beta=beta)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader.dataset)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, _ = get_loaders(**cfg["data"])
    model = VAE(**cfg["model"]).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    out = ensure_dir(cfg["log"]["out_dir"])
    for epoch in range(cfg["train"]["epochs"]):
        avg = train_one_epoch(model, train_dl, opt, cfg["train"]["beta"], device)
        print(f"epoch {epoch} loss {avg:.4f}")
        if (epoch + 1) % cfg["log"]["save_every"] == 0:
            torch.save(model.state_dict(), os.path.join(out, f"epoch_{epoch}.pt"))
    torch.save(model.state_dict(), os.path.join(out, "last.pt"))


if __name__ == "__main__":
    main()
