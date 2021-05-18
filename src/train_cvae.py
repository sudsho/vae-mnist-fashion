"""Train conditional VAE - mostly a copy of train.py with the cvae model and label-aware loop."""
import argparse
import os
import torch
import torch.optim as optim

from .data import get_loaders
from .cvae import CVAE
from .loss import vae_loss
from .utils import load_config, set_seed, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cvae.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, _ = get_loaders(**cfg["data"])
    m_cfg = {k: v for k, v in cfg["model"].items() if k != "arch"}
    model = CVAE(**m_cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    out = ensure_dir(cfg["log"]["out_dir"])

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = model(x, y)
            loss, _, _ = vae_loss(x_hat, x, mu, logvar, beta=cfg["train"]["beta"])
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch} loss {total / len(train_dl.dataset):.4f}")
    torch.save(model.state_dict(), os.path.join(out, "last.pt"))


if __name__ == "__main__":
    main()
