"""Training loop for VAE on Fashion-MNIST.

Logs to MLflow if MLFLOW_TRACKING_URI is set, else prints to stdout.
"""
import argparse
import os
import torch
import torch.optim as optim

from .data import get_loaders
from .model import build_vae
from .loss import vae_loss
from .utils import load_config, set_seed, ensure_dir

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


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


@torch.no_grad()
def evaluate(model, loader, beta, device):
    model.eval()
    total_recon = 0.0
    total_kl = 0.0
    for x, _ in loader:
        x = x.to(device)
        x_hat, mu, logvar = model(x)
        _, r, k = vae_loss(x_hat, x, mu, logvar, beta=beta)
        total_recon += r.item()
        total_kl += k.item()
    n = len(loader.dataset)
    return total_recon / n, total_kl / n


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, test_dl = get_loaders(**cfg["data"])
    model = build_vae(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    out = ensure_dir(cfg["log"]["out_dir"])
    if _HAS_MLFLOW and os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_experiment("vae-fmnist")
        mlflow.start_run()
        mlflow.log_params({**cfg["model"], **cfg["train"]})
    for epoch in range(cfg["train"]["epochs"]):
        avg = train_one_epoch(model, train_dl, opt, cfg["train"]["beta"], device)
        r, k = evaluate(model, test_dl, cfg["train"]["beta"], device)
        print(f"epoch {epoch} train_loss {avg:.4f} test_recon {r:.4f} test_kl {k:.4f}")
        if _HAS_MLFLOW and os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.log_metrics({"train_loss": avg, "test_recon": r, "test_kl": k}, step=epoch)
        if (epoch + 1) % cfg["log"]["save_every"] == 0:
            torch.save(model.state_dict(), os.path.join(out, f"epoch_{epoch}.pt"))
    torch.save(model.state_dict(), os.path.join(out, "last.pt"))
    if _HAS_MLFLOW and os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.end_run()


if __name__ == "__main__":
    main()
