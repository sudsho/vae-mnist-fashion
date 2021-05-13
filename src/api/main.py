"""FastAPI service.

Endpoints:
  /generate        sample from prior
  /interpolate     interp between two random latents
  /reconstruct     reconstruct an image (passed as base64 png)
"""
import io
import base64
import os
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from ..model import VAE


app = FastAPI(title="vae-mnist-fashion")
_MODEL = None


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    ckpt = os.environ.get("VAE_CKPT", "runs/default/last.pt")
    m = VAE()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    _MODEL = m
    return m


def _to_b64(t):
    arr = (t.numpy() * 255).astype("uint8").reshape(28, 28)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class GenerateReq(BaseModel):
    n: int = 1


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate")
def generate(req: GenerateReq):
    m = _load()
    z = torch.randn(req.n, m.latent_dim)
    with torch.no_grad():
        x = m.decoder(z).view(-1, 28, 28).cpu()
    return {"images": [_to_b64(x[i]) for i in range(req.n)]}
