"""Streamlit explorer for the trained VAE.

Tabs:
  Generate  - sample N images from prior
  Interpolate - linear interp between two test images
  Latent Explorer - 2D sliders (works for latent_dim=2 model)
"""
import os
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.model import VAE


@st.cache(allow_output_mutation=True)
def load_model(ckpt_path, latent_dim=20, arch="mlp"):
    m = VAE(latent_dim=latent_dim, arch=arch)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    m.eval()
    return m


def to_pil(t):
    arr = (t.detach().cpu().numpy() * 255).astype(np.uint8).squeeze()
    return Image.fromarray(arr)


st.title("VAE on Fashion-MNIST")
ckpt = st.sidebar.text_input("checkpoint path", "runs/default/last.pt")
ld = st.sidebar.number_input("latent_dim", value=20, step=1)
arch = st.sidebar.selectbox("arch", ["mlp", "cnn"])

if not os.path.exists(ckpt):
    st.warning("checkpoint not found, train a model first.")
    st.stop()

model = load_model(ckpt, latent_dim=int(ld), arch=arch)

tab1, tab2, tab3 = st.tabs(["Generate", "Interpolate", "Latent Explorer"])

with tab1:
    n = st.slider("how many", 1, 32, 8)
    if st.button("sample"):
        z = torch.randn(n, model.latent_dim)
        with torch.no_grad():
            x = model.decoder(z).view(-1, 28, 28)
        cols = st.columns(min(n, 8))
        for i in range(n):
            cols[i % 8].image(to_pil(x[i]), width=80)


with tab2:
    steps = st.slider("steps", 2, 20, 10)
    if st.button("interpolate"):
        z1 = torch.randn(1, model.latent_dim)
        z2 = torch.randn(1, model.latent_dim)
        alphas = torch.linspace(0, 1, steps).view(-1, 1)
        zs = (1 - alphas) * z1 + alphas * z2
        with torch.no_grad():
            x = model.decoder(zs).view(-1, 28, 28)
        cols = st.columns(steps)
        for i in range(steps):
            cols[i].image(to_pil(x[i]), width=64)


with tab3:
    if model.latent_dim != 2:
        st.info("Latent Explorer expects latent_dim=2. Train with configs/latent2d.yaml.")
    else:
        zx = st.slider("z[0]", -3.0, 3.0, 0.0)
        zy = st.slider("z[1]", -3.0, 3.0, 0.0)
        z = torch.tensor([[zx, zy]], dtype=torch.float32)
        with torch.no_grad():
            x = model.decoder(z).view(28, 28)
        st.image(to_pil(x), width=200)
