# vae-mnist-fashion

Variational autoencoder on Fashion-MNIST. The plan is to train a vanilla VAE and a Beta-VAE,
then poke around the latent space (t-SNE on means, 2D grid traversal, interpolation between
two test images).

## Dataset
Fashion-MNIST via `torchvision.datasets.FashionMNIST`. 60k train / 10k test, 28x28 grayscale,
10 clothing classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Boot).

## Approach

The standard VAE objective:

    L = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

The first term is the reconstruction term (BCE for binary-ish images works fine on FMNIST).
The second is the KL-divergence between the approximate posterior (the encoder) and the unit
gaussian prior. The Beta-VAE multiplies the KL term by a scalar beta > 1 to encourage more
disentangled latent codes (Higgins et al. 2017).

## Setup

    pip install -r requirements.txt

## Train

    python -m src.train --config configs/default.yaml

## todo
- [ ] vanilla VAE (MLP encoder/decoder)
- [ ] beta-VAE comparison
- [ ] latent space viz (t-SNE on mu)
- [ ] interpolation demo
- [ ] streamlit explorer with 2D latent slider
