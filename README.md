# vae-mnist-fashion

Variational autoencoder on Fashion-MNIST. The plan is to train a vanilla VAE and a Beta-VAE,
then poke around the latent space (t-SNE on means, 2D grid traversal, interpolation between
two test images).

## Dataset
Fashion-MNIST via `torchvision.datasets.FashionMNIST`. 60k train / 10k test, 28x28 grayscale,
10 clothing classes (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Boot).

## todo
- [ ] vanilla VAE (MLP encoder/decoder)
- [ ] beta-VAE comparison
- [ ] latent space viz (t-SNE on mu)
- [ ] interpolation demo
- [ ] streamlit explorer with 2D latent slider
