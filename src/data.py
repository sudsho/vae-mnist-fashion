"""Fashion-MNIST loader."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(root="./data", batch_size=128, num_workers=2):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dl = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, test_dl
