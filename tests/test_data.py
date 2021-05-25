"""Loader smoke tests. Skipped if torchvision can't fetch the dataset."""
import os
import pytest
import torch

from src.data import get_loaders


@pytest.mark.skipif(
    not os.environ.get("RUN_DOWNLOAD_TESTS"),
    reason="needs RUN_DOWNLOAD_TESTS=1 (downloads ~30MB)",
)
def test_loader_shapes():
    train_dl, test_dl = get_loaders(batch_size=8, num_workers=0)
    x, y = next(iter(train_dl))
    assert x.shape == (8, 1, 28, 28)
    assert y.shape == (8,)
    assert x.dtype == torch.float32
    assert (x >= 0).all() and (x <= 1).all()
