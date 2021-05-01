"""tiny helpers."""
import os
import random
import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p
