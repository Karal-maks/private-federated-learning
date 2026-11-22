import torch.nn as nn


def make_model(d_in):
    # plain Linear layers only: compatible with Opacus (no BatchNorm)
    return nn.Sequential(
        nn.Linear(d_in, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )