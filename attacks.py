import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from model import make_model


@torch.no_grad()
def per_sample_loss(state, X, y, d_in):
    m = make_model(d_in)
    m.load_state_dict(state)
    m.eval()
    out = m(torch.from_numpy(X))
    return nn.functional.cross_entropy(out, torch.from_numpy(y).long(), reduction="none").numpy()


def mia_auc(state, X_members, y_members, X_non, y_non, d_in, n=3000, seed=0):
    rng = np.random.RandomState(seed)
    n = min(n, len(y_members), len(y_non))
    i = rng.choice(len(y_members), n, replace=False)
    j = rng.choice(len(y_non), n, replace=False)
    loss_in = per_sample_loss(state, X_members[i], y_members[i], d_in)
    loss_out = per_sample_loss(state, X_non[j], y_non[j], d_in)
    scores = -np.concatenate([loss_in, loss_out])  # low loss => predicted member
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    return float(roc_auc_score(labels, scores))