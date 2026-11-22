import warnings
warnings.filterwarnings("ignore", message=".*Secure RNG.*")
warnings.filterwarnings("ignore", message=".*detected new dataset object.*")
warnings.filterwarnings("ignore", message=".*Full backward hook.*")
import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from model import make_model

DELTA = 1e-5


class Client:
    def __init__(self, X, y, dp=False, sigma=1.0, clip=1.0, batch_size=64):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
        self.n = len(y)
        self.bs, self.dp, self.sigma, self.clip = batch_size, dp, sigma, clip
        # one engine per client, kept across rounds, so privacy accounting accumulates
        self.engine = PrivacyEngine() if dp else None

    def train(self, global_state, d_in, epochs, lr):
        model = make_model(d_in)
        model.load_state_dict(global_state)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        loader = DataLoader(TensorDataset(self.X, self.y), batch_size=self.bs, shuffle=True)
        if self.dp:
            model, opt, loader = self.engine.make_private(
                module=model, optimizer=opt, data_loader=loader,
                noise_multiplier=self.sigma, max_grad_norm=self.clip)
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
        inner = getattr(model, "_module", model)  # unwrap the Opacus wrapper
        return {k: v.detach().clone() for k, v in inner.state_dict().items()}

    def epsilon(self):
        return float(self.engine.get_epsilon(DELTA)) if self.dp else 0.0


def fedavg(states, sizes):
    total = sum(sizes)
    return {k: sum(s[k] * (n / total) for s, n in zip(states, sizes)) for k in states[0]}


@torch.no_grad()
def evaluate(state, X, y, d_in):
    m = make_model(d_in)
    m.load_state_dict(state)
    m.eval()
    logits = m(torch.from_numpy(X))
    pred = logits.argmax(1).numpy()
    prob = torch.softmax(logits, 1)[:, 1].numpy()
    return {"accuracy": float((pred == y).mean()),
            "f1": float(f1_score(y, pred, average="macro")),
            "auc": float(roc_auc_score(y, prob))}


def run_federated(Xtr, ytr, Xte, yte, parts, rounds=15, local_epochs=2, lr=0.1,
                  dp=False, sigma=1.0, clip=1.0, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in = Xtr.shape[1]
    clients = [Client(Xtr[p], ytr[p], dp=dp, sigma=sigma, clip=clip) for p in parts]
    state = make_model(d_in).state_dict()
    history = []
    for r in range(rounds):
        states = [c.train(state, d_in, local_epochs, lr) for c in clients]
        state = fedavg(states, [c.n for c in clients])
        history.append({"round": r + 1, **evaluate(state, Xte, yte, d_in)})
        print(f"  round {r + 1}/{rounds}  acc={history[-1]['accuracy']:.3f}", flush=True)
    eps = max(c.epsilon() for c in clients)
    return state, history, eps