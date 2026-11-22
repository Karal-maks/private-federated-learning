import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]


def _read_adult():
    src = "adult.data" if os.path.exists("adult.data") else UCI_URL
    return pd.read_csv(src, header=None, names=COLS, na_values="?", skipinitialspace=True)


def load_adult(n_train=10000, n_test=5000):
    df = _read_adult().dropna(subset=["income"]).reset_index(drop=True)
    y = df["income"].astype(str).str.contains(">50K").astype(int).values
    X = df.drop(columns=["income", "fnlwgt"])  # fnlwgt is a census sampling weight, not a feature
    num = list(X.select_dtypes("number").columns)
    cat = [c for c in X.columns if c not in num]

    idx = np.random.RandomState(SEED).permutation(len(X))
    tr, te = idx[:n_train], idx[n_train:n_train + n_test]

    pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")),
                        ("s", StandardScaler())]), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    Xtr = pre.fit_transform(X.iloc[tr]).astype(np.float32)
    Xte = pre.transform(X.iloc[te]).astype(np.float32)
    return Xtr, y[tr], Xte, y[te]


def partition(y, n_clients, alpha=None, seed=SEED):
    """Split row indices across clients. alpha=None gives an even random split;
    a small alpha (e.g. 0.5) gives a skewed (non-IID) label split."""
    rng = np.random.RandomState(seed)
    if alpha is None:
        return np.array_split(rng.permutation(len(y)), n_clients)
    for _ in range(50):  # retry until no client is tiny
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet([alpha] * n_clients)
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx_c, cuts)):
                parts[i] += chunk.tolist()
        if min(len(p) for p in parts) >= 100:
            return [np.array(p) for p in parts]
    raise RuntimeError("Could not make a balanced non-IID split; try a larger alpha.")