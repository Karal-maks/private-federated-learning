import os
import time

import numpy as np
import pandas as pd

from attacks import mia_auc
from data import load_adult, partition
from fl import evaluate, run_federated

try:
    import mlflow
except ImportError:
    mlflow = None

QUICK = os.getenv("QUICK") == "1"
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
N_TRAIN = int(os.getenv("N_TRAIN", "10000"))
ROUNDS = int(os.getenv("ROUNDS", "5" if QUICK else "15"))
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "2"))
LR = 1.0                                   # same for every configuration
N_CLIENTS = 5
SIGMAS = [1.0, 3.0] if QUICK else [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
SEEDS = [0] if QUICK else [0, 1, 2]


def log_mlflow(name, params, metrics):
    if mlflow:
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if mlflow:
        mlflow.set_experiment(RESULTS_DIR)
    Xtr, ytr, Xte, yte = load_adult(n_train=N_TRAIN)
    d_in = Xtr.shape[1]
    print(f"settings: n_train={len(ytr)} rounds={ROUNDS} local_epochs={LOCAL_EPOCHS} "
          f"seeds={SEEDS} sigmas={SIGMAS} -> {RESULTS_DIR}/", flush=True)

    splits = {"none": [np.arange(len(ytr))],
              "iid": partition(ytr, N_CLIENTS, None),
              "noniid": partition(ytr, N_CLIENTS, 0.5)}

    experiments = [("centralized", "none", None, False)]
    for pname in ["iid", "noniid"]:
        experiments.append(("federated", pname, None, False))
        for s in SIGMAS:
            experiments.append(("federated_dp", pname, s, True))

    rows, curves = [], []
    total = len(experiments) * len(SEEDS)
    done = 0
    for method, pname, sigma, dp in experiments:
        for seed in SEEDS:
            done += 1
            name = f"{method}|{pname}|sigma={sigma}|seed={seed}"
            print(f"\n=== [{done}/{total}] {name} ===", flush=True)
            t0 = time.time()
            state, hist, eps = run_federated(
                Xtr, ytr, Xte, yte, splits[pname], rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS, lr=LR, dp=dp, sigma=sigma or 1.0, seed=seed)
            mia = mia_auc(state, Xtr, ytr, Xte, yte, d_in, seed=seed)
            train_acc = evaluate(state, Xtr, ytr, d_in)["accuracy"]
            last = hist[-1]
            row = {"run": name, "method": method, "partition": pname, "sigma": sigma,
                   "seed": seed, "epsilon": eps, "accuracy": last["accuracy"],
                   "train_acc": train_acc, "f1": last["f1"], "auc": last["auc"],
                   "mia_auc": mia, "seconds": round(time.time() - t0)}
            rows.append(row)
            curves += [{"run": name, **h} for h in hist]
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(f"{RESULTS_DIR}/results.csv", index=False)
            pd.DataFrame(curves).to_csv(f"{RESULTS_DIR}/history.csv", index=False)
            log_mlflow(name, {"method": method, "partition": pname, "sigma": sigma, "seed": seed},
                       {k: row[k] for k in ["epsilon", "accuracy", "train_acc", "f1", "auc", "mia_auc"]})


if __name__ == "__main__":
    main()