import os

import matplotlib.pyplot as plt
import pandas as pd

D = os.getenv("RESULTS_DIR", "results")
df = pd.read_csv(f"{D}/results.csv")
df["sigma"] = df["sigma"].fillna(-1)
agg = (df.groupby(["method", "partition", "sigma"])
         .agg(acc=("accuracy", "mean"), train_acc=("train_acc", "mean"),
              f1=("f1", "mean"), f1_sd=("f1", "std"), eps=("epsilon", "mean"),
              mia=("mia_auc", "mean"), mia_sd=("mia_auc", "std"))
         .reset_index().fillna(0))
agg.to_csv(f"{D}/summary.csv", index=False)
print(agg.round(3).to_string(index=False))

# 1. privacy vs utility
fig, ax = plt.subplots(figsize=(7.5, 4.8))
dp = agg[agg.method == "federated_dp"]
for part, g in dp.groupby("partition"):
    g = g.sort_values("eps")
    ax.errorbar(g.eps, g.f1, yerr=g.f1_sd, marker="o", capsize=3, label=f"Federated + DP ({part})")
for _, r in agg[agg.method != "federated_dp"].iterrows():
    ax.axhline(r.f1, linestyle="--", alpha=0.6, label=f"{r.method} ({r.partition}), no DP")
ax.set_xscale("log")
ax.set_xlabel("epsilon (lower = more private)")
ax.set_ylabel("Test macro F1")
ax.set_title("Privacy-utility trade-off (mean over seeds)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{D}/privacy_vs_f1.png", dpi=150)

# 2. membership inference attack
fig, ax = plt.subplots(figsize=(10, 4.8))
labels = [f"{r.method}\n{r.partition}\n" + ("" if r.sigma < 0 else f"σ={r.sigma}") for r in agg.itertuples()]
ax.bar(range(len(agg)), agg.mia, yerr=agg.mia_sd, capsize=2)
ax.axhline(0.5, color="red", linestyle="--", label="random guessing (0.5)")
ax.set_xticks(range(len(agg)))
ax.set_xticklabels(labels, fontsize=6)
ax.set_ylabel("Attack AUC")
ax.set_title("Membership inference attack success")
ax.legend()
fig.tight_layout()
fig.savefig(f"{D}/attack_auc.png", dpi=150)
print(f"saved plots to {D}/")