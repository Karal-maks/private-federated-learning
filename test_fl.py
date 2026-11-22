from data import load_adult, partition
from fl import run_federated

Xtr, ytr, Xte, yte = load_adult()
parts = partition(ytr, 5, None)

print("Federated, no DP")
_, h, _ = run_federated(Xtr, ytr, Xte, yte, parts, rounds=3)
print(h[-1])

print("Federated + DP (sigma=1.0)")
_, h, eps = run_federated(Xtr, ytr, Xte, yte, parts, rounds=3, dp=True, sigma=1.0)
print(h[-1], "epsilon (worst client):", round(eps, 2))