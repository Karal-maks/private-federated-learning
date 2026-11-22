from data import load_adult, partition

Xtr, ytr, Xte, yte = load_adult()
print("train", Xtr.shape, "test", Xte.shape, "positive rate", ytr.mean().round(3))
for name, alpha in [("iid", None), ("non-iid", 0.5)]:
    parts = partition(ytr, 5, alpha)
    print(name, "client sizes", [len(p) for p in parts],
          "positive rates", [round(float(ytr[p].mean()), 2) for p in parts])