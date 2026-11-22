import time
from data import load_adult, partition
from fl import run_federated

Xtr, ytr, Xte, yte = load_adult()
parts = partition(ytr, 5, None)

for lr in [0.5, 1.0, 2.0]:
    t0 = time.time()
    _, h, eps = run_federated(Xtr, ytr, Xte, yte, parts, rounds=5,
                              local_epochs=2, lr=lr, dp=True, sigma=1.0)
    r = h[-1]
    print(f"\n>>> lr={lr}: acc={r['accuracy']:.3f} f1={r['f1']:.3f} "
          f"auc={r['auc']:.3f} eps={eps:.2f} time={time.time() - t0:.0f}s\n", flush=True)