"""Vanishing-gradient experiments (NumPy, manual backprop).

Exp 1  per-layer gradient norm at initialisation, deep MLP (depth 30)
Exp 2  gradient norm of the first layer vs network depth
Exp 3  training a depth-20 MLP: loss, test accuracy, gradient ratio

Data: scikit-learn 8x8 digits (1797 images, 10 classes), bundled offline.
Usage:  python experiments.py            (writes results/*.json, figures/*.png)
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from vg.nn import MLP, SGD, softmax_cross_entropy  # noqa: E402

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

SEEDS = [0, 1, 2]
WIDTH = 64

# name -> (activation, init, batchnorm, residual)
CONFIGS = {
    "sigmoid + naive init":   ("sigmoid", "naive", False, False),
    "sigmoid + Xavier":       ("sigmoid", "xavier", False, False),
    "tanh + naive init":      ("tanh", "naive", False, False),
    "tanh + Xavier":          ("tanh", "xavier", False, False),
    "ReLU + naive init":      ("relu", "naive", False, False),
    "ReLU + He":              ("relu", "he", False, False),
    "sigmoid + Xavier + BN":  ("sigmoid", "xavier", True, False),
    "sigmoid + Xavier + residual": ("sigmoid", "xavier", False, True),
}


def load_data(seed=0):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=seed, stratify=y)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, ytr, (Xte - mu) / sd, yte


def build(name, depth, seed):
    act, init, bn, res = CONFIGS[name]
    return MLP(64, 10, width=WIDTH, depth=depth, act=act, init=init,
               batchnorm=bn, residual=res, seed=seed)


def grads_at_init(model, X, y, which="weight"):
    model.train(True)
    loss, g = softmax_cross_entropy(model.forward(X), y)
    model.backward(g)
    return model.grad_norms() if which == "weight" else model.delta_norms()


# ------------------------------------------------------------------ Exp 1
def exp1(Xtr, ytr, depth=30):
    out = {}
    for name in CONFIGS:
        out[name] = {}
        for which in ("delta", "weight"):
            n = np.array([grads_at_init(build(name, depth, s), Xtr[:256], ytr[:256], which)
                          for s in SEEDS])
            out[name][which] = {"mean": n.mean(0).tolist(), "std": n.std(0).tolist()}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    L = np.arange(1, depth + 2)
    for name, r in out.items():
        ax[0].semilogy(L, r["delta"]["mean"], marker=".", label=name)
        ax[1].semilogy(L, r["weight"]["mean"], marker=".", label=name)
    ax[0].set(xlabel="layer index (1 = input side)",
              ylabel=r"RMS $\|\partial L/\partial z_l\|$ (back-propagated error)",
              title="Error signal reaching each layer")
    ax[1].set(xlabel="layer index (1 = input side)",
              ylabel=r"$\|\partial L/\partial W_l\|_F$",
              title="Weight gradient of each layer")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.suptitle(f"Depth-{depth} MLP at initialisation (mean of {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig("figures/exp1_layer_gradients.png", dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ Exp 2
def exp2(Xtr, ytr, depths=(2, 4, 8, 12, 16, 20, 25, 30, 40)):
    names = ["sigmoid + Xavier", "tanh + Xavier", "ReLU + He",
             "sigmoid + Xavier + BN", "sigmoid + Xavier + residual"]
    out = {}
    for name in names:
        vals = []
        for d in depths:
            n = [grads_at_init(build(name, d, s), Xtr[:256], ytr[:256]) for s in SEEDS]
            vals.append(float(np.mean([v[0] / v[-1] for v in n])))
        out[name] = vals
    plt.figure(figsize=(7, 5))
    for name, v in out.items():
        plt.semilogy(depths, v, marker="o", label=name)
    plt.xlabel("network depth (number of weight layers - 1)")
    plt.ylabel(r"$\|\nabla W_1\| \,/\, \|\nabla W_L\|$ at initialisation")
    plt.title("First-layer vs last-layer gradient ratio")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/exp2_ratio_vs_depth.png", dpi=150)
    plt.close()
    return {"depths": list(depths), "ratio": out}


# ------------------------------------------------------------------ Exp 3
def evaluate(model, X, y):
    model.train(False)
    logits = model.forward(X)
    model.train(True)
    loss, _ = softmax_cross_entropy(logits, y)
    return float(loss), float((logits.argmax(1) == y).mean())


def train_one(name, data, depth, seed, epochs=40, lr=0.02, bs=64):
    Xtr, ytr, Xte, yte = data
    model = build(name, depth, seed)
    opt = SGD(model, lr=lr, momentum=0.9)
    rng = np.random.default_rng(seed)
    hist = {"train_loss": [], "test_acc": [], "ratio": [], "first": [], "last": []}
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr))
        ratios, first, last = [], [], []
        # drop the last incomplete batch: a tiny final batch (e.g. 3 samples)
        # gives very noisy BatchNorm statistics that corrupt the running mean/var
        for k in range(0, len(idx) - bs + 1, bs):
            b = idx[k:k + bs]
            loss, g = softmax_cross_entropy(model.forward(Xtr[b]), ytr[b])
            model.backward(g)
            gn = model.grad_norms()
            ratios.append(gn[0] / gn[-1]); first.append(gn[0]); last.append(gn[-1])
            opt.step()
        tl, _ = evaluate(model, Xtr, ytr)
        _, ta = evaluate(model, Xte, yte)
        hist["train_loss"].append(tl); hist["test_acc"].append(ta)
        hist["ratio"].append(float(np.mean(ratios)))
        hist["first"].append(float(np.mean(first))); hist["last"].append(float(np.mean(last)))
    return hist


LRS = (0.003, 0.01, 0.03)


def select_lr(name, data, depth, epochs):
    """Pick the learning rate by mean validation accuracy (val = 20% of train),
    so that every configuration is compared at its own best step size."""
    Xtr, ytr, _, _ = data
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=0, stratify=ytr)
    scores = {}
    for lr in LRS:
        accs = [train_one(name, (Xa, ya, Xv, yv), depth, s, epochs, lr)["test_acc"][-1]
                for s in SEEDS]
        scores[lr] = float(np.nan_to_num(np.mean(accs)))
    return max(scores, key=scores.get), scores


def exp3(data, depth=20, epochs=40):
    out = {}
    for name in CONFIGS:
        lr, scores = select_lr(name, data, depth, epochs)
        runs = [train_one(name, data, depth, s, epochs, lr) for s in SEEDS]
        agg = {k: np.array([r[k] for r in runs]) for k in runs[0]}
        out[name] = {k: {"mean": v.mean(0).tolist(), "std": v.std(0).tolist()}
                     for k, v in agg.items()}
        out[name]["lr"] = lr
        out[name]["val_acc_by_lr"] = {str(k): v for k, v in scores.items()}
        print(f"  {name:30s} lr={lr:<6} final test acc = "
              f"{agg['test_acc'][:, -1].mean():.3f} +/- {agg['test_acc'][:, -1].std():.3f}")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ep = np.arange(1, epochs + 1)
    for name, r in out.items():
        m, s = np.array(r["train_loss"]["mean"]), np.array(r["train_loss"]["std"])
        ax[0].plot(ep, m, label=name)  # no std band: m - s can be <= 0 on a log axis
        m, s = np.array(r["test_acc"]["mean"]), np.array(r["test_acc"]["std"])
        ax[1].plot(ep, m, label=name); ax[1].fill_between(ep, m - s, m + s, alpha=0.15)
        ax[2].semilogy(ep, r["ratio"]["mean"], label=name)
    ax[0].set(xlabel="epoch", ylabel="training loss", title="Training loss", yscale="log")
    ax[1].set(xlabel="epoch", ylabel="test accuracy", title="Test accuracy", ylim=(0, 1))
    ax[2].set(xlabel="epoch", ylabel=r"$\|\nabla W_1\|/\|\nabla W_L\|$",
              title="Gradient ratio during training")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
    ax[2].legend(fontsize=7, loc="best")
    fig.suptitle(f"Depth-{depth} MLP on digits (mean ± std over {len(SEEDS)} seeds)")
    fig.tight_layout()
    fig.savefig("figures/exp3_training.png", dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    t0 = time.time()
    data = load_data()
    Xtr, ytr = data[0], data[1]
    print("Exp 1: gradients at init"); r1 = exp1(Xtr, ytr)
    print("Exp 2: ratio vs depth"); r2 = exp2(Xtr, ytr)
    print("Exp 3: training"); r3 = exp3(data)
    json.dump({"exp1": r1, "exp2": r2, "exp3": r3},
              open("results/results.json", "w"), indent=1)
    # compact summary table
    with open("results/summary.md", "w") as f:
        f.write("| configuration | ‖δ₁‖/‖δ_L‖ at init (depth 30) | ‖∇W₁‖/‖∇W_L‖ at init (depth 30) | "
                "selected lr | final train loss (depth 20) | final test acc (depth 20) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name in CONFIGS:
            m = r1[name]["weight"]["mean"]
            tl = r3[name]["train_loss"]; ta = r3[name]["test_acc"]
            d = r1[name]["delta"]["mean"]
            f.write(f"| {name} | {d[0] / d[-1]:.2e} | {m[0] / m[-1]:.2e} | {r3[name]['lr']} | "
                    f"{tl['mean'][-1]:.3f} ± {tl['std'][-1]:.3f} | "
                    f"{ta['mean'][-1]:.3f} ± {ta['std'][-1]:.3f} |\n")
    print(open("results/summary.md").read())
    print(f"done in {time.time() - t0:.0f}s")
