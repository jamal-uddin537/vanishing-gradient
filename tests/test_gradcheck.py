"""Finite-difference check of the hand-written backward passes.

Compares analytic dL/dW against central differences
    (L(W + eps) - L(W - eps)) / (2 eps)
for every configuration used in the experiments.
Run:  python -m pytest tests  (or)  python tests/test_gradcheck.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vg.nn import MLP, softmax_cross_entropy  # noqa: E402


def relative_error(model, x, y, eps=1e-5, n_checks=6, seed=0):
    rng = np.random.default_rng(seed)
    loss, g = softmax_cross_entropy(model.forward(x), y)
    model.backward(g)
    worst = 0.0
    for layer in model.linear_layers():
        for _ in range(n_checks):
            i, j = rng.integers(layer.W.shape[0]), rng.integers(layer.W.shape[1])
            old = layer.W[i, j]
            layer.W[i, j] = old + eps
            lp, _ = softmax_cross_entropy(model.forward(x), y)
            layer.W[i, j] = old - eps
            lm, _ = softmax_cross_entropy(model.forward(x), y)
            layer.W[i, j] = old
            num = (lp - lm) / (2 * eps)
            ana = layer.dW[i, j]
            denom = max(abs(num) + abs(ana), 1e-10)
            worst = max(worst, abs(num - ana) / denom)
    return worst


def test_all_configs():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(16, 10))
    y = rng.integers(0, 3, size=16)
    for act, bn, res in itertools.product(["sigmoid", "tanh", "relu"],
                                          [False, True], [False, True]):
        init = "he" if act == "relu" else "xavier"
        m = MLP(10, 3, width=8, depth=5, act=act, init=init,
                batchnorm=bn, residual=res, seed=2)
        err = relative_error(m, x, y)
        print(f"{act:8s} bn={bn!s:5s} res={res!s:5s} max rel err = {err:.2e}")
        # ReLU kinks can occasionally give a larger error; 1e-4 is a safe bar
        assert err < 1e-4, (act, bn, res, err)


if __name__ == "__main__":
    test_all_configs()
    print("all gradient checks passed")
