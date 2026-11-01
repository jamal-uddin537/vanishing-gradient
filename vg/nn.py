"""Minimal NumPy MLP with manual backpropagation.

Every layer implements forward(x) and backward(grad_out) and stores the
gradients of its parameters, so per-layer gradient norms can be inspected
directly. No autograd is used: the chain rule is written out by hand.
"""
import numpy as np


# ---------------------------------------------------------------- activations
class Sigmoid:
    def forward(self, z):
        self.a = 1.0 / (1.0 + np.exp(-z))
        return self.a

    def backward(self, g):
        # sigma'(z) = sigma(z) (1 - sigma(z)) <= 1/4
        return g * self.a * (1.0 - self.a)


class Tanh:
    def forward(self, z):
        self.a = np.tanh(z)
        return self.a

    def backward(self, g):
        # tanh'(z) = 1 - tanh(z)^2 <= 1
        return g * (1.0 - self.a ** 2)


class ReLU:
    def forward(self, z):
        self.mask = z > 0
        return z * self.mask

    def backward(self, g):
        # relu'(z) = 1 for z > 0, else 0
        return g * self.mask


ACTIVATIONS = {"sigmoid": Sigmoid, "tanh": Tanh, "relu": ReLU}


# ---------------------------------------------------------------- init
def init_weight(fan_in, fan_out, scheme, rng):
    if scheme == "naive":      # small fixed std, common textbook default
        std = 0.1
    elif scheme == "xavier":   # Glorot & Bengio (2010): Var = 2/(fan_in+fan_out)
        std = np.sqrt(2.0 / (fan_in + fan_out))
    elif scheme == "he":       # He et al. (2015): Var = 2/fan_in
        std = np.sqrt(2.0 / fan_in)
    else:
        raise ValueError(scheme)
    return rng.normal(0.0, std, size=(fan_in, fan_out))


# ---------------------------------------------------------------- layers
class Linear:
    def __init__(self, fan_in, fan_out, scheme, rng):
        self.W = init_weight(fan_in, fan_out, scheme, rng)
        self.b = np.zeros(fan_out)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        # g = dL/dz_l (the back-propagated error signal "delta" at this layer)
        self.delta_norm = np.linalg.norm(g) / np.sqrt(g.shape[0])
        self.dW = self.x.T @ g
        self.db = g.sum(axis=0)
        return g @ self.W.T

    def params(self):
        return [(self.W, "dW"), (self.b, "db")]


class BatchNorm:
    """Batch normalisation (Ioffe & Szegedy, 2015), training-mode statistics."""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps, self.momentum = eps, momentum
        self.run_mean, self.run_var = np.zeros(dim), np.ones(dim)
        self.training = True

    def forward(self, x):
        if self.training:
            mu, var = x.mean(0), x.var(0)
            self.run_mean = (1 - self.momentum) * self.run_mean + self.momentum * mu
            self.run_var = (1 - self.momentum) * self.run_var + self.momentum * var
        else:
            mu, var = self.run_mean, self.run_var
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu) * self.inv_std
        return self.gamma * self.xhat + self.beta

    def backward(self, g):
        n = g.shape[0]
        self.dgamma = (g * self.xhat).sum(0)
        self.dbeta = g.sum(0)
        dxhat = g * self.gamma
        # standard closed-form BN backward
        return (self.inv_std / n) * (n * dxhat - dxhat.sum(0)
                                     - self.xhat * (dxhat * self.xhat).sum(0))

    def params(self):
        return [(self.gamma, "dgamma"), (self.beta, "dbeta")]


class Block:
    """Hidden block: Linear -> [BN] -> activation, optionally with a skip
    connection  h_{l+1} = h_l + f(h_l)  (He et al., 2016)."""

    def __init__(self, dim, act, scheme, rng, batchnorm=False, residual=False):
        self.lin = Linear(dim, dim, scheme, rng)
        self.bn = BatchNorm(dim) if batchnorm else None
        self.act = ACTIVATIONS[act]()
        self.residual = residual

    def forward(self, x):
        z = self.lin.forward(x)
        if self.bn:
            z = self.bn.forward(z)
        a = self.act.forward(z)
        return x + a if self.residual else a

    def backward(self, g):
        d = self.act.backward(g)
        if self.bn:
            d = self.bn.backward(d)
        d = self.lin.backward(d)
        return g + d if self.residual else d

    def layers(self):
        return [self.lin] + ([self.bn] if self.bn else [])


# ---------------------------------------------------------------- network
class MLP:
    def __init__(self, n_in, n_out, width=64, depth=20, act="sigmoid",
                 init="xavier", batchnorm=False, residual=False, seed=0):
        rng = np.random.default_rng(seed)
        self.inp = Linear(n_in, width, init, rng)
        self.inp_act = ACTIVATIONS[act]()
        self.blocks = [Block(width, act, init, rng, batchnorm, residual)
                       for _ in range(depth - 1)]
        self.out = Linear(width, n_out, init, rng)

    def train(self, flag=True):
        for b in self.blocks:
            if b.bn:
                b.bn.training = flag

    def forward(self, x):
        h = self.inp_act.forward(self.inp.forward(x))
        for b in self.blocks:
            h = b.forward(h)
        return self.out.forward(h)

    def backward(self, g):
        g = self.out.backward(g)
        for b in reversed(self.blocks):
            g = b.backward(g)
        self.inp.backward(self.inp_act.backward(g))

    def linear_layers(self):
        """Weight layers ordered from input (layer 1) to output (layer L)."""
        return [self.inp] + [b.lin for b in self.blocks] + [self.out]

    def all_layers(self):
        ls = [self.inp]
        for b in self.blocks:
            ls += b.layers()
        return ls + [self.out]

    def grad_norms(self):
        return np.array([np.linalg.norm(l.dW) for l in self.linear_layers()])

    def delta_norms(self):
        """RMS-per-sample norm of dL/dz_l for every weight layer."""
        return np.array([l.delta_norm for l in self.linear_layers()])


# ---------------------------------------------------------------- loss / optim
def softmax_cross_entropy(logits, y):
    # numerically stable log-softmax (no clamping, so the loss stays smooth)
    z = logits - logits.max(1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(1, keepdims=True))
    p = np.exp(log_p)
    n = len(y)
    loss = -log_p[np.arange(n), y].mean()
    g = p.copy()
    g[np.arange(n), y] -= 1.0
    return loss, g / n


class SGD:
    def __init__(self, model, lr=0.1, momentum=0.9):
        self.model, self.lr, self.mom = model, lr, momentum
        self.v = {}

    def step(self):
        for layer in self.model.all_layers():
            for p, gname in layer.params():
                key = id(p)
                v = self.v.get(key, np.zeros_like(p))
                v = self.mom * v - self.lr * getattr(layer, gname)
                self.v[key] = v
                p += v
