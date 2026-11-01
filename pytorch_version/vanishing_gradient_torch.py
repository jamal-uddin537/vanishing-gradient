"""Optional PyTorch/MNIST port of the NumPy experiments.

NOTE: this script was NOT executed when the results in the README were
produced (PyTorch was unavailable in that environment). It is provided as a
starting point; please run it yourself and verify before reporting numbers.

    pip install torch torchvision matplotlib
    python pytorch_version/vanishing_gradient_torch.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH, WIDTH, EPOCHS, LR = 20, 128, 5, 0.01

ACTS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}


class Block(nn.Module):
    def __init__(self, act, bn, residual):
        super().__init__()
        self.lin = nn.Linear(WIDTH, WIDTH)
        self.bn = nn.BatchNorm1d(WIDTH) if bn else nn.Identity()
        self.act = ACTS[act]()
        self.residual = residual

    def forward(self, x):
        y = self.act(self.bn(self.lin(x)))
        return x + y if self.residual else y


def make(act, init, bn=False, residual=False):
    layers = [nn.Flatten(), nn.Linear(784, WIDTH), ACTS[act]()]
    layers += [Block(act, bn, residual) for _ in range(DEPTH - 1)]
    layers += [nn.Linear(WIDTH, 10)]
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            if init == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            else:
                nn.init.normal_(m.weight, std=0.1)
            nn.init.zeros_(m.bias)
    return net.to(DEVICE)


CONFIGS = {
    "sigmoid + Xavier": ("sigmoid", "xavier", False, False),
    "tanh + Xavier": ("tanh", "xavier", False, False),
    "ReLU + He": ("relu", "he", False, False),
    "sigmoid + Xavier + BN": ("sigmoid", "xavier", True, False),
    "sigmoid + Xavier + residual": ("sigmoid", "xavier", False, True),
}


def layer_grad_norms(net):
    return [m.weight.grad.norm().item() for m in net.modules() if isinstance(m, nn.Linear)]


def main():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("data", train=True, download=True, transform=tf)
    test = datasets.MNIST("data", train=False, download=True, transform=tf)
    tl = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, drop_last=True)
    vl = torch.utils.data.DataLoader(test, batch_size=1000)
    loss_fn = nn.CrossEntropyLoss()

    plt.figure(figsize=(8, 5))
    for name, cfg in CONFIGS.items():
        net = make(*cfg)
        opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
        # gradient profile at initialisation
        x, y = next(iter(tl))
        net.train(); opt.zero_grad()
        loss_fn(net(x.to(DEVICE)), y.to(DEVICE)).backward()
        plt.semilogy(layer_grad_norms(net), marker=".", label=name)
        # short training run
        for _ in range(EPOCHS):
            for x, y in tl:
                opt.zero_grad()
                loss_fn(net(x.to(DEVICE)), y.to(DEVICE)).backward()
                opt.step()
        net.eval(); correct = 0
        with torch.no_grad():
            for x, y in vl:
                correct += (net(x.to(DEVICE)).argmax(1).cpu() == y).sum().item()
        print(f"{name:30s} test acc = {correct / len(test):.4f}")
    plt.xlabel("layer index"); plt.ylabel("||dL/dW||"); plt.legend(); plt.grid(alpha=.3)
    plt.title(f"Per-layer gradient norm at init, depth {DEPTH} (MNIST)")
    plt.tight_layout(); plt.savefig("pytorch_layer_gradients.png", dpi=150)


if __name__ == "__main__":
    main()
