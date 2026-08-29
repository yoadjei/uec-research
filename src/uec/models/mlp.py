import numpy as np
import torch
from torch import nn


ACTIVATIONS = {"silu": nn.SiLU, "gelu": nn.GELU, "softplus": nn.Softplus, "relu": nn.ReLU}


class MLP(nn.Module):
    """Single-logit binary classifier. Explainers target the logit, matching the theory and the
    closed-form Bayes reference, which are both stated in log-odds space.

    The default activation is smooth, not ReLU. Proposition 1 is stated for C^1 functions, and a
    ReLU network is not one: its gradient is piecewise constant, so Riemann quadrature for IG
    converges only as O(1/n_steps) and completeness holds to ~2e-3 at 512 steps. That error is the
    same order as the quantity E7 tests. ReLU is kept as an ablation, not as the default.
    """

    def __init__(self, d_in: int, hidden=(64, 64), activation: str = "silu"):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def logits(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.as_tensor(X, dtype=torch.float32)).numpy().astype(np.float64)


def probabilities(model: nn.Module, X: np.ndarray) -> np.ndarray:
    z = logits(model, X)
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    return float(((probabilities(model, X) >= 0.5).astype(int) == y).mean())


def expected_calibration_error(model, X, y, bins: int = 15) -> float:
    p = probabilities(model, X)
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
