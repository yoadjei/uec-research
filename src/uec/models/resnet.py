"""Compact ResNet for 32x32, with a smooth activation for the same reason the MLP has one:
Proposition 1 assumes C^1, and a ReLU network is not one, so IG completeness would hold only to
O(1/n_steps) and contaminate the bound check.

Explanations target a *fixed* class -- the source model's prediction -- held constant across
checkpoints. Explaining each checkpoint's own argmax would compare attributions for different
quantities and make the difference uninterpretable.
"""

import numpy as np
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.act = nn.SiLU()
        self.skip = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
            if stride != 1 or cin != cout
            else nn.Identity()
        )

    def forward(self, x):
        h = self.act(self.b1(self.c1(x)))
        return self.act(self.b2(self.c2(h)) + self.skip(x))


class SmallResNet(nn.Module):
    def __init__(self, n_classes: int = 10, width: int = 32, blocks=(2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.SiLU())
        stages, cin = [], width
        for i, n in enumerate(blocks):
            cout = width * 2**i
            for j in range(n):
                stages.append(Block(cin, cout, stride=2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(cin, n_classes)
        self.feature_layer = self.stages[-1]

    def forward(self, x):
        h = self.stages(self.stem(x))
        return self.head(h.mean(dim=(2, 3)))


class FixedClassLogit(nn.Module):
    """Wraps a classifier so it emits a single scalar: the logit of a per-input fixed class."""

    def __init__(self, model, targets):
        super().__init__()
        self.model = model
        self.register_buffer("targets", torch.as_tensor(targets, dtype=torch.long))
        self._offset = 0

    def set_offset(self, offset: int):
        self._offset = offset

    def forward(self, x):
        out = self.model(x)
        n = out.shape[0]
        t = self.targets[self._offset : self._offset + n]
        if t.shape[0] != n:  # captum expands a batch across integration steps
            t = t.repeat_interleave(n // t.shape[0])
        return out.gather(1, t.view(-1, 1)).squeeze(1)


@torch.no_grad()
def predict_logits(model, X, batch: int = 256):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(torch.as_tensor(X[i : i + batch], dtype=torch.float32)).numpy())
    return np.concatenate(out).astype(np.float64)


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)
