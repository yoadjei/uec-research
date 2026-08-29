"""CIFAR-10 with locally generated corruptions.

Corruptions are implemented here rather than downloaded as CIFAR-10-C: the archive is 2.9 GB, and
the four families used are reproducible in a few lines. They serve only as a covariate-shift
*generator* for the model update -- input-perturbation stability is FASS's object, not ours.

The spurious variant plants a class-correlated corner tint in the source and removes it in the
target, which is the vision analogue of the synthetic shortcut: the model should stop using it,
so the warranted change is non-zero.
"""

import numpy as np
import torch
from torchvision import datasets, transforms

from ..paths import ROOT

CACHE = ROOT / "data" / "cifar"
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
SEVERITY = {"gaussian_noise": 0.08, "contrast": 0.4, "brightness": 0.25, "pixelate": 3}


def load_cifar(train: bool = True):
    CACHE.mkdir(parents=True, exist_ok=True)
    ds = datasets.CIFAR10(str(CACHE), train=train, download=True, transform=transforms.ToTensor())
    X = ds.data.astype(np.float32) / 255.0
    return X, np.asarray(ds.targets, dtype=np.int64)


def gaussian_noise(X, c=SEVERITY["gaussian_noise"], rng=None):
    rng = rng or np.random.default_rng(0)
    return np.clip(X + rng.normal(0, c, X.shape).astype(np.float32), 0, 1)


def contrast(X, c=SEVERITY["contrast"], **_):
    m = X.mean(axis=(1, 2), keepdims=True)
    return np.clip((X - m) * c + m, 0, 1)


def brightness(X, c=SEVERITY["brightness"], **_):
    return np.clip(X + c, 0, 1)


def pixelate(X, c=SEVERITY["pixelate"], **_):
    n, h, w, ch = X.shape
    small = X.reshape(n, h // c, c, w // c, c, ch).mean(axis=(2, 4))
    return np.repeat(np.repeat(small, c, axis=1), c, axis=2)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "contrast": contrast,
    "brightness": brightness,
    "pixelate": pixelate,
}


def corrupt(X, kinds=("gaussian_noise", "contrast"), rng=None):
    out = X
    for k in kinds:
        out = CORRUPTIONS[k](out, rng=rng) if k == "gaussian_noise" else CORRUPTIONS[k](out)
    return out.astype(np.float32)


def plant_shortcut(X, y, strength: float = 0.35, patch: int = 6, rng=None):
    """Tint a corner patch by class parity. Predictive in the source, absent in the target."""
    rng = rng or np.random.default_rng(0)
    out = X.copy()
    tint = np.where(y % 2 == 0, strength, -strength).astype(np.float32)
    out[:, :patch, :patch, 0] = np.clip(out[:, :patch, :patch, 0] + tint[:, None, None], 0, 1)
    return out


def normalise(X):
    return ((X - MEAN) / STD).transpose(0, 3, 1, 2).copy()


def to_tensor(X):
    return torch.as_tensor(normalise(X), dtype=torch.float32)


def subset(X, y, n, rng, classes=None):
    if classes is not None:
        m = np.isin(y, classes)
        X, y = X[m], y[m]
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[idx], y[idx]
