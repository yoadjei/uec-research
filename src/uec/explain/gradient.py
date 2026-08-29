"""Gradient-family explainers, all targeting the logit.

Computed in float64. This is not fussiness: Proposition 1(i) is an *equality*
(sum_j dIG_j = delta(x) - delta(b)) and E7 tests it per probe point, where delta is small by
construction. In float32 Captum's completeness error is ~5e-3 on logits of order 5, which is the
same size as the quantity being tested. Double precision removes numerical error as a confound.

Batching keeps IG at 64 steps affordable across the ~200 checkpoints an experiment produces.
"""

import copy

import numpy as np
import torch
from captum.attr import GradientShap, IntegratedGradients

CHUNK = 128
_DOUBLE_CACHE: dict[int, torch.nn.Module] = {}


def as_double(model):
    key = id(model)
    cached = _DOUBLE_CACHE.get(key)
    if cached is None or cached[0] is not _fingerprint(model):
        cached = (_fingerprint(model), copy.deepcopy(model).double().eval())
        _DOUBLE_CACHE[key] = cached
    return cached[1]


def _fingerprint(model):
    p = next(model.parameters())
    return (p.data_ptr(), float(p.detach().reshape(-1)[0]))


def _t(X):
    return torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=torch.float64)


def _chunks(n, size=CHUNK):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def saliency(model, X, **_):
    m = as_double(model)
    out = np.empty((len(X), np.shape(X)[1]), dtype=np.float64)
    for sl in _chunks(len(X)):
        x = _t(X[sl]).requires_grad_(True)
        out[sl] = torch.autograd.grad(m(x).sum(), x)[0].numpy()
    return out


def gradient_x_input(model, X, **_):
    return np.asarray(X, dtype=np.float64) * saliency(model, X)


def integrated_gradients(model, X, baseline=None, n_steps: int = 64, **_):
    m = as_double(model)
    ig = IntegratedGradients(m)
    b = _t(np.zeros(np.shape(X)[1]) if baseline is None else baseline).unsqueeze(0)
    out = np.empty((len(X), np.shape(X)[1]), dtype=np.float64)
    for sl in _chunks(len(X)):
        x = _t(X[sl])
        a = ig.attribute(x, baselines=b.expand_as(x), n_steps=n_steps, method="riemann_middle")
        out[sl] = a.detach().numpy()
    return out


def expected_gradients(model, X, background=None, n_samples: int = 32, run: int = 0, **_):
    """Path-integrated with a sampled baseline: stochastic, so it carries a real noise floor."""
    m = as_double(model)
    gs = GradientShap(m)
    bg = _t(background)
    out = np.empty((len(X), np.shape(X)[1]), dtype=np.float64)
    for sl in _chunks(len(X)):
        torch.manual_seed(9_000 + 17 * run)
        a = gs.attribute(_t(X[sl]), baselines=bg, n_samples=n_samples, stdevs=0.0)
        out[sl] = a.detach().numpy()
    return out


def smoothgrad(model, X, sigma: float = 0.15, n_samples: int = 32, run: int = 0, **_):
    X = np.asarray(X, dtype=np.float64)
    scale = sigma * (X.std(0, keepdims=True) + 1e-12)
    gen = torch.Generator().manual_seed(4_000 + 17 * run)
    acc = np.zeros_like(X)
    for _ in range(n_samples):
        acc += saliency(model, X + torch.randn(X.shape, generator=gen).numpy() * scale)
    return acc / n_samples
