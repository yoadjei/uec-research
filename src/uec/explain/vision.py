"""Vision explainers.

Grad-CAM lives on a coarse feature grid rather than on pixels, so its attributions are never
compared against a pixel-space explainer's on a raw distance. UEC compares each explainer only
with its own floors, which is what makes cross-family comparison legitimate at all.
"""

import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerGradCam, Saliency

CHUNK = 32


def _batches(n, size=CHUNK):
    for i in range(0, n, size):
        yield i, min(i + size, n)


def _tensor(X):
    return torch.as_tensor(X, dtype=torch.float32)


def vision_saliency(wrapped, X, **_):
    out = []
    for a, b in _batches(len(X)):
        wrapped.set_offset(a)
        x = _tensor(X[a:b]).requires_grad_(True)
        out.append(Saliency(wrapped).attribute(x, abs=False).detach().numpy())
    return np.concatenate(out).reshape(len(X), -1).astype(np.float64)


def vision_gradient_x_input(wrapped, X, **_):
    return np.asarray(X, np.float64).reshape(len(X), -1) * vision_saliency(wrapped, X)


def vision_integrated_gradients(wrapped, X, n_steps: int = 32, **_):
    ig = IntegratedGradients(wrapped)
    out = []
    for a, b in _batches(len(X)):
        wrapped.set_offset(a)
        x = _tensor(X[a:b])
        out.append(
            ig.attribute(x, baselines=torch.zeros_like(x), n_steps=n_steps,
                         method="riemann_middle").detach().numpy()
        )
    return np.concatenate(out).reshape(len(X), -1).astype(np.float64)


def vision_grad_cam(wrapped, X, layer=None, **_):
    cam = LayerGradCam(wrapped, layer or wrapped.model.feature_layer)
    out = []
    for a, b in _batches(len(X)):
        wrapped.set_offset(a)
        out.append(cam.attribute(_tensor(X[a:b]), relu_attributions=False).detach().numpy())
    return np.concatenate(out).reshape(len(X), -1).astype(np.float64)


def vision_smoothgrad(wrapped, X, sigma: float = 0.1, n_samples: int = 8, run: int = 0, **_):
    X = np.asarray(X, np.float32)
    gen = torch.Generator().manual_seed(3_100 + 17 * run)
    scale = sigma * float(X.std())
    acc = np.zeros((len(X), X[0].size))
    for _ in range(n_samples):
        noise = torch.randn(X.shape, generator=gen).numpy().astype(np.float32) * scale
        acc += vision_saliency(wrapped, X + noise)
    return acc / n_samples


VISION_EXPLAINERS = {
    "saliency": (vision_saliency, False),
    "gradient_x_input": (vision_gradient_x_input, False),
    "integrated_gradients": (vision_integrated_gradients, False),
    "grad_cam": (vision_grad_cam, False),
    "smoothgrad": (vision_smoothgrad, True),
}
