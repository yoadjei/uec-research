"""Faithfulness, measured on each checkpoint separately.

Stability and faithfulness are different constructs and the paper needs them apart. Stability is
`Δ` *between* checkpoints; faithfulness is how well an attribution describes *one* checkpoint. The
question E6 answers is whether the observed change is really faithfulness loss in disguise: if an
explainer stays faithful to both checkpoints and still disagrees with itself across them, the
disagreement is a property of the two functions, not a failure of the explainer.

On the synthetic generator there are two distinct notions and we report both:

- **faithfulness to the model** — deletion/insertion curves against the checkpoint's own output
- **fidelity to the mechanism** — agreement with the exact Bayes attribution, which exists only
  because the generator is known
"""

import numpy as np


def _curves(predict, X, A, baseline, descending=True):
    n, d = X.shape
    order = np.argsort(-np.abs(A) if descending else np.abs(A), axis=1)

    h_full = predict(X)
    h_base = predict(np.repeat(baseline[None, :], n, axis=0))
    span = h_full - h_base

    ins = np.empty((n, d + 1))
    dele = np.empty((n, d + 1))
    for j in range(d + 1):
        keep = np.zeros((n, d), bool)
        if j:
            rows = np.repeat(np.arange(n), j)
            keep[rows, order[:, :j].ravel()] = True
        x_ins = np.where(keep, X, baseline)
        x_del = np.where(keep, baseline, X)
        ins[:, j] = predict(x_ins)
        dele[:, j] = predict(x_del)

    ok = np.abs(span) > 1e-9
    norm = np.where(ok, span, 1.0)[:, None]
    return (ins - h_base[:, None]) / norm, (dele - h_base[:, None]) / norm, ok


def deletion_insertion(predict, X, A, baseline):
    """Returns (insertion AUC, deletion AUC, valid mask), normalised so a perfect explainer has
    insertion AUC near 1 and deletion AUC near 0."""
    ins, dele, ok = _curves(predict, X, A, baseline)
    return ins.mean(1), dele.mean(1), ok


def faithfulness(predict, X, A, baseline):
    """Insertion minus deletion AUC. Larger is more faithful; ~0 means the ranking carries no
    information about this model's output."""
    ins, dele, ok = deletion_insertion(predict, X, A, baseline)
    out = ins - dele
    return np.where(ok, out, np.nan)


def mechanism_fidelity(A, gt, normalise, distance):
    """1 - distance to the exact Bayes attribution. Only defined where the generator is known."""
    return 1.0 - distance(normalise(A), normalise(gt))
