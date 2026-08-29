"""Exact integrated gradients of the Bayes-optimal logit.

This is the warranted-change reference omega. It is a closed form, not an estimate: for the
polynomial-plus-linear Bayes logit the path integral has an analytic solution, and the quadrature
version below exists only to check it.
"""

import numpy as np

from .synthetic import CAUSAL, D, S0, Environment


def gt_attributions(env: Environment, X: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """IG of `env.bayes_logodds` at `X` w.r.t. `baseline`.

    For h(x) = b'x + sum_{(j,k)} c_jk x_j x_k the path integral collapses to
    IG_j = (x_j - b_j) * (beta_j + sum_k c_jk (b_k + x_k) / 2).
    """
    xc, bc = X[:, CAUSAL], baseline[CAUSAL]
    coef = np.repeat(env.beta[None, :], len(X), axis=0).astype(float)
    for (j, k), c in zip(env.pairs, env.gamma):
        coef[:, j] += c * (bc[k] + xc[:, k]) / 2.0
        coef[:, k] += c * (bc[j] + xc[:, j]) / 2.0

    A = np.zeros((len(X), D))
    A[:, CAUSAL] = (xc - bc) * coef
    A[:, S0] = (X[:, S0] - baseline[S0]) * env.shortcut_coef
    return A


def grad_bayes_logodds(env: Environment, Z: np.ndarray) -> np.ndarray:
    g = np.zeros_like(Z)
    g[:, CAUSAL] = env.beta
    for (j, k), c in zip(env.pairs, env.gamma):
        g[:, CAUSAL[j]] += c * Z[:, CAUSAL[k]]
        g[:, CAUSAL[k]] += c * Z[:, CAUSAL[j]]
    g[:, S0] += env.shortcut_coef
    return g


def gt_attributions_quadrature(env, X, baseline, steps: int = 512) -> np.ndarray:
    alpha = ((np.arange(steps) + 0.5) / steps)[:, None, None]
    path = baseline[None, None, :] + alpha * (X - baseline)[None, :, :]
    grads = np.stack([grad_bayes_logodds(env, path[i]) for i in range(steps)])
    return (X - baseline) * grads.mean(0)


def omega_reference(src: Environment, tgt: Environment, X, baseline, normalise, distance):
    """Per-point warranted change: how far a mechanism-tracking explainer *should* move."""
    a_s = normalise(gt_attributions(src, X, baseline))
    a_t = normalise(gt_attributions(tgt, X, baseline))
    return distance(a_s, a_t)
