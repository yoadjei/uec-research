"""Per-point measurement of the Propositions.

Proposition 1(i) says |sum_j dIG_j(x)| <= |delta(x)| + |delta(b)| for *every* point, with no escape
clause. Proposition 2 says no such bound exists for gradient x input. Proposition 3 holds only if
coalition values are preserved, which data-level preservation does not deliver. All three are
measured here rather than asserted.
"""

import numpy as np

from ..explain.gradient import gradient_x_input, integrated_gradients
from ..models.mlp import logits


def _endpoint_gap(model_a, model_b, X, baseline):
    da = logits(model_a, X) - logits(model_b, X)
    db = logits(model_a, baseline[None, :])[0] - logits(model_b, baseline[None, :])[0]
    return np.abs(da) + abs(db), da, db


def ig_bound(model_a, model_b, X, baseline, n_steps: int = 256) -> dict:
    """Slack must be <= 1 under Proposition 1(i), up to quadrature error.

    `residual` checks the sharper, exact form of the proposition: the aggregate gap *equals*
    delta(x) - delta(b). `quad_tol` propagates that residual into the slack scale, so the bound
    can be tested against a derived tolerance rather than a guessed one.
    """
    a = integrated_gradients(model_a, X, baseline=baseline, n_steps=n_steps)
    b = integrated_gradients(model_b, X, baseline=baseline, n_steps=n_steps)
    agg = (a - b).sum(1)
    denom, da, db = _endpoint_gap(model_a, model_b, X, baseline)
    residual = agg - (da - db)
    ok = denom > 0
    return {
        "slack": np.divide(np.abs(agg), denom, out=np.full(len(X), np.nan), where=ok),
        "aggregate": agg,
        "residual": residual,
        "denom": denom,
        "quad_tol": float(np.abs(residual[ok] / denom[ok]).max()) if ok.any() else np.nan,
    }


def gradient_bound(model_a, model_b, X, baseline) -> dict:
    agg = (gradient_x_input(model_a, X) - gradient_x_input(model_b, X)).sum(1)
    denom, _, _ = _endpoint_gap(model_a, model_b, X, baseline)
    return {
        "slack": np.divide(np.abs(agg), denom, out=np.full(len(X), np.nan), where=denom > 0),
        "aggregate": agg,
        "denom": denom,
    }


def coalition_epsilon(model_a, model_b, X, background, n_coalitions: int = 64, rng=None):
    """max_S |v_a(S) - v_b(S)| over sampled coalitions, against |f_a(x) - f_b(x)|.

    v(S) = E_{z~background}[ f(x_S ; z_{-S}) ]: masked composites that lie off the data manifold,
    where fine-tuning places no constraint on either model.
    """
    rng = rng or np.random.default_rng(0)
    X, bg = np.asarray(X, float), np.asarray(background, float)
    n, d = X.shape
    out = np.zeros(n)

    for i in range(n):
        masks = rng.random((n_coalitions, d)) < 0.5
        comp = np.where(masks[:, None, :], X[i][None, None, :], bg[None, :, :])
        flat = comp.reshape(-1, d)
        va = logits(model_a, flat).reshape(n_coalitions, len(bg)).mean(1)
        vb = logits(model_b, flat).reshape(n_coalitions, len(bg)).mean(1)
        out[i] = np.abs(va - vb).max()

    eps_data = np.abs(logits(model_a, X) - logits(model_b, X))
    return out, eps_data
