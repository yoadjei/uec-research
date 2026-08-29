"""Shared-support probe sampling.

Probe points must be in-distribution for both checkpoints, otherwise measured explanation change
confounds extrapolation with the effect of interest. On synthetic data the density ratio is closed
form, so membership is exact rather than estimated.
"""

import numpy as np

from .synthetic import Environment


def log_density_ratio(src: Environment, tgt: Environment, X: np.ndarray) -> np.ndarray:
    return tgt.log_density(X) - src.log_density(X)


def shared_support_probe(
    src: Environment,
    tgt: Environment,
    n: int,
    rng: np.random.Generator,
    tau: float = 2.0,
    oversample: int = 600,
):
    """Draw from a balanced source/target mixture, keep the overlap, subsample to `n`.

    Sampling the mixture rather than the source keeps the probe centred on the overlap region
    instead of the source mode.
    """
    pool = []
    drawn = 0
    while sum(len(p) for p in pool) < n and drawn < oversample * n:
        m = max(n, 4096)
        xs, _ = src.sample(m, rng)
        xt, _ = tgt.sample(m, rng)
        X = np.vstack([xs, xt])
        drawn += 2 * m
        pool.append(X[np.abs(log_density_ratio(src, tgt, X)) <= tau])

    kept = np.vstack(pool) if pool else np.empty((0, xs.shape[1]))
    if len(kept) < n:
        raise RuntimeError(
            f"shared support too thin: {len(kept)}/{n} kept from {drawn} draws at tau={tau}"
        )
    return kept[rng.choice(len(kept), size=n, replace=False)]


def overlap_fraction(src, tgt, rng, tau: float = 2.0, n: int = 20000) -> float:
    X, _ = src.sample(n, rng)
    return float(np.mean(np.abs(log_density_ratio(src, tgt, X)) <= tau))
