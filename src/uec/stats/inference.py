"""Statistics. Seeds are the unit of dependence, so every confidence interval bootstraps over
seeds rather than over probe points, which are paired within a seed and not independent.
"""

import numpy as np
from scipy.stats import kendalltau, wilcoxon


def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b), computed by rank rather than by the O(n^2) pair scan."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n, m = len(a), len(b)
    pooled = np.concatenate([a, b])
    order = np.argsort(pooled, kind="mergesort")
    ranks = np.empty(len(pooled))
    ranks[order] = np.arange(1, len(pooled) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(pooled, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]

    r1 = ranks[:n].sum()
    u1 = r1 - n * (n + 1) / 2
    return float(2 * u1 / (n * m) - 1)


def paired_test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="zsplit").pvalue)


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0, stat=np.mean):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = stat(v[rng.integers(0, v.size, size=(n_boot, v.size))], axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi)


def ratio_ci(numer, denom, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Paired bootstrap over seeds: numerator and denominator come from the same seed, so they
    must be resampled together."""
    a, b = np.asarray(numer, float), np.asarray(denom, float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    draws = a[idx].mean(1) / b[idx].mean(1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() / b.mean()), float(lo), float(hi)


def ranking_agreement(scores_a: dict, scores_b: dict) -> float:
    keys = sorted(set(scores_a) & set(scores_b))
    if len(keys) < 3:
        return np.nan
    return float(kendalltau([scores_a[k] for k in keys], [scores_b[k] for k in keys]).statistic)
