"""Ranking-reliability partition.

The Attribution Impossibility (arXiv:2605.21492) shows no ranking is faithful, stable and complete
under collinearity, and fixes a Z-threshold of 1.96 below which a pairwise ranking is unreliable
regardless of any shift. We reproduce that test on the probe set so the headline result can be
reported on the reliable partition: if unwarranted change survives there, "it is just collinearity"
is not an available deflation.

The statistic is our operationalisation -- a paired z-test that feature j outranks feature k by
attribution magnitude across probe points -- not a verbatim reimplementation of their estimator.
"""

import numpy as np

Z_CRIT = 1.96


def ranking_z(A: np.ndarray) -> np.ndarray:
    """Pairwise z for |a_j| > |a_k| across probe points; entry (j,k)."""
    M = np.abs(np.asarray(A, float))
    n = len(M)
    diff = M[:, :, None] - M[:, None, :]
    mean = diff.mean(0)
    se = diff.std(0, ddof=1) / np.sqrt(n)
    return np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)


def unreliable_pairs(A, z_crit: float = Z_CRIT):
    z = ranking_z(A)
    d = z.shape[0]
    return [(j, k) for j in range(d) for k in range(j + 1, d) if abs(z[j, k]) < z_crit]


def reliable_features(A, z_crit: float = Z_CRIT) -> np.ndarray:
    """Features whose ranking is resolvable against every other feature."""
    tied = {i for pair in unreliable_pairs(A, z_crit) for i in pair}
    return np.array([i for i in range(np.shape(A)[1]) if i not in tied], dtype=int)
