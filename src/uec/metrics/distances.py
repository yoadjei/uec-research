"""Row-wise attribution distances, all in [0, 1], larger = more different.

None of these is novel; they are the standard stability distances (Alvarez-Melis & Jaakkola 2018;
Yeh et al. 2019; Agarwal et al. 2022) and are used here only as instruments. `spearman` is primary
because it is what RSP and FASS report, which keeps our numbers comparable to theirs.
"""

import numpy as np
from scipy.stats import rankdata

TOPK = 5


def _rank(a):
    return rankdata(a, axis=-1)


def d_spearman(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ra, rb = _rank(A), _rank(B)
    ra = ra - ra.mean(-1, keepdims=True)
    rb = rb - rb.mean(-1, keepdims=True)
    num = (ra * rb).sum(-1)
    den = np.sqrt((ra**2).sum(-1) * (rb**2).sum(-1))
    rho = np.divide(num, den, out=np.ones_like(num, dtype=float), where=den > 0)
    return (1.0 - rho) / 2.0


def d_topk(A: np.ndarray, B: np.ndarray, k: int = TOPK) -> np.ndarray:
    ta = np.argsort(-A, axis=-1)[..., :k]
    tb = np.argsort(-B, axis=-1)[..., :k]
    out = np.empty(len(A))
    for i, (x, y) in enumerate(zip(ta, tb)):
        inter = len(set(x.tolist()) & set(y.tolist()))
        out[i] = 1.0 - inter / (2 * k - inter)
    return out


def d_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(A, axis=-1)
    nb = np.linalg.norm(B, axis=-1)
    den = na * nb
    cos = np.divide((A * B).sum(-1), den, out=np.ones(len(A)), where=den > 0)
    return (1.0 - np.clip(cos, -1.0, 1.0)) / 2.0


def d_l1(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(A - B).sum(-1)


def topk_with(k: int):
    def f(A, B):
        return d_topk(A, B, k=k)

    f.__name__ = f"d_topk{k}"
    return f


DISTANCES = {"spearman": d_spearman, "topk": d_topk, "cosine": d_cosine, "l1": d_l1}
# l1 is primary. Rank correlation is the natural choice for comparability with RSP and FASS, but
# it is dominated by tie structure when many features carry zero attribution: on a reference
# supported on 5 of 20 features a full sign flip of the mechanism registers as 1-rho = 0.002, while
# the same change is 0.11 in l1. Spearman is retained as the comparability ablation.
PRIMARY = "l1"

# k must be smaller than the support of the reference: top-k Jaccard cannot see a change that
# reallocates mass *within* a set of exactly k informative features.
ABLATION_DISTANCES = {**DISTANCES, "topk3": topk_with(3), "topk10": topk_with(10)}
