"""The decomposition.

UEC is deliberately a difference, not a ratio. A ratio to output change (ROS, Agarwal et al. 2022)
is undefined and explosive exactly in the prediction-preserving regime this work is about.
"""

from dataclasses import dataclass, field

import numpy as np


def change(A, B, normalise, distance, features=None) -> np.ndarray:
    A, B = np.asarray(A, float), np.asarray(B, float)
    if features is not None:
        A, B = A[:, features], B[:, features]
    return distance(normalise(A), normalise(B))


def preserved_mask(p_source: np.ndarray, p_updated: np.ndarray, eps: float) -> np.ndarray:
    return np.abs(p_source - p_updated) <= eps


@dataclass
class Summary:
    delta: float
    omega: float
    nu: float
    rho_null: float
    rho_seed: float
    uec: float
    ratio: float
    ratio_seed: float
    exceedance: float
    floor_q95: float
    n_probe: int
    n_preserved: int
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d["preserved_frac"] = self.n_preserved / self.n_probe if self.n_probe else np.nan
        return {**d, **self.extra}


def summarise(delta, omega, nu, rho_null, rho_seed, n_probe, **extra) -> Summary:
    delta, omega = np.asarray(delta, float), np.asarray(omega, float)
    nu, rho_null, rho_seed = (np.asarray(x, float) for x in (nu, rho_null, rho_seed))

    pool = np.concatenate([nu, rho_null]) if nu.size else rho_null
    thr = float(np.quantile(pool, 0.95)) if pool.size else np.nan

    m_delta, m_omega, m_nu = delta.mean(), omega.mean(), nu.mean() if nu.size else 0.0
    m_null, m_seed = rho_null.mean(), rho_seed.mean()

    return Summary(
        delta=float(m_delta),
        omega=float(m_omega),
        nu=float(m_nu),
        rho_null=float(m_null),
        rho_seed=float(m_seed),
        uec=float(m_delta - m_omega - max(m_nu, m_null)),
        ratio=float(m_delta / m_null) if m_null > 0 else np.nan,
        ratio_seed=float(m_delta / m_seed) if m_seed > 0 else np.nan,
        exceedance=float((delta > thr).mean()) if delta.size and np.isfinite(thr) else np.nan,
        floor_q95=thr,
        n_probe=int(n_probe),
        n_preserved=int(delta.size),
        extra=extra,
    )
