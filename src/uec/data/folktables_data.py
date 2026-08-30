"""ACS/Folktables loaders with documented natural shifts.

Real data has no closed-form density ratio, but a calibrated domain classifier estimates it
directly: logit P(domain = target | x) = log[p_T(x)/p_S(x)] + log(n_T/n_S). The shared-support
screen is therefore the same criterion as on synthetic data, with an estimated rather than exact
ratio -- and that substitution is the approximation, stated as such.

Because omega is not computable here, legitimacy claims are restricted to shifts that pass
`mechanism_stability`, which tests whether a source-trained model stays calibrated on the
shared-support region of the target. That is a necessary condition for P(Y|X) stability, not a
sufficient one, and the paper says so.
"""

from dataclasses import dataclass

import numpy as np
from folktables import ACSDataSource, ACSIncome
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..paths import ROOT

CACHE = ROOT / "data" / "acs"
FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]


@dataclass
class TabularDomain:
    name: str
    X: np.ndarray
    y: np.ndarray


def load_acs(state: str, year: str = "2018") -> TabularDomain:
    CACHE.mkdir(parents=True, exist_ok=True)
    ds = ACSDataSource(
        survey_year=year, horizon="1-Year", survey="person", root_dir=str(CACHE)
    )
    df = ds.get_data(states=[state], download=True)
    X, y, _ = ACSIncome.df_to_numpy(df)
    return TabularDomain(f"{state}{year}", X.astype(np.float64), y.astype(np.int64))


class SourceScaler:
    """Standardisation fitted on the source and frozen. A per-checkpoint or per-domain scaler
    would change the input parameterisation between checkpoints and make attribution differences
    uninterpretable."""

    def __init__(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8

    def __call__(self, X):
        return (np.asarray(X, float) - self.mu) / self.sd


def domain_logit(Xs, Xt, seed: int = 0, n_splits: int = 4):
    """Cross-fitted log density ratio estimate for the pooled points, plus the fitted AUC."""
    X = np.vstack([Xs, Xt])
    d = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))]).astype(int)
    out = np.zeros(len(X))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, d):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], d[tr])
        out[te] = clf.decision_function(X[te])
    return out, float(roc_auc_score(d, out))


def shared_support_probe(Xs, Xt, n: int, rng, tau: float = 1.0, seed: int = 0, ys=None, yt=None):
    """Points whose estimated log density ratio lies within tau of the prior-corrected origin.

    Returns a `Probe` carrying X, the true labels, the domain each point came from, and the pooled
    classifier AUC. Labels must be carried through the screen, not reconstructed afterwards: the
    calibration check in `mechanism_stability` is meaningless against approximated labels.
    """
    X = np.vstack([Xs, Xt])
    origin = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))]).astype(int)
    y = None if ys is None or yt is None else np.concatenate([ys, yt])

    logit, auc = domain_logit(Xs, Xt, seed=seed)
    keep = np.abs(logit - np.log(len(Xt) / len(Xs))) <= tau
    if keep.sum() < n:
        raise RuntimeError(
            f"shared support too thin: {int(keep.sum())}/{n} at tau={tau} (auc={auc:.3f})"
        )

    # Draw evenly from each domain's overlap. Pooling and subsampling would follow the domain
    # sizes, and with a small target state (SD has 4899 rows against CA's 195665) the probe would
    # be almost entirely source points -- not a shared-support probe at all, and too few target
    # points to compute the calibration-transfer check that stands in for omega.
    idx = []
    for d in (0, 1):
        pool = np.flatnonzero(keep & (origin == d))
        take = min(n // 2, len(pool))
        idx.append(rng.choice(pool, take, replace=False))
    chosen = np.concatenate(idx)
    if len(chosen) < n:
        rest = np.setdiff1d(np.flatnonzero(keep), chosen)
        chosen = np.concatenate([chosen, rng.choice(rest, n - len(chosen), replace=False)])
    chosen = rng.permutation(chosen)
    return Probe(X[chosen], None if y is None else y[chosen], origin[chosen], auc)


@dataclass
class Probe:
    X: np.ndarray
    y: np.ndarray | None
    origin: np.ndarray
    domain_auc: float

    def by_domain(self, which: int):
        m = self.origin == which
        return self.X[m], (None if self.y is None else self.y[m]), m


def probe_balance_auc(probe, origin, seed: int = 0) -> float:
    """Cross-fitted domain AUC *within* the probe. Near 0.5 means the probe really is in the
    overlap: source and target points there are not separable."""
    if len(np.unique(origin)) < 2 or min(np.bincount(origin)) < 10:
        return np.nan
    pred = np.zeros(len(probe))
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=seed).split(probe, origin):
        clf = LogisticRegression(max_iter=2000).fit(probe[tr], origin[tr])
        pred[te] = clf.decision_function(probe[te])
    return float(roc_auc_score(origin, pred))


def mechanism_stability(y_source, p_source, y_target, p_target, bins: int = 10):
    """Reliability-curve distance between source and target on the shared support.

    Small distance means a source-trained model's conditional probabilities transfer, which is
    what P(Y|X) stability would produce. Necessary, not sufficient.
    """
    def curve(p, y):
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
        return np.array([
            y[idx == b].mean() - p[idx == b].mean() if (idx == b).sum() > 20 else np.nan
            for b in range(bins)
        ])

    a, b = curve(p_source, y_source), curve(p_target, y_target)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a[ok] - b[ok]).mean()) if ok.any() else np.nan
