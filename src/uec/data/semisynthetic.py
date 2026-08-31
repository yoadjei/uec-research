"""A known mechanism over *real* covariates.

omega is defined against the Bayes-optimal predictor, so computing it exactly requires knowing that
predictor. No observational dataset supplies one. The synthetic generator solves this by making up
the covariates too, which leaves an obvious objection: the closed form might only behave because the
inputs are well-conditioned Gaussians.

This module removes that half of the assumption. Covariates are real ACS rows -- real marginals,
real skew, real discreteness, real correlations, real duplicated codes -- and only the *labels* are
regenerated from a mechanism we choose. That mechanism is then Bayes-optimal by construction, so
omega is exact, and the exactness does not depend on the covariate distribution:

    h(x) = beta . x + sum_{(j,k) in pairs} gamma_jk x_j x_k
    IG_j(x; b) = (x_j - b_j) (beta_j + sum_k gamma_jk (b_k + x_k) / 2)

The path integral collapses analytically for any x whatsoever; that is a property of the polynomial,
not of the data. `Mechanism.ig_quadrature` re-derives it numerically so the claim is checked on the
real rows rather than asserted.

What this does and does not establish: it validates that the omega machinery is correct and behaves
as claimed on realistic covariates. It does not validate omega against real *labels* -- that is not
possible for any method, since the real Bayes predictor is unobservable. The scope is stated in the
paper rather than papered over.
"""

from dataclasses import dataclass

import numpy as np

from .folktables_data import SourceScaler, load_acs


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@dataclass(frozen=True)
class Mechanism:
    """A quadratic logit. Bayes-optimal for the labels it generates, by construction."""

    beta: np.ndarray
    gamma: np.ndarray
    pairs: tuple

    def logodds(self, X):
        out = X @ self.beta
        for (j, k), c in zip(self.pairs, self.gamma):
            out = out + c * X[:, j] * X[:, k]
        return out

    def label(self, X, rng):
        return (rng.uniform(size=len(X)) < _sigmoid(self.logodds(X))).astype(np.int64)

    def ig(self, X, baseline):
        """Closed-form integrated gradients of the logit. Exact for arbitrary X."""
        coef = np.repeat(self.beta[None, :], len(X), axis=0).astype(float)
        for (j, k), c in zip(self.pairs, self.gamma):
            coef[:, j] += c * (baseline[k] + X[:, k]) / 2.0
            coef[:, k] += c * (baseline[j] + X[:, j]) / 2.0
        return (X - baseline) * coef

    def ig_quadrature(self, X, baseline, steps: int = 512):
        """Midpoint quadrature of the same integral, to check `ig` on the real rows."""
        alpha = ((np.arange(steps) + 0.5) / steps)[:, None, None]
        path = baseline[None, None, :] + alpha * (X - baseline)[None, :, :]
        grads = np.empty_like(path)
        grads[:] = self.beta
        for (j, k), c in zip(self.pairs, self.gamma):
            grads[:, :, j] += c * path[:, :, k]
            grads[:, :, k] += c * path[:, :, j]
        return (X - baseline) * grads.mean(0)


def fit_mechanism(X, y, pairs=((0, 2), (3, 7)), gamma=(0.4, -0.3), seed: int = 0):
    """Take beta from a ridge fit to the *real* labels.

    An arbitrary beta would make the labelling task unrelated to income and the covariates would
    stop being informative in a realistic way. Fitting to the real outcome keeps the mechanism a
    plausible one for these features; the interactions are added so the reference is not linear,
    since a linear logit makes IG trivially equal to beta * (x - b) and would not exercise the
    closed form.
    """
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(X, y)
    return Mechanism(beta=lr.coef_.ravel().copy(), gamma=np.asarray(gamma, float), pairs=pairs)


def covariate_tilt(X, rng, n, feature: int = 0, strength: float = 1.0):
    """Resample real rows with weight exp(strength * x_feature), leaving the mechanism alone.

    This is a genuine covariate shift over real data -- the rows are real and unmodified, only their
    frequency changes -- and omega is exactly zero because the labelling mechanism is untouched.
    """
    w = np.exp(strength * np.clip(X[:, feature], -3, 3))
    p = w / w.sum()
    return rng.choice(len(X), n, replace=True, p=p)


def concept_shift(mech: Mechanism, delta_beta, seed: int = 0):
    """Perturb the mechanism itself. omega is then non-zero and computable in closed form."""
    beta = mech.beta.copy()
    for j, d in delta_beta.items():
        beta[j] += d
    return Mechanism(beta=beta, gamma=mech.gamma, pairs=mech.pairs)


def add_redundant(Z, mech, n_copies: int, sigma: float, rng):
    """Append noisy copies of the most-weighted features, carrying no mechanism weight.

    The copies are collinear with informative features but beta is zero on them, so they change
    omega not at all while giving the model somewhere to move attribution mass without changing
    predictions. This is the synthetic generator's REDUNDANT block, reconstructed on real
    covariates, and it is the one structural feature real ACS data lacks (its largest off-diagonal
    correlation is 0.48; the synthetic causal-redundant pairs sit at 0.94).
    """
    src = np.argsort(-np.abs(mech.beta))[:n_copies]
    extra = Z[:, src] + sigma * rng.normal(size=(len(Z), n_copies))
    extra = (extra - extra.mean(0)) / (extra.std(0) + 1e-8)
    Zr = np.hstack([Z, extra])
    padded = Mechanism(beta=np.concatenate([mech.beta, np.zeros(n_copies)]),
                       gamma=mech.gamma, pairs=mech.pairs)
    return Zr, padded


def add_noise_features(Z, mech, n_noise: int, rng):
    """Append independent features carrying no information and no mechanism weight.

    The synthetic generator has eight of these; every real ACS feature carries weight. They differ
    from redundant copies in being uncorrelated with anything, so they test a different hypothesis:
    that attributions need somewhere *uninformative* to wander, rather than somewhere collinear.
    """
    extra = rng.normal(size=(len(Z), n_noise))
    Zn = np.hstack([Z, extra])
    padded = Mechanism(beta=np.concatenate([mech.beta, np.zeros(n_noise)]),
                       gamma=mech.gamma, pairs=mech.pairs)
    return Zn, padded


def load_semisynthetic(state: str = "CA", year: str = "2018", seed: int = 0,
                       n_redundant: int = 0, sigma_r: float = 0.35, n_noise: int = 0):
    """Real ACS covariates, standardised on the source, with a mechanism fitted to real labels."""
    dom = load_acs(state, year)
    scaler = SourceScaler(dom.X)
    Z = scaler(dom.X)
    mech = fit_mechanism(Z, dom.y, seed=seed)
    rng = np.random.default_rng(seed)
    if n_redundant:
        Z, mech = add_redundant(Z, mech, n_redundant, sigma_r, rng)
    if n_noise:
        Z, mech = add_noise_features(Z, mech, n_noise, rng)
    return Z, dom.y, mech
