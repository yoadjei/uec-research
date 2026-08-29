"""Structural generator with a closed-form Bayes-optimal logit.

Feature blocks (d = 20):
    causal    0:5    drive Y
    shortcut  5:8    only index 5 (S0) is coupled to Y, and only in environments with a > 0
    redundant 8:12   noisy copies of causal[0:4]; collinear by construction
    noise     12:20  independent of everything

Because S0 is drawn from Y and is conditionally independent of X_C given Y, the Bayes logit is
available in closed form (see `Environment.bayes_logodds`). That closed form is what makes the
warranted-change reference omega exact rather than estimated.
"""

from dataclasses import dataclass, field, replace

import numpy as np

D = 20
CAUSAL = np.arange(0, 5)
SHORTCUT = np.arange(5, 8)
REDUNDANT = np.arange(8, 12)
NOISE = np.arange(12, 20)
S0 = 5
BLOCKS = {"causal": CAUSAL, "shortcut": SHORTCUT, "redundant": REDUNDANT, "noise": NOISE}


def _logn(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


@dataclass
class Environment:
    name: str
    beta: np.ndarray
    gamma: np.ndarray
    pairs: tuple = ((0, 1), (2, 3))
    mu_c: np.ndarray = field(default_factory=lambda: np.zeros(len(CAUSAL)))
    sd_c: np.ndarray = field(default_factory=lambda: np.ones(len(CAUSAL)))
    mu_n: np.ndarray = field(default_factory=lambda: np.zeros(len(NOISE)))
    sd_n: np.ndarray = field(default_factory=lambda: np.ones(len(NOISE)))
    a: float = 0.0
    sigma_s: float = 0.8
    sigma_r: float = 0.35

    @property
    def shortcut_coef(self) -> float:
        """log p(s|y=1) - log p(s|y=0) = 2as/sigma_s^2 for s|y ~ N(a(2y-1), sigma_s^2)."""
        return 2.0 * self.a / self.sigma_s**2

    def g(self, xc: np.ndarray) -> np.ndarray:
        out = xc @ self.beta
        for (j, k), c in zip(self.pairs, self.gamma):
            out = out + c * xc[:, j] * xc[:, k]
        return out

    def bayes_logodds(self, X: np.ndarray) -> np.ndarray:
        return self.g(X[:, CAUSAL]) + self.shortcut_coef * X[:, S0]

    def sample(self, n: int, rng: np.random.Generator):
        xc = rng.normal(self.mu_c, self.sd_c, size=(n, len(CAUSAL)))
        y = (rng.uniform(size=n) < _sigmoid(self.g(xc))).astype(np.int64)

        X = np.empty((n, D))
        X[:, CAUSAL] = xc
        X[:, S0] = self.a * (2 * y - 1) + self.sigma_s * rng.normal(size=n)
        X[:, SHORTCUT[1:]] = rng.normal(size=(n, len(SHORTCUT) - 1))
        X[:, REDUNDANT] = xc[:, : len(REDUNDANT)] + self.sigma_r * rng.normal(
            size=(n, len(REDUNDANT))
        )
        X[:, NOISE] = rng.normal(self.mu_n, self.sd_n, size=(n, len(NOISE)))
        return X, y

    def log_density(self, X: np.ndarray) -> np.ndarray:
        xc = X[:, CAUSAL]
        lp = _logn(xc, self.mu_c, self.sd_c).sum(1)
        lp += _logn(X[:, NOISE], self.mu_n, self.sd_n).sum(1)
        lp += _logn(X[:, REDUNDANT], xc[:, : len(REDUNDANT)], self.sigma_r).sum(1)
        lp += _logn(X[:, SHORTCUT[1:]], 0.0, 1.0).sum(1)

        p1 = _sigmoid(self.g(xc))
        s = X[:, S0]
        mix = p1 * np.exp(_logn(s, self.a, self.sigma_s)) + (1 - p1) * np.exp(
            _logn(s, -self.a, self.sigma_s)
        )
        return lp + np.log(np.clip(mix, 1e-300, None))


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def base_environment(a: float = 0.0) -> Environment:
    return Environment(
        name="source",
        beta=np.array([1.2, -0.9, 0.7, 1.0, -0.6]),
        gamma=np.array([0.8, -0.5]),
        a=a,
    )


def make_pair(family: str, magnitude: float = 0.75, shortcut_a: float = 0.8):
    """Source/target pair for a shift family. Only `covariate` moves P(X) on the causal block."""
    if family == "none":
        src = base_environment()
        return src, replace(src, name="target")

    if family == "covariate":
        src = base_environment()
        tgt = replace(
            src,
            name="target",
            mu_c=src.mu_c + magnitude,
            sd_c=src.sd_c * (1.0 + 0.3 * magnitude),
            mu_n=src.mu_n + 0.5 * magnitude,
        )
        return src, tgt

    if family == "concept":
        src = base_environment()
        beta = src.beta.copy()
        beta[[1, 2]] *= -1.0
        beta[4] += 1.4
        gamma = src.gamma * np.array([0.25, -1.0])
        return src, replace(src, name="target", beta=beta, gamma=gamma)

    if family == "shortcut":
        src = base_environment(a=shortcut_a)
        return src, replace(src, name="target", a=0.0)

    raise ValueError(f"unknown shift family: {family}")


def mechanism_changed(family: str) -> bool:
    return family in {"concept", "shortcut"}
