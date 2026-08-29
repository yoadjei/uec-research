"""Shared-support machinery for real data, tested where the answer is known.

For two Gaussians with a common covariance the Bayes domain classifier is exactly linear, so
logistic regression recovers the true log density ratio up to the class-prior offset. That makes
the estimated screen checkable against the closed form it is standing in for.
"""

import numpy as np
import pytest

from uec.data.folktables_data import (
    SourceScaler,
    domain_logit,
    mechanism_stability,
    probe_balance_auc,
    shared_support_probe,
)


def _gaussian_pair(n=20000, d=6, shift=0.8, seed=0):
    rng = np.random.default_rng(seed)
    Xs = rng.normal(0.0, 1.0, (n, d))
    Xt = rng.normal(shift, 1.0, (n, d))
    return Xs, Xt


def test_domain_logit_recovers_the_true_log_density_ratio():
    Xs, Xt = _gaussian_pair(n=20000, d=4, shift=0.8)
    est, auc = domain_logit(Xs, Xt, seed=0)

    # closed form for N(0,I) vs N(m,I): log p_T/p_S = m'x - ||m||^2/2
    m = np.full(4, 0.8)
    X = np.vstack([Xs, Xt])
    truth = X @ m - 0.5 * m @ m

    assert auc > 0.7
    assert np.corrcoef(est, truth)[0, 1] > 0.99
    assert abs(np.std(est) / np.std(truth) - 1.0) < 0.10


def test_shared_support_probe_is_near_chance_for_a_domain_classifier():
    Xs, Xt = _gaussian_pair(shift=0.8)
    probe, origin, auc = shared_support_probe(Xs, Xt, 800, np.random.default_rng(1), tau=0.3)
    assert len(probe) == 800
    assert auc > 0.7  # the domains are separable overall
    assert 0.35 < probe_balance_auc(probe, origin) < 0.65  # but not inside the probe


def test_probe_is_more_balanced_than_the_raw_pool():
    """The screen must actually do something: the unscreened pool should be separable."""
    Xs, Xt = _gaussian_pair(shift=1.2)
    _, pooled_auc = domain_logit(Xs, Xt, seed=2)
    probe, origin, _ = shared_support_probe(Xs, Xt, 800, np.random.default_rng(2), tau=0.3)
    assert pooled_auc > 0.85
    assert probe_balance_auc(probe, origin) < pooled_auc - 0.20


def test_tighter_tau_yields_a_tighter_probe():
    Xs, Xt = _gaussian_pair(shift=1.0)
    rng = np.random.default_rng(3)
    wide, _, _ = shared_support_probe(Xs, Xt, 500, rng, tau=1.5)
    tight, _, _ = shared_support_probe(Xs, Xt, 500, rng, tau=0.3)
    m = np.full(6, 1.0)
    spread = lambda P: np.std(P @ m)
    assert spread(tight) < spread(wide)


def test_source_scaler_is_frozen_to_the_source():
    Xs, Xt = _gaussian_pair(shift=2.0)
    sc = SourceScaler(Xs)
    assert np.abs(sc(Xs).mean(0)).max() < 0.05
    assert sc(Xt).mean(0).min() > 1.0  # the target is not re-centred


def test_mechanism_stability_is_small_when_conditional_is_stable():
    rng = np.random.default_rng(4)
    n = 8000
    p_source = rng.uniform(0.05, 0.95, n)
    y_source = (rng.uniform(size=n) < p_source).astype(int)
    p_target = rng.uniform(0.05, 0.95, n)
    y_target = (rng.uniform(size=n) < p_target).astype(int)
    assert mechanism_stability(y_source, p_source, y_target, p_target) < 0.05


def test_mechanism_stability_detects_a_shifted_conditional():
    rng = np.random.default_rng(5)
    n = 8000
    p_source = rng.uniform(0.05, 0.95, n)
    y_source = (rng.uniform(size=n) < p_source).astype(int)
    p_target = rng.uniform(0.05, 0.95, n)
    y_target = (rng.uniform(size=n) < np.clip(p_target + 0.25, 0, 1)).astype(int)
    assert mechanism_stability(y_source, p_source, y_target, p_target) > 0.15


@pytest.mark.parametrize("shift", [0.5, 1.0])
def test_probe_lies_between_the_domains(shift):
    Xs, Xt = _gaussian_pair(shift=shift)
    probe, _, _ = shared_support_probe(Xs, Xt, 500, np.random.default_rng(6), tau=0.5)
    m = probe.mean(0).mean()
    assert min(0.0, shift) - 0.4 < m < max(0.0, shift) + 0.4
