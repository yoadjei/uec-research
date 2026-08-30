import numpy as np
import pytest

from uec.data.groundtruth import gt_attributions
from uec.data.support import shared_support_probe
from uec.data.synthetic import CAUSAL, D, REDUNDANT, make_pair
from uec.metrics.collinearity import ranking_z, reliable_features, unreliable_pairs
from uec.metrics.distances import DISTANCES, d_spearman
from uec.metrics.normalise import l1_abs, l1_signed
from uec.metrics.uec import change, preserved_mask, summarise
from uec.stats.inference import bootstrap_ci, cliffs_delta, holm, ratio_ci

ZERO = np.zeros(0)


def _summary(delta, omega=None, nu=None, rho_null=None, rho_seed=None, n=100):
    z = np.zeros(len(delta))
    return summarise(
        delta,
        z if omega is None else omega,
        ZERO if nu is None else nu,
        z + 0.1 if rho_null is None else rho_null,
        z + 0.1 if rho_seed is None else rho_seed,
        n_probe=n,
    )


def test_distance_identity_symmetry_and_bounds():
    rng = np.random.default_rng(0)
    A = np.abs(rng.normal(size=(50, D)))
    B = np.abs(rng.normal(size=(50, D)))
    for name, dist in DISTANCES.items():
        assert np.allclose(dist(l1_abs(A), l1_abs(A)), 0.0), name
        d = dist(l1_abs(A), l1_abs(B))
        assert np.allclose(d, dist(l1_abs(B), l1_abs(A))), name
        assert d.min() >= -1e-12 and d.max() <= 1.0 + 1e-12, name


def test_distances_are_invariant_to_positive_rescaling():
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(40, D)), rng.normal(size=(40, D))
    for name, dist in DISTANCES.items():
        base = dist(l1_abs(A), l1_abs(B))
        scaled = dist(l1_abs(A * 17.0), l1_abs(B * 0.003))
        assert np.allclose(base, scaled), name


def test_uec_is_not_positive_without_change():
    d = _summary(np.zeros(80))
    assert d.delta == 0.0
    assert d.uec <= 0.0


def test_uec_recovers_planted_omega():
    """Validity test: when the model moves by exactly the warranted amount, UEC must read zero.

    This is the property that separates the estimator from every prior magnitude metric -- those
    would flag the concept-shift movement as instability."""
    rng = np.random.default_rng(2)
    src, tgt = make_pair("concept")
    X = shared_support_probe(src, tgt, 500, rng)
    b = np.zeros(D)

    A = gt_attributions(src, X, b)
    B = gt_attributions(tgt, X, b)
    delta = change(A, B, l1_abs, d_spearman)
    omega = delta.copy()

    s = summarise(delta, omega, ZERO, np.zeros(len(X)), np.zeros(len(X)), n_probe=len(X))
    assert s.delta > 0.0
    assert abs(s.uec) < 1e-12


def test_exceedance_is_scale_free():
    rng = np.random.default_rng(3)
    delta = rng.uniform(0, 0.5, 200)
    nu = rng.uniform(0, 0.05, 200)
    rho = rng.uniform(0, 0.1, 200)
    a = summarise(delta, np.zeros(200), nu, rho, rho, n_probe=200)
    b = summarise(delta, np.zeros(200), nu, rho, rho, n_probe=200)
    assert a.exceedance == b.exceedance
    assert 0.0 <= a.exceedance <= 1.0


def test_ratio_uses_the_matched_null():
    delta = np.full(50, 0.30)
    s = summarise(delta, np.zeros(50), ZERO, np.full(50, 0.10), np.full(50, 0.20), n_probe=50)
    assert s.ratio == pytest.approx(3.0)
    assert s.ratio_seed == pytest.approx(1.5)


def test_preserved_mask_selects_on_probability_gap():
    p = np.array([0.10, 0.50, 0.90])
    q = np.array([0.11, 0.70, 0.90])
    assert preserved_mask(p, q, 0.05).tolist() == [True, False, True]


def test_signed_and_abs_normalisers_agree_on_positive_input():
    A = np.abs(np.random.default_rng(4).normal(size=(10, D)))
    assert np.allclose(l1_abs(A), l1_signed(A))


def test_redundant_features_are_flagged_unreliable():
    """R is a noisy copy of C, so their ranking is a coin flip by the Attribution Impossibility.
    The partition must catch that without being told."""
    rng = np.random.default_rng(5)
    n = 400
    A = np.zeros((n, D))
    A[:, CAUSAL[0]] = rng.normal(1.0, 0.3, n)
    A[:, REDUNDANT[0]] = A[:, CAUSAL[0]] + rng.normal(0, 0.02, n)
    A[:, CAUSAL[1]] = rng.normal(5.0, 0.3, n)

    pairs = unreliable_pairs(A)
    assert (CAUSAL[0], REDUNDANT[0]) in pairs
    assert CAUSAL[1] in reliable_features(A)
    assert abs(ranking_z(A)[CAUSAL[1], CAUSAL[0]]) > 1.96


def test_cliffs_delta_endpoints_and_ties():
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == pytest.approx(-1.0)
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_holm_is_monotone_and_conservative():
    p = np.array([0.001, 0.02, 0.04, 0.5])
    adj = holm(p)
    assert np.all(adj >= p)
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)


def test_bootstrap_ci_covers_the_mean():
    rng = np.random.default_rng(6)
    v = rng.normal(3.0, 0.5, 40)
    m, lo, hi = bootstrap_ci(v, n_boot=2000)
    assert lo < m < hi
    assert lo < 3.0 < hi


def test_ratio_ci_is_paired():
    rng = np.random.default_rng(7)
    num = rng.normal(0.30, 0.02, 10)
    den = rng.normal(0.10, 0.01, 10)
    r, lo, hi = ratio_ci(num, den, n_boot=2000)
    assert lo < r < hi
    assert lo > 1.0


def test_registry_regime_survives_a_csv_round_trip(tmp_path):
    """`null` is in pandas' default NA list: a regime by that name reads back as NaN and every
    matched-null row silently disappears for anyone reproducing from the registry."""
    import pandas as pd

    from uec.train.harness import STREAM

    assert "null" not in STREAM, "regime label 'null' round-trips as NaN"

    path = tmp_path / "registry.csv"
    pd.DataFrame({"regime": list(STREAM) + ["seed"]}).to_csv(path, index=False)
    back = pd.read_csv(path)
    assert back.regime.isna().sum() == 0
    assert set(back.regime) == set(STREAM) | {"seed"}
