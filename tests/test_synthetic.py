import numpy as np
import pytest

from uec.data.groundtruth import (
    gt_attributions,
    gt_attributions_quadrature,
    omega_reference,
)
from uec.data.support import log_density_ratio, overlap_fraction, shared_support_probe
from uec.data.synthetic import D, NOISE, REDUNDANT, S0, SHORTCUT, make_pair
from uec.metrics.distances import DISTANCES, topk_with
from uec.metrics.normalise import l1_abs

FAMILIES = ["none", "covariate", "concept", "shortcut"]


@pytest.mark.parametrize("family", FAMILIES)
def test_bayes_logodds_matches_empirical(family):
    """The closed-form Bayes logit must reproduce the conditional label rate it claims to be."""
    rng = np.random.default_rng(0)
    for env in make_pair(family):
        X, y = env.sample(400_000, rng)
        h = env.bayes_logodds(X)
        edges = np.quantile(h, np.linspace(0, 1, 21))
        idx = np.clip(np.digitize(h, edges[1:-1]), 0, 19)
        for b in range(20):
            m = idx == b
            if m.sum() < 500:
                continue
            predicted = 1.0 / (1.0 + np.exp(-h[m].mean()))
            assert abs(y[m].mean() - predicted) < 0.025


@pytest.mark.parametrize("family", FAMILIES)
def test_ig_closed_form_matches_quadrature(family):
    rng = np.random.default_rng(1)
    for env in make_pair(family):
        X, _ = env.sample(64, rng)
        b = np.zeros(D)
        exact = gt_attributions(env, X, b)
        quad = gt_attributions_quadrature(env, X, b, steps=256)
        assert np.abs(exact - quad).max() < 1e-9


@pytest.mark.parametrize("family", FAMILIES)
def test_ig_completeness(family):
    rng = np.random.default_rng(2)
    for env in make_pair(family):
        X, _ = env.sample(256, rng)
        b = np.zeros(D)
        total = gt_attributions(env, X, b).sum(1)
        expected = env.bayes_logodds(X) - env.bayes_logodds(b[None, :])[0]
        assert np.abs(total - expected).max() < 1e-9


def test_gt_attributes_nothing_to_redundant_or_noise():
    """Y is independent of R and N given X_C, so the Bayes reference must ignore them."""
    rng = np.random.default_rng(3)
    src, _ = make_pair("covariate")
    X, _ = src.sample(128, rng)
    A = gt_attributions(src, X, np.zeros(D))
    assert np.abs(A[:, REDUNDANT]).max() == 0.0
    assert np.abs(A[:, NOISE]).max() == 0.0
    assert np.abs(A[:, SHORTCUT[1:]]).max() == 0.0


def test_omega_is_zero_under_covariate_shift_and_positive_otherwise():
    """Covariate shift leaves P(Y|X) fixed, so the Bayes reference is bitwise identical and
    omega vanishes; concept and shortcut shift move it. Distances are checked at machine
    epsilon because the identity holds on the attributions, not on the float reduction."""
    rng = np.random.default_rng(4)
    b = np.zeros(D)
    for family, expect_zero in [("none", True), ("covariate", True),
                                ("concept", False), ("shortcut", False)]:
        src, tgt = make_pair(family)
        X = shared_support_probe(src, tgt, 400, rng)
        same = np.array_equal(gt_attributions(src, X, b), gt_attributions(tgt, X, b))
        assert same is expect_zero

        for name, dist in DISTANCES.items():
            w = omega_reference(src, tgt, X, b, l1_abs, dist)
            if expect_zero:
                assert w.max() < 1e-12, f"{family}/{name} should have omega == 0"
            elif name != "topk":
                assert w.mean() > 0.0, f"{family}/{name} should have omega > 0"


def test_topk5_is_blind_to_concept_shift_by_construction():
    """The Bayes reference is supported on exactly the 5 causal features when a = 0, so top-5
    Jaccard cannot register a concept shift that only reweights them. k must be < the support
    size; topk3 sees it. This is why k is an ablation axis rather than a fixed choice."""
    rng = np.random.default_rng(9)
    src, tgt = make_pair("concept")
    X = shared_support_probe(src, tgt, 400, rng)
    b = np.zeros(D)
    assert omega_reference(src, tgt, X, b, l1_abs, DISTANCES["topk"]).max() == 0.0
    assert omega_reference(src, tgt, X, b, l1_abs, topk_with(3)).mean() > 0.0


def test_shortcut_reference_drops_s0_in_target():
    rng = np.random.default_rng(5)
    src, tgt = make_pair("shortcut")
    X, _ = src.sample(256, rng)
    b = np.zeros(D)
    assert np.abs(gt_attributions(src, X, b)[:, S0]).mean() > 0.1
    assert np.abs(gt_attributions(tgt, X, b)[:, S0]).max() == 0.0


def test_density_ratio_importance_identity():
    """E_{x~p_S}[p_T(x)/p_S(x)] == 1 validates both log-densities at once."""
    rng = np.random.default_rng(6)
    for family in FAMILIES:
        src, tgt = make_pair(family)
        X, _ = src.sample(300_000, rng)
        w = np.exp(np.clip(log_density_ratio(src, tgt, X), -30, 30))
        assert abs(w.mean() - 1.0) < 0.05, f"{family}: E[w] = {w.mean():.4f}"


@pytest.mark.parametrize("family", FAMILIES)
def test_shared_support_probe_respects_tau(family):
    rng = np.random.default_rng(7)
    src, tgt = make_pair(family)
    tau = 2.0
    X = shared_support_probe(src, tgt, 500, rng, tau=tau)
    assert X.shape == (500, D)
    assert np.abs(log_density_ratio(src, tgt, X)).max() <= tau


def test_covariate_shift_leaves_usable_overlap():
    rng = np.random.default_rng(8)
    src, tgt = make_pair("covariate")
    assert overlap_fraction(src, tgt, rng) > 0.10
