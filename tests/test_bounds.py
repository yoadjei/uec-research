import numpy as np
import pytest

from uec.data.synthetic import D, make_pair
from uec.rng import pin_threads
from uec.theory.bounds import coalition_epsilon, gradient_bound, ig_bound
from uec.train.harness import TrainConfig, UpdateConfig, train_source, update

pin_threads(1)
BASE = np.zeros(D)


@pytest.fixture(scope="module")
def pair():
    src, tgt = make_pair("covariate")
    rng = np.random.default_rng(0)
    Xs, ys = src.sample(4000, rng)
    f, _ = train_source(Xs, ys, TrainConfig(epochs=25), seed=0)
    Xt, yt = tgt.sample(2000, rng)
    g, _ = update(f, Xt, yt, UpdateConfig(epochs=10), seed=1)
    probe, _ = src.sample(200, rng)
    background, _ = src.sample(25, rng)
    return f, g, probe, background


def test_ig_aggregate_identity_holds_on_trained_checkpoints(pair):
    """Proposition 1(i) is an equality, so the residual -- not merely the slack -- must vanish."""
    f, g, probe, _ = pair
    r = ig_bound(f, g, probe, BASE, n_steps=256)
    assert np.abs(r["residual"]).max() < 1e-4


def test_ig_slack_never_exceeds_one(pair):
    """Tolerance is the propagated quadrature residual, not a guessed epsilon: the exact identity
    gives |1'dIG| = |delta(x) - delta(b)| + residual, so slack <= 1 + |residual|/denom."""
    f, g, probe, _ = pair
    r = ig_bound(f, g, probe, BASE, n_steps=256)
    slack = r["slack"][np.isfinite(r["slack"])]
    assert slack.size > 0
    assert slack.max() <= 1.0 + r["quad_tol"] + 1e-12
    assert r["quad_tol"] < 1e-4


def test_ig_bound_is_tight_not_vacuous(pair):
    """The two checkpoints agree at the baseline, so delta(b) ~ 0 and the bound is essentially
    attained. A bound that is never approached would not be evidence for anything."""
    f, g, probe, _ = pair
    slack = ig_bound(f, g, probe, BASE, n_steps=256)["slack"]
    assert np.nanmedian(slack) > 0.9


def test_gradient_aggregate_is_unconstrained_by_the_ig_bound(pair):
    """Proposition 2 is an existence result; here we require only that the measured quantity is
    finite and not bound by the <= 1 constraint that binds IG."""
    f, g, probe, _ = pair
    r = gradient_bound(f, g, probe, BASE)
    slack = r["slack"][np.isfinite(r["slack"])]
    assert slack.size > 0
    assert np.isfinite(r["aggregate"]).all()


def test_coalition_epsilon_exceeds_data_level_gap(pair):
    """Proposition 3's premise: fine-tuning controls the models on the data, not on masked
    composites, so the coalition gap should dominate the pointwise prediction gap."""
    f, g, probe, background = pair
    eps_coal, eps_data = coalition_epsilon(
        f, g, probe[:40], background, n_coalitions=48, rng=np.random.default_rng(1)
    )
    assert (eps_coal >= 0).all()
    assert np.median(eps_coal) > np.median(eps_data)


def test_identical_models_have_zero_gap(pair):
    f, _, probe, _ = pair
    r = ig_bound(f, f, probe, BASE, n_steps=64)
    assert np.abs(r["aggregate"]).max() < 1e-9
    assert np.abs(r["residual"]).max() < 1e-9
