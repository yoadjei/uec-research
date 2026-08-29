import numpy as np
import pytest
import torch

from uec.data.synthetic import make_pair
from uec.explain.cache import attribute, probe_hash
from uec.explain.registry import EXPLAINERS
from uec.metrics.distances import d_spearman
from uec.metrics.normalise import l1_abs
from uec.models.mlp import logits
from uec.rng import pin_threads
from uec.train.harness import TrainConfig, train_source

pin_threads(1)


@pytest.fixture(scope="module")
def fitted():
    src, _ = make_pair("none")
    rng = np.random.default_rng(0)
    X, y = src.sample(3000, rng)
    model, _ = train_source(X, y, TrainConfig(epochs=25), seed=0)
    probe, _ = src.sample(64, rng)
    background, _ = src.sample(25, rng)
    return model, probe, background


def test_ig_completeness_on_a_trained_model(fitted):
    """Validates our IG wiring: sum of attributions must equal f(x) - f(baseline).

    Proposition 1(i) is an equality and E7 tests it per probe point, so completeness has to hold
    far below the size of the effect, not merely 'approximately'."""
    model, probe, _ = fitted
    b = np.zeros(probe.shape[1])
    a = attribute(model, probe, "integrated_gradients", use_cache=False, baseline=b, n_steps=512)
    expected = logits(model, probe) - logits(model, b[None, :])[0]
    assert np.abs(a.sum(1) - expected).max() < 1e-5


def test_ig_quadrature_order_depends_on_activation_smoothness():
    """A ReLU network is piecewise linear, so its gradient is a step function and Riemann
    quadrature for IG converges as O(1/n) rather than O(1/n^2). At 512 steps completeness holds
    only to ~5e-3 -- the same order as the quantity E7 tests. This is why the default activation
    is smooth, and why Proposition 1's C^1 premise is not a technicality."""
    src, _ = make_pair("none")
    rng = np.random.default_rng(0)
    X, y = src.sample(3000, rng)
    probe, _ = src.sample(32, rng)
    b = np.zeros(probe.shape[1])

    order = {}
    for act in ("silu", "relu"):
        model, _ = train_source(X, y, TrainConfig(epochs=25, activation=act), seed=0)
        expected = logits(model, probe) - logits(model, b[None, :])[0]
        err = [
            np.abs(
                attribute(model, probe, "integrated_gradients", use_cache=False,
                          baseline=b, n_steps=n).sum(1) - expected
            ).max()
            for n in (64, 512)
        ]
        order[act] = err[0] / err[1]

    assert order["silu"] > 30, "smooth activation should show ~O(1/n^2) convergence"
    assert order["relu"] < 20, "ReLU should show ~O(1/n) convergence"


def test_gradient_x_input_equals_input_times_saliency(fitted):
    model, probe, _ = fitted
    s = attribute(model, probe, "saliency", use_cache=False)
    gi = attribute(model, probe, "gradient_x_input", use_cache=False)
    assert np.allclose(gi, probe * s)


@pytest.mark.parametrize("name", ["saliency", "gradient_x_input", "integrated_gradients"])
def test_deterministic_explainers_reproduce_bitwise(fitted, name):
    """nu == 0 for these, which is what makes the noise floor meaningful for the others."""
    model, probe, _ = fitted
    a = attribute(model, probe, name, run=0, use_cache=False)
    b = attribute(model, probe, name, run=1, use_cache=False)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", ["smoothgrad", "expected_gradients", "kernel_shap"])
def test_stochastic_explainers_have_a_positive_noise_floor(fitted, name):
    model, probe, background = fitted
    small = probe[:24]
    a = attribute(model, small, name, run=0, use_cache=False, background=background)
    b = attribute(model, small, name, run=1, use_cache=False, background=background)
    assert not np.array_equal(a, b)
    assert d_spearman(l1_abs(a), l1_abs(b)).mean() > 0.0


@pytest.mark.parametrize("name", ["smoothgrad", "kernel_shap"])
def test_stochastic_explainers_are_reproducible_within_a_run(fitted, name):
    model, probe, background = fitted
    small = probe[:16]
    a = attribute(model, small, name, run=2, use_cache=False, background=background)
    b = attribute(model, small, name, run=2, use_cache=False, background=background)
    assert np.allclose(a, b)


def test_all_explainers_return_the_right_shape(fitted):
    model, probe, background = fitted
    small = probe[:12]
    for name in EXPLAINERS:
        a = attribute(model, small, name, use_cache=False, background=background)
        assert a.shape == small.shape, name
        assert np.isfinite(a).all(), name


def test_cache_key_tracks_checkpoint_weights(fitted):
    model, probe, _ = fitted
    a = attribute(model, probe[:8], "saliency", use_cache=True)
    with torch.no_grad():
        next(iter(model.parameters())).add_(0.05)
    b = attribute(model, probe[:8], "saliency", use_cache=True)
    with torch.no_grad():
        next(iter(model.parameters())).add_(-0.05)
    assert not np.array_equal(a, b)


def test_probe_hash_distinguishes_probes():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(10, 4)), rng.normal(size=(10, 4))
    assert probe_hash(a) != probe_hash(b)
    assert probe_hash(a) == probe_hash(a.copy())
