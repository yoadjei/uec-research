import numpy as np
import pytest
import torch

from uec.data.support import shared_support_probe
from uec.data.synthetic import make_pair
from uec.models.mlp import accuracy, probabilities
from uec.paths import checkpoint_hash
from uec.rng import pin_threads
from uec.train.harness import (
    TrainConfig,
    UpdateConfig,
    agreement_rate,
    build_checkpoints,
    operator_signature,
    train_source,
)

pin_threads(1)

CFG = TrainConfig(epochs=25)
UCFG = UpdateConfig(epochs=10)
N_SOURCE, N_UPDATE = 4000, 2000


def _checkpoints(family, seed=0):
    src, tgt = make_pair(family)
    return src, tgt, build_checkpoints(src, tgt, seed, N_SOURCE, N_UPDATE, CFG, UCFG)


def test_training_is_deterministic():
    src, _ = make_pair("none")
    X, y = src.sample(2000, np.random.default_rng(0))
    a, _ = train_source(X, y, CFG, seed=3)
    b, _ = train_source(X, y, CFG, seed=3)
    assert checkpoint_hash(a.state_dict()) == checkpoint_hash(b.state_dict())


def test_different_seed_gives_different_weights():
    src, _ = make_pair("none")
    X, y = src.sample(2000, np.random.default_rng(0))
    a, _ = train_source(X, y, CFG, seed=3)
    b, _ = train_source(X, y, CFG, seed=4)
    assert checkpoint_hash(a.state_dict()) != checkpoint_hash(b.state_dict())


def test_matched_null_shares_the_operator_with_the_treatment():
    """Guards spec D2: the null and the treatment differ only in the sampling distribution."""
    _, _, ck = _checkpoints("covariate")
    assert operator_signature(ck["null"][1]) == operator_signature(ck["treatment"][1])


def test_source_model_learns():
    src, _ = make_pair("none")
    rng = np.random.default_rng(1)
    X, y = src.sample(N_SOURCE, rng)
    model, _ = train_source(X, y, CFG, seed=0)
    Xte, yte = src.sample(4000, rng)
    assert accuracy(model, Xte, yte) > 0.75


@pytest.mark.parametrize("family", ["covariate", "concept", "shortcut"])
def test_prediction_agreement_on_shared_support(family):
    """Covariate-shift updates must preserve predictions on the probe; a mechanism-changing shift
    need not, and that asymmetry is the point of the design."""
    src, tgt, ck = _checkpoints(family)
    probe = shared_support_probe(src, tgt, 800, np.random.default_rng(2))
    rate = agreement_rate(ck["source"][0], ck["treatment"][0], probe)
    if family == "covariate":
        assert rate >= 0.90, f"covariate agreement {rate:.3f} below 0.90"
    else:
        assert rate >= 0.50


def test_frozen_model_has_no_instance_level_change():
    """The category error, as a test: with a frozen model and a deterministic explainer the
    instance-level attribution difference is identically zero whatever the input distribution."""
    src, tgt, ck = _checkpoints("covariate")
    model = ck["source"][0]
    probe_s = shared_support_probe(src, tgt, 200, np.random.default_rng(3))
    p1 = probabilities(model, probe_s)
    p2 = probabilities(model, probe_s)
    assert np.array_equal(p1, p2)

    def grads(X):
        x = torch.as_tensor(X, dtype=torch.float32).requires_grad_(True)
        return torch.autograd.grad(model(x).sum(), x)[0].numpy()

    assert np.array_equal(grads(probe_s), grads(probe_s))

    # the same holds on target-distributed inputs: changing P(X) moves which x are seen,
    # never the value of the explanation at a fixed x
    probe_t, _ = tgt.sample(200, np.random.default_rng(4))
    assert np.array_equal(grads(probe_t), grads(probe_t))
