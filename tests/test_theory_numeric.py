"""Numeric verification of Propositions 1-3 in docs/theory.md.

Each proposition is checked against functions whose truth is available in closed form. A failure
here means the proof is wrong, not that the test is flaky.
"""

import itertools
import math

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

STEPS = 8192
QUAD_TOL = 1e-7  # midpoint-rule error for the smooth deltas used below


def integrated_gradients(fn, x, b, steps=STEPS):
    alpha = ((torch.arange(steps) + 0.5) / steps).unsqueeze(1)
    z = (b + alpha * (x - b)).requires_grad_(True)
    grad = torch.autograd.grad(fn(z).sum(), z)[0]
    return (x - b) * grad.mean(0)


def gradient_x_input(fn, x):
    z = x.clone().unsqueeze(0).requires_grad_(True)
    grad = torch.autograd.grad(fn(z).sum(), z)[0][0]
    return x * grad


def test_ig_aggregate_identity():
    """P1(i): sum_j dIG_j(x) == delta(x) - delta(b), exactly."""
    rng = np.random.default_rng(0)
    d, eps = 8, 0.1
    for _ in range(20):
        w = torch.tensor(rng.normal(0, 0.7, d))
        c = float(rng.normal())
        x = torch.tensor(rng.normal(0, 1, d))
        b = torch.tensor(rng.normal(0, 0.5, d))
        base = torch.tensor(rng.normal(0, 1, d))

        f = lambda z: (z @ base) + 0.3 * (z**2).sum(-1)
        delta = lambda z: eps * torch.sin(z @ w + c)
        f_prime = lambda z: f(z) - delta(z)

        d_ig = integrated_gradients(f, x, b) - integrated_gradients(f_prime, x, b)
        expected = delta(x.unsqueeze(0))[0] - delta(b.unsqueeze(0))[0]

        assert abs(d_ig.sum().item() - expected.item()) < QUAD_TOL
        assert abs(expected.item()) <= 2 * eps + 1e-12


@pytest.mark.parametrize("omega", [10.0, 50.0, 200.0, 1000.0])
def test_ig_components_unbounded(omega):
    """P1(iii): sup|delta| <= eps globally, yet |dIG_1| = eps*omega grows without bound."""
    d, eps = 4, 0.05
    x, b = torch.ones(d), torch.zeros(d)

    f = lambda z: (z**2).sum(-1)
    delta = lambda z: eps * torch.sin(omega * (z[..., 0] - z[..., 1]))
    f_prime = lambda z: f(z) - delta(z)

    d_ig = integrated_gradients(f, x, b) - integrated_gradients(f_prime, x, b)

    # delta is bounded by eps everywhere, not merely on the path
    probe = torch.tensor(np.random.default_rng(1).normal(0, 3, (2000, d)))
    assert delta(probe).abs().max().item() <= eps + 1e-12

    assert d_ig[0].item() == pytest.approx(eps * omega, rel=1e-6)
    assert d_ig[1].item() == pytest.approx(-eps * omega, rel=1e-6)
    assert abs(d_ig.sum().item()) < QUAD_TOL  # aggregate stays pinned, per P1(i)


@pytest.mark.parametrize("k", [1, 5, 25, 100])
def test_gradient_aggregate_unbounded(k):
    """P2: the aggregate gap is unbounded for gradient x input, unlike IG."""
    d, eps = 4, 0.05
    x = torch.full((d,), 1.3)
    omega = 2 * math.pi * k / x[0].item()  # cos(omega*x_1) = 1

    f = lambda z: (z**2).sum(-1)
    delta = lambda z: eps * torch.sin(omega * z[..., 0])
    f_prime = lambda z: f(z) - delta(z)

    d_gi = gradient_x_input(f, x) - gradient_x_input(f_prime, x)
    assert d_gi.sum().item() == pytest.approx(x[0].item() * eps * omega, rel=1e-6)

    # same premise bounds IG's aggregate to 2*eps
    b = torch.zeros(d)
    d_ig = integrated_gradients(f, x, b) - integrated_gradients(f_prime, x, b)
    assert abs(d_ig.sum().item()) <= 2 * eps + QUAD_TOL


def exact_shapley(values, d):
    """Shapley values by full enumeration; `values` maps a frozenset coalition to a payoff."""
    phi = np.zeros(d)
    for j in range(d):
        others = [i for i in range(d) if i != j]
        for r in range(d):
            for subset in itertools.combinations(others, r):
                s = frozenset(subset)
                w = math.factorial(r) * math.factorial(d - r - 1) / math.factorial(d)
                phi[j] += w * (values[s | {j}] - values[s])
    return phi


def test_shapley_coalition_bound():
    """P3: coalition-level preservation within eps bounds Shapley movement by 2*eps."""
    rng = np.random.default_rng(3)
    d, eps = 6, 0.02
    coalitions = [frozenset(c) for r in range(d + 1) for c in itertools.combinations(range(d), r)]

    for _ in range(50):
        v_f = {s: rng.normal(0, 1) for s in coalitions}
        v_g = {s: v_f[s] + rng.uniform(-eps, eps) for s in coalitions}
        gap = np.abs(exact_shapley(v_f, d) - exact_shapley(v_g, d)).max()
        assert gap <= 2 * eps + 1e-12


def test_shapley_bound_is_not_implied_by_pointwise_preservation():
    """The premise of P3 is coalition-level; agreeing on the grand coalition alone gives nothing."""
    rng = np.random.default_rng(4)
    d, eps = 6, 0.02
    coalitions = [frozenset(c) for r in range(d + 1) for c in itertools.combinations(range(d), r)]

    v_f = {s: rng.normal(0, 1) for s in coalitions}
    v_g = dict(v_f)
    full, empty = frozenset(range(d)), frozenset()
    for s in coalitions:
        if s not in (full, empty):
            v_g[s] = v_f[s] + rng.normal(0, 1.0)  # unconstrained off the endpoints

    assert abs(v_g[full] - v_f[full]) <= eps  # "prediction preserved"
    gap = np.abs(exact_shapley(v_f, d) - exact_shapley(v_g, d)).max()
    assert gap > 2 * eps  # yet Shapley values move freely
