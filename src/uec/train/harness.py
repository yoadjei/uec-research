"""Training and update operators.

The design constraint that matters here is the matched-operator null (docs/spec.md D2): the control
for a fine-tune-on-target treatment is the *same* fine-tune applied to a fresh draw from the source
distribution, not a from-scratch retrain. `build_checkpoints` constructs both from one operator
specification so they cannot drift apart, and `operator_signature` is what the guard test compares.
"""

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

from ..models.mlp import MLP, accuracy
from ..rng import seed_everything


@dataclass(frozen=True)
class TrainConfig:
    hidden: tuple = (64, 64)
    activation: str = "silu"
    lr: float = 2e-3
    epochs: int = 60
    batch_size: int = 256
    weight_decay: float = 1e-5


@dataclass(frozen=True)
class UpdateConfig:
    lr: float = 5e-4
    epochs: int = 20
    batch_size: int = 256
    weight_decay: float = 1e-5
    freeze_layers: int = 0


def _fit(model, X, y, lr, epochs, batch_size, weight_decay, seed, freeze_layers=0):
    seed_everything(seed)
    params = list(model.parameters())
    if freeze_layers:
        for p in params[: 2 * freeze_layers]:
            p.requires_grad_(False)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)

    model.train()
    steps = 0
    for _ in range(epochs):
        order = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
            steps += 1

    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return steps


def train_source(X, y, cfg: TrainConfig, seed: int):
    seed_everything(seed)
    model = MLP(X.shape[1], cfg.hidden, cfg.activation)
    steps = _fit(model, X, y, cfg.lr, cfg.epochs, cfg.batch_size, cfg.weight_decay, seed)
    return model, {"n_train": len(X), "n_steps": steps, "lr": cfg.lr, "epochs": cfg.epochs}


def update(model, X, y, ucfg: UpdateConfig, seed: int):
    child = copy.deepcopy(model)
    steps = _fit(
        child, X, y, ucfg.lr, ucfg.epochs, ucfg.batch_size, ucfg.weight_decay, seed,
        ucfg.freeze_layers,
    )
    return child, {"n_train": len(X), "n_steps": steps, "lr": ucfg.lr, "epochs": ucfg.epochs}


def operator_signature(meta: dict) -> tuple:
    return (meta["n_train"], meta["n_steps"], meta["lr"], meta["epochs"])


STREAM = {"source": 0, "matched_null": 1, "treatment": 2, "scratch": 4}


def _stream(seed: int, tag: str) -> np.random.Generator:
    """Named streams, not a shared sequential generator: regimes must be constructible in any
    order and in any subset, otherwise a caller that requests only some of them silently shifts
    the draws of the rest."""
    return np.random.default_rng([seed, STREAM[tag]])


def build_checkpoints(src_env, tgt_env, seed, n_source, n_update, cfg, ucfg, regimes=None):
    """Source model plus every control and treatment checkpoint for one seed.

    `null` and `treatment` are trained by the same operator on the same number of points; only the
    sampling distribution differs. `seed` reuses the source model's own training data with a
    different initialisation, which is the Rashomon floor as EvoXplain and Laberge et al. measure
    it -- resampling the data as well would confound seed variation with sampling variation.
    """
    regimes = regimes or ("matched_null", "seed", "treatment", "scratch")

    Xs, ys = src_env.sample(n_source, _stream(seed, "source"))
    f_source, meta_source = train_source(Xs, ys, cfg, seed)
    out = {"source": (f_source, meta_source)}

    if "matched_null" in regimes:
        Xn, yn = src_env.sample(n_update, _stream(seed, "matched_null"))
        out["matched_null"] = update(f_source, Xn, yn, ucfg, seed + 1)

    if "treatment" in regimes:
        Xt, yt = tgt_env.sample(n_update, _stream(seed, "treatment"))
        out["treatment"] = update(f_source, Xt, yt, ucfg, seed + 1)

    if "seed" in regimes:
        out["seed"] = train_source(Xs, ys, cfg, seed + 5000)

    if "scratch" in regimes:
        Xt2, yt2 = tgt_env.sample(n_source, _stream(seed, "scratch"))
        out["scratch"] = train_source(Xt2, yt2, cfg, seed + 7000)

    return out


def with_magnitude(cfg: UpdateConfig, **kw) -> UpdateConfig:
    return replace(cfg, **kw)


def agreement_rate(model_a, model_b, X, threshold: float = 0.5) -> float:
    from ..models.mlp import probabilities

    return float(
        ((probabilities(model_a, X) >= threshold) == (probabilities(model_b, X) >= threshold)).mean()
    )


__all__ = [
    "TrainConfig", "UpdateConfig", "train_source", "update", "build_checkpoints",
    "operator_signature", "agreement_rate", "accuracy", "with_magnitude",
]
