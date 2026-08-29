"""Content-addressed attribution cache.

The key includes the checkpoint weight hash, so editing a model can never serve a stale array —
the failure mode that would silently invalidate every downstream number.
"""

import hashlib

import numpy as np

from ..paths import RUNS, checkpoint_hash
from .registry import EXPLAINERS

CACHE = RUNS / "attributions"


def probe_hash(X: np.ndarray, n: int = 12) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(X, dtype=np.float64)).tobytes()).hexdigest()[:n]


def key(model_hash: str, explainer: str, probe: str, run: int) -> str:
    return f"{explainer}__{model_hash}__{probe}__r{run}"


def attribute(model, X, explainer: str, run: int = 0, use_cache: bool = True, **kw):
    e = EXPLAINERS[explainer]
    if not e.stochastic:
        run = 0

    mh = checkpoint_hash(model.state_dict()) if hasattr(model, "state_dict") else str(id(model))
    path = CACHE / f"{key(mh, explainer, probe_hash(X), run)}.npz"
    if use_cache and path.exists():
        return np.load(path)["a"]

    a = np.asarray(e(model, X, run=run, **kw), dtype=np.float64)
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, a=a)
    return a
