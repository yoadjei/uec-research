"""Model-agnostic explainers. All explain the logit, so they are on the same scale as the
gradient family and as the Bayes reference.

KernelSHAP and LIME are the expensive members and the ones with a real noise floor; both are
seeded per run so the floor is reproducible rather than incidental.
"""

import contextlib
import io

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

from ..models.mlp import logits


def logit_fn(model):
    def f(X):
        return logits(model, np.asarray(X, dtype=np.float64))

    return f


def kernel_shap(model, X, background=None, nsamples: int = 128, run: int = 0, **_):
    np.random.seed(7_000 + 31 * run)
    expl = shap.KernelExplainer(logit_fn(model), np.asarray(background), keep_index=False)
    with contextlib.redirect_stdout(io.StringIO()):
        vals = expl.shap_values(np.asarray(X), nsamples=nsamples, silent=True, l1_reg="num_features(20)")
    return np.asarray(vals, dtype=np.float64).reshape(len(X), -1)


def lime_tabular(model, X, background=None, num_samples: int = 1500, run: int = 0, **_):
    d = X.shape[1]
    expl = LimeTabularExplainer(
        np.asarray(background),
        mode="regression",
        discretize_continuous=False,
        random_state=5_000 + 31 * run,
    )
    f = logit_fn(model)
    out = np.zeros((len(X), d))
    for i, x in enumerate(np.asarray(X)):
        e = expl.explain_instance(x, f, num_features=d, num_samples=num_samples)
        for j, w in e.as_map()[1]:
            out[i, j] = w
    return out


def tree_shap(booster, X, **_):
    expl = shap.TreeExplainer(booster)
    return np.asarray(expl.shap_values(np.asarray(X)), dtype=np.float64).reshape(len(X), -1)
