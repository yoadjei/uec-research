"""Gradient-boosted trees, to test whether the phenomenon is specific to differentiable models.

Trees are not fine-tuned, so the update operator here is a full retrain. That makes the matched
null exact rather than approximate: source, null and treatment are the *same* training procedure on
the same number of points, and only the sampling distribution differs. TreeSHAP is exact, so this
setting also has no explainer noise floor to subtract.
"""

import numpy as np
import xgboost as xgb


def default_params(seed: int = 0) -> dict:
    return {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": 4,
        "tree_method": "hist",
        "eval_metric": "logloss",
    }


def train(X, y, seed: int = 0, **overrides):
    model = xgb.XGBClassifier(**{**default_params(seed), **overrides})
    model.fit(X, y)
    return model, {"n_train": len(X), "n_steps": model.n_estimators,
                   "lr": model.learning_rate, "epochs": model.n_estimators}


def logits(model, X) -> np.ndarray:
    return model.predict(X, output_margin=True).astype(np.float64)


def probabilities(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float64)


def accuracy(model, X, y) -> float:
    return float((probabilities(model, X) >= 0.5).astype(int).__eq__(y).mean())


def build_checkpoints(src_env, tgt_env, seed, n_source, n_update, **overrides):
    """Retrain-only analogue of the MLP harness, with the same matched-null contract.

    `matched_null` and `treatment` train on the same number of fresh points with identical
    hyperparameters; only the distribution differs.
    """
    Xs, ys = src_env.sample(n_source, np.random.default_rng([seed, 0]))
    Xn, yn = src_env.sample(n_update, np.random.default_rng([seed, 1]))
    Xt, yt = tgt_env.sample(n_update, np.random.default_rng([seed, 2]))

    out = {
        "source": train(Xs, ys, seed, **overrides),
        "matched_null": train(Xn, yn, seed, **overrides),
        "treatment": train(Xt, yt, seed, **overrides),
        "seed": train(Xs, ys, seed + 5000, **overrides),
    }
    assert out["matched_null"][1]["n_train"] == out["treatment"][1]["n_train"]
    return out
