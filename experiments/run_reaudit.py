"""Re-audit of Delta-Audit's published setting with a matched-operator null.

Delta-Audit (Hemmat & Fatemi, arXiv:2508.19589) audits 45 settings -- five classical learner
families x three UCI datasets x three A/B configuration pairs -- by differencing occlusion
attributions and flagging movement uncorrelated with behaviour change as "risky reliance
redistribution". Their setting has no distribution shift at all: A and B differ in
hyperparameters, on the same data.

The question they cannot answer is what their own procedure produces when *nothing* changes. The
matched null here is the same learner, the same hyperparameters, refit on a bootstrap resample of
the same training data -- an operator that changes nothing about the model specification and
nothing about the distribution. Whatever attribution movement that produces is the floor their
flagged movement has to clear.

We reimplement their attribution (occlusion in standardised space against a class-anchored margin,
averaged over mean and median baselines) and their reported metrics (JSD redistribution,
rank-overlap@10, behaviour-attribution coupling), so the comparison is on their terms.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.datasets import load_breast_cancer, load_digits, load_wine  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from uec.metrics.baselines import jensen_shannon  # noqa: E402
from uec.metrics.distances import d_l1, d_topk  # noqa: E402
from uec.metrics.normalise import l1_abs  # noqa: E402
from uec.paths import RESULTS  # noqa: E402
from uec.stats.inference import cliffs_delta, paired_test  # noqa: E402

DATASETS = {"breast_cancer": load_breast_cancer, "wine": load_wine, "digits": load_digits}

# Three A/B configuration pairs per family, in the spirit of the settings Delta-Audit reports.
FAMILIES = {
    "logreg": (LogisticRegression, dict(max_iter=2000), [
        ("C=1", {"C": 1.0}, "C=0.01", {"C": 0.01}),
        ("l2", {"penalty": "l2"}, "l2-weak", {"penalty": "l2", "C": 0.1}),
        ("unweighted", {"class_weight": None}, "balanced", {"class_weight": "balanced"}),
    ]),
    "svc": (SVC, dict(probability=False, decision_function_shape="ovr"), [
        ("poly", {"kernel": "poly"}, "rbf", {"kernel": "rbf"}),
        ("gamma=scale", {"gamma": "scale"}, "gamma=auto", {"gamma": "auto"}),
        ("C=1", {"C": 1.0}, "C=10", {"C": 10.0}),
    ]),
    "rf": (RandomForestClassifier, dict(n_estimators=200), [
        ("depth=None", {"max_depth": None}, "depth=3", {"max_depth": 3}),
        ("gini", {"criterion": "gini"}, "entropy", {"criterion": "entropy"}),
        ("sqrt", {"max_features": "sqrt"}, "log2", {"max_features": "log2"}),
    ]),
    "gb": (GradientBoostingClassifier, dict(), [
        ("depth=3", {"max_depth": 3}, "depth=6", {"max_depth": 6}),
        ("lr=0.1", {"learning_rate": 0.1}, "lr=0.01", {"learning_rate": 0.01}),
        ("n=100", {"n_estimators": 100}, "n=300", {"n_estimators": 300}),
    ]),
    "knn": (KNeighborsClassifier, dict(), [
        ("k=5", {"n_neighbors": 5}, "k=25", {"n_neighbors": 25}),
        ("uniform", {"weights": "uniform"}, "distance", {"weights": "distance"}),
        ("auto", {"algorithm": "auto"}, "brute", {"algorithm": "brute"}),
    ]),
}


def margin(model, X):
    """Class-anchored margin: decision value for the positive/predicted class."""
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return d if d.ndim == 1 else d.max(1) - np.sort(d, axis=1)[:, -2]
    p = model.predict_proba(X)
    q = np.sort(p, axis=1)
    return np.log(np.clip(q[:, -1], 1e-9, 1)) - np.log(np.clip(q[:, -2], 1e-9, 1))


def occlusion(model, X, baselines):
    """Delta-Audit's attribution: clamp each feature to a baseline and read the margin drop,
    averaged over the mean and median baselines."""
    base = margin(model, X)
    out = np.zeros((len(X), X.shape[1]))
    for b in baselines:
        for j in range(X.shape[1]):
            Xj = X.copy()
            Xj[:, j] = b[j]
            out[:, j] += base - margin(model, Xj)
    return out / len(baselines)


def fit(cls, common, overrides, X, y, seed):
    kw = dict(common)
    kw.update(overrides)
    if "random_state" in cls().get_params():
        kw["random_state"] = seed
    return cls(**kw).fit(X, y)


def audit_pair(A, B, X, baselines):
    """Delta-Audit's own reported quantities, computed on a pair of checkpoints."""
    a, b = occlusion(A, X, baselines), occlusion(B, X, baselines)
    move = d_l1(l1_abs(a), l1_abs(b))
    jsd = jensen_shannon(a, b)
    rank = 1.0 - d_topk(np.abs(a), np.abs(b), k=min(10, X.shape[1]))
    behave = np.abs(margin(A, X) - margin(B, X))
    bac = (float(np.corrcoef(move, behave)[0, 1])
           if move.std() > 1e-12 and behave.std() > 1e-12 else np.nan)
    return {"move": float(move.mean()), "jsd": float(jsd.mean()),
            "rank_overlap10": float(rank.mean()), "bac": bac,
            "behaviour": float(behave.mean())}, move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-probe", type=int, default=200)
    a = ap.parse_args()

    rows = []
    for ds_name, loader in DATASETS.items():
        data = loader()
        X_raw, y = data.data, data.target
        scaler = StandardScaler().fit(X_raw)
        X = scaler.transform(X_raw)
        baselines = [X.mean(0), np.median(X, axis=0)]

        for fam, (cls, common, pairs) in FAMILIES.items():
            for label_a, cfg_a, label_b, cfg_b in pairs:
                t0 = time.time()
                for seed in range(a.seeds):
                    rng = np.random.default_rng([seed, 7])
                    idx = rng.choice(len(X), len(X), replace=True)   # bootstrap resample
                    probe = X[rng.choice(len(X), min(a.n_probe, len(X)), replace=False)]

                    A = fit(cls, common, cfg_a, X, y, seed)
                    B = fit(cls, common, cfg_b, X, y, seed)            # their treatment
                    Anull = fit(cls, common, cfg_a, X[idx], y[idx], seed)  # matched null

                    treat, move_t = audit_pair(A, B, probe, baselines)
                    null, move_n = audit_pair(A, Anull, probe, baselines)

                    rows.append({
                        "dataset": ds_name, "family": fam, "pair": f"{label_a}->{label_b}",
                        "seed": seed,
                        **{f"treat_{k}": v for k, v in treat.items()},
                        **{f"null_{k}": v for k, v in null.items()},
                        "ratio_move": treat["move"] / null["move"] if null["move"] > 0 else np.nan,
                        "ratio_jsd": treat["jsd"] / null["jsd"] if null["jsd"] > 0 else np.nan,
                    })
                print(f"  {ds_name:14s} {fam:7s} {label_a}->{label_b:12s} "
                      f"({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "reaudit.parquet")

    # one verdict per setting: does the A/B movement clear the resample floor?
    verdicts = []
    for (ds, fam, pair), s in df.groupby(["dataset", "family", "pair"]):
        p = paired_test(s.treat_move.values, s.null_move.values)
        verdicts.append({
            "dataset": ds, "family": fam, "pair": pair,
            "treat_move": s.treat_move.mean(), "null_move": s.null_move.mean(),
            "ratio": s.ratio_move.mean(), "p": p,
            "cliffs_delta": cliffs_delta(s.treat_move.values, s.null_move.values),
            "clears_floor": bool(p < 0.05 and s.treat_move.mean() > s.null_move.mean()),
        })
    v = pd.DataFrame(verdicts)
    v.to_parquet(RESULTS / "reaudit_verdicts.parquet")

    print("\n" + v.round(4).to_string(index=False))
    n = len(v)
    print(f"\nsettings audited: {n}")
    print(f"movement clears the resample floor: {int(v.clears_floor.sum())} / {n} "
          f"({100 * v.clears_floor.mean():.0f}%)")
    print(f"movement BELOW the floor:           {int((v.ratio < 1).sum())} / {n}")
    print(f"median ratio to floor:              {v.ratio.median():.3f}")


if __name__ == "__main__":
    main()
