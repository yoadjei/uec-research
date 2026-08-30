"""Real-data demonstration on ACS Income.

Legitimacy claims here are restricted to covariate-style shifts where omega = 0 is the defensible
null, and even then the null is only *checked*, not proved, via `mechanism_stability`. Nothing in
this script labels a change warranted.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.data.folktables_data import (  # noqa: E402
    SourceScaler,
    load_acs,
    mechanism_stability,
    probe_balance_auc,
    shared_support_probe,
)
from uec.explain.cache import attribute  # noqa: E402
from uec.explain.registry import EXPENSIVE, EXPLAINERS  # noqa: E402
from uec.metrics.collinearity import reliable_features  # noqa: E402
from uec.metrics.distances import DISTANCES  # noqa: E402
from uec.metrics.normalise import NORMALISERS  # noqa: E402
from uec.metrics.uec import change, preserved_mask, summarise  # noqa: E402
from uec.models.mlp import accuracy, expected_calibration_error, probabilities  # noqa: E402
from uec.paths import RESULTS  # noqa: E402
from uec.rng import pin_threads  # noqa: E402
from uec.train.harness import TrainConfig, UpdateConfig, agreement_rate, train_source, update  # noqa: E402

EPS_GRID = [0.01, 0.02, 0.05, 0.10]


def matched_update_size(Xs, Xt, n_source, requested):
    """The update size the *smaller* domain can supply.

    Small target states (SD has 4899 rows) cannot supply the requested update set. Silently
    shrinking only the treatment would break the matched-operator guarantee -- the null would train
    on more data and more steps than the treatment, and the ratio would measure that instead of the
    shift. Both arms use this size.
    """
    return int(min(requested, len(Xs) - n_source, len(Xt)))


def build(Xs, ys, Xt, yt, seed, n_source, n_update, cfg, ucfg):
    """Same operator for the null and the treatment; only the sampling domain differs."""
    rs = np.random.default_rng([seed, 0])
    i_src = rs.choice(len(Xs), n_source, replace=False)
    f0, meta0 = train_source(Xs[i_src], ys[i_src], cfg, seed)

    rn = np.random.default_rng([seed, 1])
    rest = np.setdiff1d(np.arange(len(Xs)), i_src)
    i_null = rn.choice(rest, n_update, replace=False)
    f_null, meta_n = update(f0, Xs[i_null], ys[i_null], ucfg, seed + 1)

    rt = np.random.default_rng([seed, 2])
    i_tgt = rt.choice(len(Xt), n_update, replace=False)
    f_treat, meta_t = update(f0, Xt[i_tgt], yt[i_tgt], ucfg, seed + 1)

    f_seed, _ = train_source(Xs[i_src], ys[i_src], cfg, seed + 5000)
    assert meta_n["n_steps"] == meta_t["n_steps"] and meta_n["n_train"] == meta_t["n_train"]
    return {"source": f0, "matched_null": f_null, "treatment": f_treat, "seed": f_seed}, meta0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="CA")
    ap.add_argument("--targets", nargs="+", default=["MI", "MS", "SD", "PR"])
    ap.add_argument("--year", default="2018")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-source", type=int, default=20000)
    ap.add_argument("--n-update", type=int, default=4000)
    ap.add_argument("--n-probe", type=int, default=800)
    ap.add_argument("--n-probe-small", type=int, default=120)
    ap.add_argument("--update-epochs", type=int, nargs="+", default=[2, 20])
    ap.add_argument("--update-lr", type=float, default=2e-4)
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--explainers", nargs="+", default=list(EXPLAINERS))
    a = ap.parse_args()

    pin_threads(4)
    cfg = TrainConfig(epochs=40)
    src = load_acs(a.source, a.year)
    scaler = SourceScaler(src.X)
    Xs = scaler(src.X)
    rows = []

    for tgt_state in a.targets:
        tgt = load_acs(tgt_state, a.year)
        Xt = scaler(tgt.X)
        pr = shared_support_probe(
            Xs, Xt, a.n_probe, np.random.default_rng(0), tau=a.tau, ys=src.y, yt=tgt.y
        )
        probe, origin, auc = pr.X, pr.origin, pr.domain_auc
        balance = probe_balance_auc(probe, origin)
        print(f"\n{a.source}->{tgt_state}: domain auc={auc:.3f} probe balance={balance:.3f} "
              f"n_src={len(Xs)} n_tgt={len(Xt)}", flush=True)

        n_update = matched_update_size(Xs, Xt, a.n_source, a.n_update)
        print(f"  matched update size: {n_update}", flush=True)

        for seed in range(a.seeds):
            for ep in a.update_epochs:
                t0 = time.time()
                ucfg = UpdateConfig(lr=a.update_lr, epochs=ep)
                ck, _ = build(Xs, src.y, Xt, tgt.y, seed, a.n_source, n_update, cfg, ucfg)
                f0 = ck["source"]

                p0 = probabilities(f0, probe)
                pt = probabilities(ck["treatment"], probe)
                m_s, m_t = origin == 0, origin == 1
                mech = (
                    mechanism_stability(pr.y[m_s], p0[m_s], pr.y[m_t], p0[m_t])
                    if m_s.sum() > 200 and m_t.sum() > 200
                    else np.nan
                )

                diag = {
                    "target": tgt_state, "update_epochs": ep, "domain_auc": auc,
                    "n_update": n_update,
                    "probe_balance": balance, "mech_gap": mech,
                    "acc_src": accuracy(f0, Xs[:20000], src.y[:20000]),
                    "acc_src_on_tgt": accuracy(f0, Xt[:20000], tgt.y[:20000]),
                    "acc_treat_tgt": accuracy(ck["treatment"], Xt[:20000], tgt.y[:20000]),
                    "ece_treat": expected_calibration_error(ck["treatment"], Xt[:20000], tgt.y[:20000]),
                    "agree_treat": agreement_rate(f0, ck["treatment"], probe),
                    "agree_null": agreement_rate(f0, ck["matched_null"], probe),
                }

                background = probe[:50]
                for name in a.explainers:
                    X = probe[: a.n_probe_small] if name in EXPENSIVE else probe
                    n = len(X)
                    A = {k: attribute(m, X, name, background=background) for k, m in ck.items()}
                    A0b = attribute(ck["source"], X, name, run=1, background=background)
                    feats_reliable = reliable_features(A["source"])

                    for feat_name, feats in (("all", None), ("reliable", feats_reliable)):
                        if feats is not None and len(feats) < 3:
                            continue
                        for dname, dist in DISTANCES.items():
                            phi = NORMALISERS["abs"]
                            raw = {
                                k: change(A["source"], A[k], phi, dist, feats)
                                for k in ("treatment", "matched_null", "seed")
                            }
                            nu = change(A["source"], A0b, phi, dist, feats)
                            for eps in EPS_GRID:
                                mask = preserved_mask(p0[:n], pt[:n], eps)
                                if mask.sum() < 10:
                                    continue
                                s = summarise(
                                    raw["treatment"][mask], np.zeros(int(mask.sum())),
                                    nu[mask] if EXPLAINERS[name].stochastic else np.zeros(0),
                                    raw["matched_null"][mask], raw["seed"][mask], n_probe=n,
                                    seed=seed, family="covariate_state", explainer=name,
                                    explainer_family=EXPLAINERS[name].family,
                                    distance=dname, phi="abs", features=feat_name, eps=eps,
                                    **diag,
                                )
                                rows.append(s.as_dict())
                print(f"  s{seed} ep={ep:3d} {tgt_state} agree={diag['agree_treat']:.3f} "
                      f"({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "folktables_metrics.parquet")
    print(f"\nwrote {len(df)} rows")


if __name__ == "__main__":
    main()
