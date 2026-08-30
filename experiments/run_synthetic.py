"""E0-E7 on the synthetic generator.

One row per (seed, family, explainer, distance, eps, feature_set). Source, null and seed-retrain
checkpoints depend only on the source environment, so they are built once per (seed, source env)
and reused across the families that share it.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.data.groundtruth import omega_reference  # noqa: E402
from uec.data.support import shared_support_probe  # noqa: E402
from uec.data.synthetic import D, make_pair  # noqa: E402
from uec.explain.cache import attribute  # noqa: E402
from uec.explain.registry import EXPENSIVE, EXPLAINERS  # noqa: E402
from uec.metrics.collinearity import reliable_features  # noqa: E402
from uec.metrics.distances import ABLATION_DISTANCES  # noqa: E402
from uec.metrics.normalise import NORMALISERS  # noqa: E402
from uec.metrics.uec import change, preserved_mask, summarise  # noqa: E402
from uec.models.mlp import accuracy, expected_calibration_error, probabilities  # noqa: E402
from uec.paths import RESULTS, append_registry, checkpoint_hash  # noqa: E402
from uec.rng import pin_threads  # noqa: E402
from uec.theory.bounds import coalition_epsilon, gradient_bound, ig_bound  # noqa: E402
from uec.train.harness import (  # noqa: E402
    TrainConfig,
    UpdateConfig,
    agreement_rate,
    build_checkpoints,
    operator_signature,
)

BASELINE = np.zeros(D)
FAMILIES = ["none", "covariate", "concept", "shortcut"]
EPS_GRID = [0.01, 0.02, 0.05, 0.10]


def explain_all(model, probe, probe_small, background, names, runs=(0, 1)):
    out = {}
    for name in names:
        X = probe_small if name in EXPENSIVE else probe
        for r in runs if EXPLAINERS[name].stochastic else (0,):
            out[(name, r)] = attribute(model, X, name, run=r, background=background)
        if not EXPLAINERS[name].stochastic:
            out[(name, 1)] = out[(name, 0)]
    return out


def run(seeds, families, names, n_source, n_update, n_probe, n_probe_small, cfg, ucfg,
        magnitude=1.5, theory=True):
    rows, per_point, theory_rows = [], {}, []
    distances = ABLATION_DISTANCES

    for seed in seeds:
        cache_by_env = {}
        for family in families:
            t0 = time.time()
            src, tgt = make_pair(family, magnitude=magnitude)
            key = (seed, src.a)

            if key not in cache_by_env:
                cache_by_env[key] = build_checkpoints(
                    src, tgt, seed, n_source, n_update, cfg, ucfg, regimes=("matched_null", "seed")
                )
            shared = cache_by_env[key]
            ck = dict(shared)
            ck.update(
                build_checkpoints(
                    src, tgt, seed, n_source, n_update, cfg, ucfg,
                    regimes=("treatment", "scratch"),
                )
            )

            assert operator_signature(ck["matched_null"][1]) == operator_signature(ck["treatment"][1])

            rng = np.random.default_rng(500 + seed)
            probe = shared_support_probe(src, tgt, n_probe, rng)
            probe_small = probe[:n_probe_small]
            background, _ = src.sample(25, rng)

            f0 = ck["source"][0]
            models = {k: v[0] for k, v in ck.items()}
            p0 = probabilities(f0, probe)
            p_treat = probabilities(models["treatment"], probe)
            p_null = probabilities(models["matched_null"], probe)

            Xa, ya = src.sample(4000, rng)
            Xb, yb = tgt.sample(4000, rng)
            diag = {
                "acc_source_src": accuracy(f0, Xa, ya),
                "acc_treat_tgt": accuracy(models["treatment"], Xb, yb),
                "acc_treat_src": accuracy(models["treatment"], Xa, ya),
                "acc_null_src": accuracy(models["matched_null"], Xa, ya),
                "ece_source": expected_calibration_error(f0, Xa, ya),
                "ece_treat": expected_calibration_error(models["treatment"], Xb, yb),
                "agree_treat": agreement_rate(f0, models["treatment"], probe),
                "agree_null": agreement_rate(f0, models["matched_null"], probe),
            }

            attrs = {
                k: explain_all(m, probe, probe_small, background, names)
                for k, m in models.items()
            }

            for name in names:
                X = probe_small if name in EXPENSIVE else probe
                n = len(X)
                pa, pb = p0[:n], p_treat[:n]
                pn = p_null[:n]

                A0 = attrs["source"][(name, 0)]
                A0b = attrs["source"][(name, 1)]
                reliable = reliable_features(A0)

                for feat_name, feats in (("all", None), ("reliable", reliable)):
                    if feats is not None and len(feats) < 3:
                        continue
                    for dname, dist in distances.items():
                        for phi_name, phi in NORMALISERS.items():
                            if phi_name == "signed" and dname != "spearman":
                                continue
                            raw = {
                                "delta": change(A0, attrs["treatment"][(name, 0)], phi, dist, feats),
                                "nu": change(A0, A0b, phi, dist, feats),
                                "rho_null": change(A0, attrs["matched_null"][(name, 0)], phi, dist, feats),
                                "rho_seed": change(A0, attrs["seed"][(name, 0)], phi, dist, feats),
                                "delta_scratch": change(
                                    A0, attrs["scratch"][(name, 0)], phi, dist, feats
                                ),
                            }
                            omega = omega_reference(src, tgt, X, BASELINE, phi, dist)

                            for eps in EPS_GRID:
                                m = preserved_mask(pa, pb, eps)
                                m_both = m & preserved_mask(pa, pn, eps)
                                if m.sum() < 10:
                                    continue
                                s = summarise(
                                    raw["delta"][m], omega[m], raw["nu"][m],
                                    raw["rho_null"][m], raw["rho_seed"][m], n_probe=n,
                                    seed=seed, family=family, explainer=name,
                                    explainer_family=EXPLAINERS[name].family,
                                    distance=dname, phi=phi_name, features=feat_name,
                                    eps=eps, magnitude=magnitude,
                                    update_epochs=ucfg.epochs, update_lr=ucfg.lr,
                                    n_matched=int(m_both.sum()),
                                    delta_matched=float(raw["delta"][m_both].mean())
                                    if m_both.sum() else np.nan,
                                    rho_null_matched=float(raw["rho_null"][m_both].mean())
                                    if m_both.sum() else np.nan,
                                    delta_scratch=float(raw["delta_scratch"][m].mean()),
                                    **diag,
                                )
                                rows.append(s.as_dict())

                                if (dname == "spearman" and eps == 0.05
                                        and phi_name == "abs" and feat_name == "all"):
                                    per_point[f"{seed}|{family}|{name}"] = np.stack([
                                        raw["delta"][m], raw["nu"][m],
                                        raw["rho_null"][m], raw["rho_seed"][m], omega[m],
                                    ])

            if theory:
                theory_rows += theory_check(
                    models, probe, probe_small, background, seed, family, p0, p_treat
                )

            for regime, (model, meta) in ck.items():
                append_registry({
                    "run_id": f"synthetic|{family}|{regime}|s{seed}",
                    "dataset": "synthetic", "shift": family, "regime": regime, "seed": seed,
                    "config_hash": "", "ckpt_hash": checkpoint_hash(model.state_dict()),
                    **{k: meta.get(k) for k in ("n_train", "n_steps", "lr", "epochs")},
                    "acc": diag["acc_source_src"], "path": "",
                })

            print(f"  seed {seed} {family:10s} {time.time() - t0:6.1f}s "
                  f"agree={diag['agree_treat']:.3f}", flush=True)

    return pd.DataFrame(rows), per_point, pd.DataFrame(theory_rows)


def theory_check(models, probe, probe_small, background, seed, family, p0, p_treat):
    f0, f1 = models["source"], models["treatment"]
    ig = ig_bound(f0, f1, probe, BASELINE, n_steps=256)
    gi = gradient_bound(f0, f1, probe, BASELINE)
    eps_coal, eps_data = coalition_epsilon(
        f0, f1, probe_small[:60], background, n_coalitions=48,
        rng=np.random.default_rng(seed),
    )
    preserved = preserved_mask(p0, p_treat, 0.05)
    return [{
        "seed": seed, "family": family,
        "ig_slack_max": float(np.nanmax(ig["slack"])),
        "ig_slack_median": float(np.nanmedian(ig["slack"])),
        "ig_quad_tol": ig["quad_tol"],
        "ig_residual_max": float(np.abs(ig["residual"]).max()),
        "ig_violations": int(np.nansum(ig["slack"] > 1.0 + ig["quad_tol"])),
        "gi_slack_median": float(np.nanmedian(gi["slack"])),
        "gi_slack_p90": float(np.nanpercentile(gi["slack"][np.isfinite(gi["slack"])], 90)),
        "gi_exceeds_one": float(np.nanmean(gi["slack"] > 1.0)),
        "gi_exceeds_one_preserved": float(np.nanmean(gi["slack"][preserved] > 1.0)),
        "coal_ratio_median": float(np.median(eps_coal / np.maximum(eps_data[:len(eps_coal)], 1e-12))),
        "eps_coal_median": float(np.median(eps_coal)),
        "eps_data_median": float(np.median(eps_data[:len(eps_coal)])),
        "n_probe": len(probe),
    }]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--explainers", nargs="+", default=list(EXPLAINERS))
    ap.add_argument("--n-source", type=int, default=8000)
    ap.add_argument("--n-update", type=int, default=4000)
    ap.add_argument("--n-probe", type=int, default=500)
    ap.add_argument("--n-probe-small", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--update-epochs", type=int, default=2)
    ap.add_argument("--update-lr", type=float, default=2e-4)
    ap.add_argument("--magnitude", type=float, default=1.5)
    ap.add_argument("--tag", default="synthetic")
    ap.add_argument("--no-theory", action="store_true")
    a = ap.parse_args()

    pin_threads(4)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df, per_point, theory = run(
        range(a.seeds), a.families, a.explainers, a.n_source, a.n_update,
        a.n_probe, a.n_probe_small,
        TrainConfig(epochs=a.epochs), UpdateConfig(lr=a.update_lr, epochs=a.update_epochs),
        magnitude=a.magnitude,
        theory=not a.no_theory,
    )

    df.to_parquet(RESULTS / f"{a.tag}_metrics.parquet")
    np.savez_compressed(RESULTS / f"{a.tag}_perpoint.npz", **per_point)
    if len(theory):
        theory.to_parquet(RESULTS / f"{a.tag}_theory.parquet")
    print(f"\nwrote {len(df)} rows to {RESULTS / f'{a.tag}_metrics.parquet'}")


if __name__ == "__main__":
    main()
