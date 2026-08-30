"""Is the phenomenon specific to differentiable models?

Gradient-boosted trees with exact TreeSHAP. Trees are retrained rather than fine-tuned, so the
matched null here is exact: identical hyperparameters and sample size for the null and the
treatment, differing only in the sampling distribution. TreeSHAP is exact, so nu = 0 by
construction and the comparison is against the matched null alone.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.data.groundtruth import gt_attributions  # noqa: E402
from uec.data.support import shared_support_probe  # noqa: E402
from uec.data.synthetic import D, make_pair  # noqa: E402
from uec.explain.perturbation import tree_shap  # noqa: E402
from uec.metrics.collinearity import reliable_features  # noqa: E402
from uec.metrics.distances import DISTANCES  # noqa: E402
from uec.metrics.normalise import l1_abs  # noqa: E402
from uec.metrics.uec import change, preserved_mask, summarise  # noqa: E402
from uec.models.trees import accuracy, build_checkpoints, probabilities  # noqa: E402
from uec.paths import RESULTS  # noqa: E402

BASE = np.zeros(D)
EPS_GRID = [0.01, 0.02, 0.05, 0.10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--families", nargs="+", default=["none", "covariate", "concept", "shortcut"])
    ap.add_argument("--magnitude", type=float, default=1.5)
    ap.add_argument("--n-source", type=int, default=8000)
    ap.add_argument("--n-update", type=int, default=4000)
    ap.add_argument("--n-probe", type=int, default=500)
    ap.add_argument("--n-estimators", type=int, default=200)
    a = ap.parse_args()

    rows = []
    for seed in range(a.seeds):
        for family in a.families:
            t0 = time.time()
            src, tgt = make_pair(family, magnitude=a.magnitude)
            ck = build_checkpoints(src, tgt, seed, a.n_source, a.n_update,
                                   n_estimators=a.n_estimators)
            models = {k: v[0] for k, v in ck.items()}
            probe = shared_support_probe(src, tgt, a.n_probe, np.random.default_rng(500 + seed))

            p0 = probabilities(models["source"], probe)
            pt = probabilities(models["treatment"], probe)
            A = {k: tree_shap(m, probe) for k, m in models.items()}

            Xa, ya = src.sample(4000, np.random.default_rng(900 + seed))
            Xb, yb = tgt.sample(4000, np.random.default_rng(910 + seed))
            diag = {
                "acc_source_src": accuracy(models["source"], Xa, ya),
                "acc_treat_tgt": accuracy(models["treatment"], Xb, yb),
                "agree_treat": float(((p0 >= 0.5) == (pt >= 0.5)).mean()),
            }
            feats_reliable = reliable_features(A["source"])
            gt_src = gt_attributions(src, probe, BASE)
            gt_tgt = gt_attributions(tgt, probe, BASE)

            for feat_name, feats in (("all", None), ("reliable", feats_reliable)):
                if feats is not None and len(feats) < 3:
                    continue
                for dname, dist in DISTANCES.items():
                    raw = {k: change(A["source"], A[k], l1_abs, dist, feats)
                           for k in ("treatment", "matched_null", "seed")}
                    # omega must be restricted to the same feature subset as delta, or the
                    # decomposition subtracts a reference computed on a different vector space
                    omega = change(gt_src, gt_tgt, l1_abs, dist, feats)
                    for eps in EPS_GRID:
                        m = preserved_mask(p0, pt, eps)
                        if m.sum() < 10:
                            continue
                        s = summarise(
                            raw["treatment"][m], omega[m], np.zeros(0),
                            raw["matched_null"][m], raw["seed"][m], n_probe=len(probe),
                            seed=seed, family=family, explainer="tree_shap",
                            explainer_family="shapley", distance=dname, phi="abs",
                            features=feat_name, eps=eps, magnitude=a.magnitude, **diag,
                        )
                        rows.append(s.as_dict())
            print(f"  s{seed} {family:10s} agree={diag['agree_treat']:.3f} "
                  f"acc={diag['acc_source_src']:.3f} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "trees_metrics.parquet")
    q = df[(df.distance == "l1") & (df.features == "all") & (df.eps == 0.05)]
    print("\n" + q.groupby("family")[
        ["rho_null", "rho_seed", "omega", "delta", "ratio", "uec", "preserved_frac"]
    ].mean().round(4).to_string())
    print(f"\nwrote {len(df)} rows")


if __name__ == "__main__":
    main()
