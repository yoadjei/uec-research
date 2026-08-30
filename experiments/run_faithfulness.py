"""E6: is the observed change really faithfulness loss?

The objection this answers is "your explainers are simply bad on one of the two checkpoints". If
that were so, large Δ would occur only where faithfulness collapsed. We measure faithfulness on
each checkpoint separately, then ask whether Δ survives on the subset where the explainer is
faithful to *both*.
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
from uec.explain.cache import attribute  # noqa: E402
from uec.explain.registry import EXPENSIVE  # noqa: E402
from uec.metrics.distances import d_l1  # noqa: E402
from uec.metrics.faithfulness import faithfulness, mechanism_fidelity  # noqa: E402
from uec.metrics.normalise import l1_abs  # noqa: E402
from uec.metrics.uec import change, preserved_mask  # noqa: E402
from uec.models.mlp import logits, probabilities  # noqa: E402
from uec.paths import RESULTS  # noqa: E402
from uec.rng import pin_threads  # noqa: E402
from uec.train.harness import TrainConfig, UpdateConfig, build_checkpoints  # noqa: E402

BASE = np.zeros(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--families", nargs="+", default=["covariate", "shortcut"])
    ap.add_argument("--explainers", nargs="+",
                    default=["integrated_gradients", "gradient_x_input", "saliency",
                             "kernel_shap", "lime"])
    ap.add_argument("--magnitude", type=float, default=1.5)
    ap.add_argument("--n-source", type=int, default=8000)
    ap.add_argument("--n-update", type=int, default=4000)
    ap.add_argument("--n-probe", type=int, default=250)
    ap.add_argument("--update-epochs", type=int, default=2)
    ap.add_argument("--update-lr", type=float, default=2e-4)
    ap.add_argument("--eps", type=float, default=0.05)
    a = ap.parse_args()

    pin_threads(4)
    cfg = TrainConfig(epochs=60)
    ucfg = UpdateConfig(lr=a.update_lr, epochs=a.update_epochs)
    rows = []

    for seed in range(a.seeds):
        for family in a.families:
            t0 = time.time()
            src, tgt = make_pair(family, magnitude=a.magnitude)
            ck = build_checkpoints(src, tgt, seed, a.n_source, a.n_update, cfg, ucfg,
                                   regimes=("matched_null", "treatment"))
            f0, f1 = ck["source"][0], ck["treatment"][0]
            fn = ck["matched_null"][0]

            probe = shared_support_probe(src, tgt, a.n_probe, np.random.default_rng(500 + seed))
            mask = preserved_mask(probabilities(f0, probe), probabilities(f1, probe), a.eps)
            if mask.sum() < 20:
                continue
            bg, _ = src.sample(25, np.random.default_rng(seed))
            gt_s = gt_attributions(src, probe, BASE)
            gt_t = gt_attributions(tgt, probe, BASE)

            for name in a.explainers:
                X = probe[:120] if name in EXPENSIVE else probe
                m = mask[: len(X)]
                A0 = attribute(f0, X, name, background=bg)
                A1 = attribute(f1, X, name, background=bg)
                An = attribute(fn, X, name, background=bg)

                fa = faithfulness(lambda z: logits(f0, z), X, A0, BASE)
                fb = faithfulness(lambda z: logits(f1, z), X, A1, BASE)
                delta = change(A0, A1, l1_abs, d_l1)
                rho = change(A0, An, l1_abs, d_l1)

                both = np.isfinite(fa) & np.isfinite(fb)
                thr = np.nanmedian(np.minimum(fa, fb))
                faithful_both = both & (np.minimum(fa, fb) >= thr) & m
                unfaithful = both & (np.minimum(fa, fb) < thr) & m

                rows.append({
                    "seed": seed, "family": family, "explainer": name,
                    "faith_source": float(np.nanmean(fa[m])),
                    "faith_treat": float(np.nanmean(fb[m])),
                    "faith_drop": float(np.nanmean(fa[m]) - np.nanmean(fb[m])),
                    "fidelity_source": float(mechanism_fidelity(A0[m], gt_s[:len(X)][m],
                                                                l1_abs, d_l1).mean()),
                    "fidelity_treat": float(mechanism_fidelity(A1[m], gt_t[:len(X)][m],
                                                               l1_abs, d_l1).mean()),
                    "delta_all": float(delta[m].mean()),
                    "rho_all": float(rho[m].mean()),
                    "ratio_all": float(delta[m].mean() / rho[m].mean()),
                    "delta_faithful": float(delta[faithful_both].mean())
                    if faithful_both.sum() > 5 else np.nan,
                    "rho_faithful": float(rho[faithful_both].mean())
                    if faithful_both.sum() > 5 else np.nan,
                    "ratio_faithful": float(delta[faithful_both].mean() / rho[faithful_both].mean())
                    if faithful_both.sum() > 5 and rho[faithful_both].mean() > 0 else np.nan,
                    "delta_unfaithful": float(delta[unfaithful].mean())
                    if unfaithful.sum() > 5 else np.nan,
                    "corr_delta_faith": float(np.corrcoef(np.minimum(fa, fb)[m], delta[m])[0, 1])
                    if np.nanstd(np.minimum(fa, fb)[m]) > 1e-9 else np.nan,
                    "n_faithful": int(faithful_both.sum()), "n_preserved": int(m.sum()),
                })
            print(f"  s{seed} {family:10s} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "faithfulness.parquet")

    print("\n=== E6: does the effect survive where the explainer is faithful to BOTH models? ===")
    print(df.groupby(["family", "explainer"])[
        ["faith_source", "faith_treat", "faith_drop", "fidelity_source", "fidelity_treat",
         "ratio_all", "ratio_faithful", "corr_delta_faith"]
    ].mean().round(4).to_string())
    print(f"\nwrote {len(df)} rows")


if __name__ == "__main__":
    main()
