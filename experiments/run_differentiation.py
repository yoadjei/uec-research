"""T4: what each prior metric reports, on the same checkpoints, across shift families.

The shortcut family is the discriminating case. There the Bayes-optimal predictor genuinely stops
using S0, so a correctly adapting model *should* move its attributions. FASS-style filtered
distance and Delta-Audit's spurious residual have no way to express that, and report the movement
as instability. UEC subtracts omega and reports approximately nothing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.data.groundtruth import gt_attributions, omega_reference  # noqa: E402
from uec.data.support import shared_support_probe  # noqa: E402
from uec.data.synthetic import D, S0, make_pair  # noqa: E402
from uec.explain.cache import attribute  # noqa: E402
from uec.metrics.baselines import compare_all  # noqa: E402
from uec.metrics.distances import d_spearman  # noqa: E402
from uec.metrics.normalise import l1_abs  # noqa: E402
from uec.metrics.uec import change, preserved_mask  # noqa: E402
from uec.models.mlp import logits, probabilities  # noqa: E402
from uec.paths import RESULTS  # noqa: E402
from uec.rng import pin_threads  # noqa: E402
from uec.train.harness import TrainConfig, UpdateConfig, build_checkpoints  # noqa: E402

BASE = np.zeros(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--families", nargs="+", default=["covariate", "concept", "shortcut"])
    ap.add_argument("--explainers", nargs="+",
                    default=["integrated_gradients", "gradient_x_input", "kernel_shap"])
    ap.add_argument("--update-epochs", type=int, nargs="+", default=[2, 20, 100])
    ap.add_argument("--update-lr", type=float, default=5e-4)
    ap.add_argument("--magnitude", type=float, default=1.5)
    ap.add_argument("--n-source", type=int, default=8000)
    ap.add_argument("--n-update", type=int, default=4000)
    ap.add_argument("--n-probe", type=int, default=400)
    ap.add_argument("--eps", type=float, default=0.05)
    a = ap.parse_args()

    pin_threads(4)
    cfg = TrainConfig(epochs=60)
    rows = []

    for seed in range(a.seeds):
        for family in a.families:
            src, tgt = make_pair(family, magnitude=a.magnitude)
            probe = shared_support_probe(src, tgt, a.n_probe, np.random.default_rng(700 + seed))
            gt_s = gt_attributions(src, probe, BASE)
            gt_t = gt_attributions(tgt, probe, BASE)

            for ep in a.update_epochs:
                ucfg = UpdateConfig(lr=a.update_lr, epochs=ep)
                ck = build_checkpoints(src, tgt, seed, a.n_source, a.n_update, cfg, ucfg,
                                       regimes=("null", "seed", "treatment"))
                f0, f1 = ck["source"][0], ck["treatment"][0]
                p0, p1 = probabilities(f0, probe), probabilities(f1, probe)
                z0, z1 = logits(f0, probe), logits(f1, probe)
                mask = preserved_mask(p0, p1, a.eps)
                if mask.sum() < 20:
                    continue

                omega = omega_reference(src, tgt, probe, BASE, l1_abs, d_spearman)
                for name in a.explainers:
                    bg, _ = src.sample(25, np.random.default_rng(seed))
                    A0 = attribute(f0, probe, name, background=bg)
                    A1 = attribute(f1, probe, name, background=bg)
                    An = attribute(ck["null"][0], probe, name, background=bg)

                    delta = change(A0, A1, l1_abs, d_spearman)
                    floor = float(change(A0, An, l1_abs, d_spearman)[mask].mean())

                    rec = compare_all(A0, A1, z0, z1, p0, p1, delta, omega, mask, floor)
                    rec.update({
                        "seed": seed, "family": family, "explainer": name, "update_epochs": ep,
                        "delta": float(delta[mask].mean()), "rho_null": floor,
                        "ratio": float(delta[mask].mean() / floor) if floor > 0 else np.nan,
                        "n_preserved": int(mask.sum()), "n_probe": len(probe),
                        "agree": float((p0.round() == p1.round()).mean()),
                        # does the model actually track the mechanism it should?
                        "s0_mass_source": float(l1_abs(A0)[:, S0].mean()),
                        "s0_mass_treat": float(l1_abs(A1)[:, S0].mean()),
                        "s0_mass_gt_source": float(l1_abs(gt_s)[:, S0].mean()),
                        "s0_mass_gt_target": float(l1_abs(gt_t)[:, S0].mean()),
                    })
                    rows.append(rec)
            print(f"  s{seed} {family} done", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "differentiation.parquet")

    cols = ["fass_distance", "delta_audit_jsd", "delta_audit_spurious", "ros_mean",
            "omega", "uec", "ratio"]
    print("\n" + df.groupby(["family", "update_epochs", "explainer"])[cols].mean().round(4).to_string())
    print("\nS0 attribution mass (shortcut family):")
    sc = df[df.family == "shortcut"]
    if len(sc):
        print(sc.groupby(["update_epochs", "explainer"])[
            ["s0_mass_source", "s0_mass_treat", "s0_mass_gt_source", "s0_mass_gt_target"]
        ].mean().round(4).to_string())


if __name__ == "__main__":
    main()
