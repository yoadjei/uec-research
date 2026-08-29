"""Where does shift-induced explanation change separate from the matched null?

The null and the treatment apply the same operator; the only difference is the distribution. Both
therefore contain the same optimisation transient, and the shift signal is only visible once that
transient has saturated. This sweep locates the regime, over shift magnitude and update strength,
where the ratio departs from 1 -- and reports honestly if it never does.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.data.support import shared_support_probe  # noqa: E402
from uec.data.synthetic import D, base_environment, make_pair  # noqa: E402
from uec.explain.cache import attribute  # noqa: E402
from uec.metrics.distances import d_spearman  # noqa: E402
from uec.metrics.normalise import l1_abs  # noqa: E402
from uec.metrics.uec import change, preserved_mask  # noqa: E402
from uec.models.mlp import accuracy, probabilities  # noqa: E402
from uec.paths import RESULTS  # noqa: E402
from uec.rng import pin_threads  # noqa: E402
from uec.train.harness import TrainConfig, UpdateConfig, agreement_rate, build_checkpoints  # noqa: E402

BASE = np.zeros(D)
NAMES = ["integrated_gradients", "gradient_x_input"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--magnitudes", type=float, nargs="+", default=[0.5, 0.75, 1.25, 2.0])
    ap.add_argument("--update-epochs", type=int, nargs="+", default=[5, 20, 60, 150])
    ap.add_argument("--update-lrs", type=float, nargs="+", default=[5e-4, 2e-3])
    ap.add_argument("--n-source", type=int, default=8000)
    ap.add_argument("--n-update", type=int, default=4000)
    ap.add_argument("--n-probe", type=int, default=300)
    ap.add_argument("--eps", type=float, default=0.05)
    a = ap.parse_args()

    pin_threads(4)
    cfg = TrainConfig(epochs=60)
    rows = []

    for seed in range(a.seeds):
        for mag in a.magnitudes:
            src, tgt = make_pair("covariate", magnitude=mag)
            probe = shared_support_probe(src, tgt, a.n_probe, np.random.default_rng(500 + seed))
            Xa, ya = src.sample(4000, np.random.default_rng(9000 + seed))
            Xb, yb = tgt.sample(4000, np.random.default_rng(9100 + seed))

            for lr in a.update_lrs:
                for ep in a.update_epochs:
                    t0 = time.time()
                    ucfg = UpdateConfig(lr=lr, epochs=ep)
                    ck = build_checkpoints(
                        src, tgt, seed, a.n_source, a.n_update, cfg, ucfg,
                        regimes=("null", "seed", "treatment"),
                    )
                    f0 = ck["source"][0]
                    m = preserved_mask(
                        probabilities(f0, probe), probabilities(ck["treatment"][0], probe), a.eps
                    )
                    if m.sum() < 20:
                        continue

                    for name in NAMES:
                        A0 = attribute(f0, probe, name)
                        rec = {
                            r: change(A0, attribute(ck[r][0], probe, name), l1_abs, d_spearman)[m]
                            for r in ("null", "seed", "treatment")
                        }
                        rows.append({
                            "seed": seed, "magnitude": mag, "update_lr": lr, "update_epochs": ep,
                            "explainer": name,
                            "delta": rec["treatment"].mean(),
                            "rho_null": rec["null"].mean(),
                            "rho_seed": rec["seed"].mean(),
                            "ratio": rec["treatment"].mean() / rec["null"].mean(),
                            "ratio_seed": rec["treatment"].mean() / rec["seed"].mean(),
                            "preserved_frac": float(m.mean()),
                            "agree": agreement_rate(f0, ck["treatment"][0], probe),
                            "acc_src": accuracy(f0, Xa, ya),
                            "acc_treat_tgt": accuracy(ck["treatment"][0], Xb, yb),
                            "acc_src_on_tgt": accuracy(f0, Xb, yb),
                        })
                    print(f"  s{seed} mag={mag} lr={lr:g} ep={ep:3d} "
                          f"ratio={rows[-1]['ratio']:.3f} agree={rows[-1]['agree']:.3f} "
                          f"({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "sweep_regime.parquet")

    piv = df.groupby(["explainer", "magnitude", "update_lr", "update_epochs"])[
        ["ratio", "ratio_seed", "delta", "rho_null", "agree", "preserved_frac"]
    ].mean().round(3)
    print("\n" + piv.to_string())
    print(f"\nmax ratio observed: {df.ratio.max():.3f}")


if __name__ == "__main__":
    main()
