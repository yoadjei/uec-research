"""The remaining ablation axes from audit §20: IG baseline, SHAP background size, probe sampling,
architecture width, LIME kernel width.

Each varies one free choice and reports the covariate-shift ratio. The question is not whether the
number moves — it will — but whether the *conclusion* moves, so the ranking agreement against the
primary configuration is reported alongside.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.data.support import shared_support_probe  # noqa: E402
from uec.data.synthetic import D, make_pair  # noqa: E402
from uec.explain.cache import attribute  # noqa: E402
from uec.metrics.distances import d_l1  # noqa: E402
from uec.metrics.normalise import l1_abs  # noqa: E402
from uec.metrics.uec import change, preserved_mask  # noqa: E402
from uec.models.mlp import probabilities  # noqa: E402
from uec.paths import RESULTS  # noqa: E402
from uec.rng import pin_threads  # noqa: E402
from uec.train.harness import TrainConfig, UpdateConfig, build_checkpoints  # noqa: E402

BASE = np.zeros(D)


def ratio_for(ck, probe, name, eps, **kw):
    f0, f1, fn = ck["source"][0], ck["treatment"][0], ck["matched_null"][0]
    m = preserved_mask(probabilities(f0, probe), probabilities(f1, probe), eps)
    if m.sum() < 15:
        return np.nan, np.nan, np.nan, int(m.sum())
    A0 = attribute(f0, probe, name, use_cache=False, **kw)
    A1 = attribute(f1, probe, name, use_cache=False, **kw)
    An = attribute(fn, probe, name, use_cache=False, **kw)
    delta = change(A0, A1, l1_abs, d_l1)[m]
    rho = change(A0, An, l1_abs, d_l1)[m]
    return delta.mean(), rho.mean(), delta.mean() / rho.mean(), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--magnitude", type=float, default=1.5)
    ap.add_argument("--n-source", type=int, default=8000)
    ap.add_argument("--n-update", type=int, default=4000)
    ap.add_argument("--n-probe", type=int, default=300)
    ap.add_argument("--eps", type=float, default=0.05)
    a = ap.parse_args()

    pin_threads(4)
    ucfg = UpdateConfig(lr=2e-4, epochs=2)
    rows = []

    for seed in range(a.seeds):
        t0 = time.time()
        src, tgt = make_pair("covariate", magnitude=a.magnitude)
        rng = np.random.default_rng(500 + seed)
        probe_shared = shared_support_probe(src, tgt, a.n_probe, rng)
        probe_src, _ = src.sample(a.n_probe, rng)
        probe_tgt, _ = tgt.sample(a.n_probe, rng)
        bg_pool, _ = src.sample(50, rng)

        def record(axis, value, name, probe, **kw):
            d, r, ratio, n = ratio_for(ck, probe, name, a.eps, **kw)
            rows.append({"seed": seed, "axis": axis, "value": str(value), "explainer": name,
                         "delta": d, "rho_null": r, "ratio": ratio, "n_preserved": n})

        # architecture width varies the model, so it needs its own checkpoints
        for width in (32, 64, 128):
            ck = build_checkpoints(src, tgt, seed, a.n_source, a.n_update,
                                   TrainConfig(epochs=60, hidden=(width, width)), ucfg,
                                   regimes=("matched_null", "treatment"))
            for name in ("integrated_gradients", "gradient_x_input"):
                record("architecture_width", width, name, probe_shared)

        ck = build_checkpoints(src, tgt, seed, a.n_source, a.n_update,
                               TrainConfig(epochs=60), ucfg,
                               regimes=("matched_null", "treatment"))

        # frozen backbone: audit 16.6 -- does restricting which layers move change the picture?
        for frozen in (0, 1):
            ck_f = build_checkpoints(src, tgt, seed, a.n_source, a.n_update,
                                     TrainConfig(epochs=60),
                                     UpdateConfig(lr=2e-4, epochs=2, freeze_layers=frozen),
                                     regimes=("matched_null", "treatment"))
            for name in ("integrated_gradients", "gradient_x_input"):
                d, r, ratio, n = ratio_for(ck_f, probe_shared, name, a.eps)
                rows.append({"seed": seed, "axis": "frozen_layers", "value": str(frozen),
                             "explainer": name, "delta": d, "rho_null": r, "ratio": ratio,
                             "n_preserved": n})

        # IG baseline: the reference point the path integral starts from
        for label, b in (("zeros", np.zeros(D)),
                         ("source_mean", src.sample(4000, rng)[0].mean(0)),
                         ("target_mean", tgt.sample(4000, rng)[0].mean(0))):
            record("ig_baseline", label, "integrated_gradients", probe_shared, baseline=b)

        for n_bg in (10, 25, 50):
            record("shap_background", n_bg, "kernel_shap", probe_shared[:120],
                   background=bg_pool[:n_bg])

        for kw in (None, 0.5, 2.0):
            record("lime_kernel_width", kw, "lime", probe_shared[:120],
                   background=bg_pool[:25], kernel_width=kw)

        # probe sampling: shared support is the only set where both checkpoints are in-distribution
        for label, p in (("shared_support", probe_shared), ("source_only", probe_src),
                         ("target_only", probe_tgt)):
            for name in ("integrated_gradients", "gradient_x_input"):
                record("probe_sampling", label, name, p)

        print(f"  seed {seed} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "ablations_extra.parquet")
    print("\n" + df.groupby(["axis", "value", "explainer"])[
        ["delta", "rho_null", "ratio", "n_preserved"]
    ].mean().round(4).to_string())
    print(f"\nwrote {len(df)} rows")


if __name__ == "__main__":
    main()
