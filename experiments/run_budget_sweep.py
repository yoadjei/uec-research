"""How much sampling does a stochastic explainer need before it can resolve a shift effect?

Two of our seven explainers are noise-dominated at their default budgets: Expected Gradients has
nu = 0.135 against a matched null of 0.138, and LIME's nu exceeds its null. That is reported as a
finding, but a finding without a number is a shrug. This sweeps the sample budget and locates the
point where nu drops below rho_null -- below that budget the explainer cannot answer the question
at all, whatever the true effect is.
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

BUDGETS = {
    "expected_gradients": ("n_samples", [32, 128, 512]),
    "lime": ("num_samples", [1500, 5000, 15000]),
    "kernel_shap": ("nsamples", [128, 512, 2048]),
    "smoothgrad": ("n_samples", [32, 128, 512]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--magnitude", type=float, default=1.5)
    ap.add_argument("--n-source", type=int, default=8000)
    ap.add_argument("--n-update", type=int, default=4000)
    ap.add_argument("--n-probe", type=int, default=200)
    ap.add_argument("--n-probe-small", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.05)
    a = ap.parse_args()

    pin_threads(4)
    rows = []
    for seed in range(a.seeds):
        src, tgt = make_pair("covariate", magnitude=a.magnitude)
        ck = build_checkpoints(src, tgt, seed, a.n_source, a.n_update,
                               TrainConfig(epochs=60), UpdateConfig(lr=2e-4, epochs=2),
                               regimes=("matched_null", "treatment"))
        f0, f1, fn = ck["source"][0], ck["treatment"][0], ck["matched_null"][0]
        rng = np.random.default_rng(500 + seed)
        probe = shared_support_probe(src, tgt, a.n_probe, rng)
        bg, _ = src.sample(25, rng)

        for name, (param, values) in BUDGETS.items():
            X = probe[: a.n_probe_small] if name in ("lime", "kernel_shap") else probe
            m = preserved_mask(probabilities(f0, X), probabilities(f1, X), a.eps)
            if m.sum() < 15:
                continue
            for v in values:
                t0 = time.time()
                kw = {param: v, "background": bg, "use_cache": False}
                A0 = attribute(f0, X, name, run=0, **kw)
                A0b = attribute(f0, X, name, run=1, **kw)
                A1 = attribute(f1, X, name, **kw)
                An = attribute(fn, X, name, **kw)

                nu = change(A0, A0b, l1_abs, d_l1)[m].mean()
                rho = change(A0, An, l1_abs, d_l1)[m].mean()
                delta = change(A0, A1, l1_abs, d_l1)[m].mean()
                rows.append({
                    "seed": seed, "explainer": name, "param": param, "budget": v,
                    "nu": nu, "rho_null": rho, "delta": delta,
                    "ratio": delta / rho if rho > 0 else np.nan,
                    "nu_over_rho": nu / rho if rho > 0 else np.nan,
                    "resolvable": bool(nu < rho),
                    "seconds": time.time() - t0, "n_preserved": int(m.sum()),
                })
                print(f"  s{seed} {name:20s} {param}={v:<6d} nu={nu:.4f} rho={rho:.4f} "
                      f"ratio={rows[-1]['ratio']:.3f} ({rows[-1]['seconds']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "budget_sweep.parquet")
    print("\n" + df.groupby(["explainer", "budget"])[
        ["nu", "rho_null", "delta", "ratio", "nu_over_rho", "resolvable", "seconds"]
    ].mean().round(4).to_string())
    print(f"\nwrote {len(df)} rows")


if __name__ == "__main__":
    main()
