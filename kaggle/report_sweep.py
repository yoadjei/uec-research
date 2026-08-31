"""Print the update-strength sweep in the form the pre-registered decision rule needs.

    !python uec-research/kaggle/report_sweep.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO / "results"

from uec.stats.inference import cliffs_delta, paired_test, ratio_ci  # noqa: E402


def main():
    p = OUT / "scale_text_sweep.parquet"
    if not p.exists():
        print(f"missing {p}")
        return
    df = pd.read_parquet(p)
    q = df[(df.distance == "l1") & (df.eps == 0.05)]

    print("=" * 78)
    print("DistilBERT update-strength sweep  (l1, eps=0.05)")
    print("=" * 78)
    print(f"seeds: {sorted(df.seed.unique().tolist())}   rows: {len(df)}")

    print("\n--- by update strength ---")
    rows = []
    for (lr, ex), s in q.groupby(["update_lr", "explainer"]):
        r, lo, hi = ratio_ci(s.delta.values, s.rho_null.values, n_boot=20000)
        rows.append({
            "update_lr": lr, "explainer": ex, "seeds": int(s.seed.nunique()),
            "agree": float(s.agree_treat.mean()),
            "preserved": float(s.preserved_frac.mean()),
            "delta": float(s.delta.mean()), "rho_null": float(s.rho_null.mean()),
            "ratio": r, "lo": lo, "hi": hi,
            "p": paired_test(s.delta.values, s.rho_null.values),
            "cliff": cliffs_delta(s.delta.values, s.rho_null.values),
        })
    out = pd.DataFrame(rows).sort_values(["explainer", "update_lr"])
    print(out.round(4).to_string(index=False))

    print("\n--- per seed (integrated_gradients) ---")
    ig = q[q.explainer == "integrated_gradients"]
    print(ig[["seed", "update_lr", "delta", "rho_null", "ratio", "n_preserved", "agree_treat"]]
          .sort_values(["update_lr", "seed"]).round(4).to_string(index=False))

    print("\n--- pre-registered decision rule ---")
    ig_out = out[out.explainer == "integrated_gradients"]
    usable = ig_out[ig_out.preserved >= 0.15]
    if usable.empty:
        print("S3 inconclusive: no update strength kept enough preserved points")
    else:
        best = usable.loc[usable.update_lr.idxmin()]
        print(f"lightest usable update: lr={best.update_lr:g}, "
              f"ratio={best.ratio:.4f} [{best.lo:.4f}, {best.hi:.4f}], "
              f"agreement={best.agree:.3f}, preserved={best.preserved:.2f}")
        if best.ratio >= 1.25 and best.lo > 1:
            print("=> S1: the effect holds at scale")
        elif (usable.ratio <= 1.10).all():
            print("=> S2: the effect does NOT transfer to DistilBERT. "
                  "Report as a scale bound, in the abstract.")
        elif 1.10 < best.ratio < 1.25 and best.lo > 1:
            print("=> S4: attenuated but present")
        else:
            print("=> S3: inconclusive")

    print("\nPaste everything from the first ==== line down.")


if __name__ == "__main__":
    main()
