"""Analyse the GPU scale results and place them beside every other model we measured.

Reads whichever of `scale_text.parquet`, `scale_vision.parquet`, `scale_width.parquet` are present
in results/ and produces:

  T15_scale.csv        ratio with a seed-bootstrap CI per model and explainer
  fig12_scale.png      the ratio against parameter count, from a 1.7k MLP to DistilBERT's 66M

The point of the figure is a single question a reviewer can check by eye: does the effect decay
with model size? A flat line says the phenomenon is not an artefact of small models; a line
descending to 1 would bound the paper's claim, and that is a publishable answer too.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from uec.paths import FIGURES, RESULTS, ROOT  # noqa: E402
from uec.plots.style import PALETTE, label, save, use_style  # noqa: E402
from uec.stats.inference import cliffs_delta, paired_test, ratio_ci  # noqa: E402

OUT = ROOT / "paper" / "tables"

# parameter counts for everything we have measured, so the scale axis is honest
KNOWN_PARAMS = {
    "mlp32": 1_696, "mlp64": 5_440, "mlp128": 19_072,
    "mlp256": 70_912, "mlp512": 272_896, "mlp1024": 1_070_080,
    "resnet18_small": 78_042, "resnet18": 11_181_642, "distilbert": 66_955_010,
}


def _load(name):
    p = RESULTS / name
    return pd.read_parquet(p) if p.exists() else None


def summarise_arm(df, model, eps=0.05, distance="l1"):
    q = df[(df.distance == distance) & (df.eps == eps)]
    if q.empty:
        q = df[df.distance == distance]
        if q.empty:
            return []
        eps = float(q.eps.min())
        q = q[q.eps == eps]
    rows = []
    for name, s in q.groupby("explainer"):
        r, lo, hi = ratio_ci(s.delta.values, s.rho_null.values, n_boot=20000)
        rows.append({
            "model": model, "n_params": KNOWN_PARAMS.get(model, np.nan),
            "explainer": name, "eps": eps, "seeds": int(s.seed.nunique()),
            "delta": float(s.delta.mean()), "rho_null": float(s.rho_null.mean()),
            "ratio": r, "ratio_lo": lo, "ratio_hi": hi,
            "p": paired_test(s.delta.values, s.rho_null.values) if s.seed.nunique() > 1 else np.nan,
            "cliffs_delta": cliffs_delta(s.delta.values, s.rho_null.values),
            "preserved_frac": float(s.preserved_frac.mean()) if "preserved_frac" in s else np.nan,
            "agree_treat": float(s.agree_treat.mean()) if "agree_treat" in s else np.nan,
        })
    return rows


def main():
    rows = []

    width = _load("scale_width.parquet")
    if width is not None:
        for w, s in width.groupby("width"):
            for name, t in s.groupby("explainer"):
                r, lo, hi = ratio_ci(t.delta.values, t.rho_null.values, n_boot=20000)
                rows.append({
                    "model": f"mlp{w}", "n_params": int(t.n_params.iloc[0]),
                    "explainer": name, "eps": 0.05, "seeds": int(t.seed.nunique()),
                    "delta": float(t.delta.mean()), "rho_null": float(t.rho_null.mean()),
                    "ratio": r, "ratio_lo": lo, "ratio_hi": hi,
                    "p": paired_test(t.delta.values, t.rho_null.values),
                    "cliffs_delta": cliffs_delta(t.delta.values, t.rho_null.values),
                    "preserved_frac": np.nan, "agree_treat": np.nan,
                })

    cpu_vision = _load("vision_metrics.parquet")
    if cpu_vision is not None:
        rows += summarise_arm(cpu_vision[cpu_vision.family == "corruption"], "resnet18_small")

    for fname, model in (("scale_vision.parquet", "resnet18"),
                         ("scale_text.parquet", "distilbert")):
        df = _load(fname)
        if df is not None:
            rows += summarise_arm(df, model)
        else:
            print(f"  {fname} not present -- skipping {model}")

    tab = pd.DataFrame(rows).sort_values(["n_params", "explainer"])
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "T15_scale.csv", index=False)
    print("\n" + tab.round(4).to_string(index=False))

    use_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for name, s in tab.groupby("explainer"):
        s = s.dropna(subset=["n_params"]).sort_values("n_params")
        if s.empty:
            continue
        ax.errorbar(s.n_params, s.ratio,
                    yerr=[s.ratio - s.ratio_lo, s.ratio_hi - s.ratio],
                    marker="o", ms=4, capsize=2, lw=1.2, label=label(name))
    ax.axhline(1.0, color="0.4", ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("model parameters")
    ax.set_ylabel(r"$\Delta / \rho_{\mathrm{null}}$")
    ax.set_title("Does the effect decay with model size?")
    ax.legend(ncol=2, fontsize=7)
    print("\nwrote", save(fig, FIGURES / "fig12_scale.png", tab))

    big = tab[tab.n_params > 1e6]
    if len(big):
        print("\n=== models above 1M parameters ===")
        print(big[["model", "n_params", "explainer", "ratio", "ratio_lo", "ratio_hi",
                   "seeds", "preserved_frac"]].round(4).to_string(index=False))
        if (big.ratio_lo > 1).all():
            print("\nEvery interval excludes 1: the effect does not vanish at scale.")
        elif (big.ratio_hi < 1.15).all():
            print("\nIntervals sit at ~1: the effect does NOT transfer to these models. "
                  "That bounds the claim to small models and belongs in the limitations.")
        else:
            print("\nMixed. Report per model rather than as a single claim.")


if __name__ == "__main__":
    main()
