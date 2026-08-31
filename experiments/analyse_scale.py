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
    "resnet18_small": 78_042, "resnet18": 11_181_642,
    "distilbert": 66_955_010, "distilbert(heavy)": 66_955_010,
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

    # the heavy-update DistilBERT point, measured on Kaggle and recorded verbatim. It is kept
    # separate because it is confounded with update strength (docs/preregistration_scale.md).
    # the swept result supersedes the heavy-update point: it is the one that excludes the
    # update-size confound (docs/preregistration_scale.md)
    swept = RESULTS / "scale_text_sweep_ig.csv"
    if swept.exists():
        sw = pd.read_csv(swept)
        r, lo, hi = ratio_ci(sw.delta.values, sw.rho_null.values, n_boot=20000)
        rows.append({
            "model": "distilbert", "n_params": 66_955_010,
            "explainer": "integrated_gradients", "eps": 0.05,
            "seeds": int(sw.seed.nunique()),
            "delta": float(sw.delta.mean()), "rho_null": float(sw.rho_null.mean()),
            "ratio": r, "ratio_lo": lo, "ratio_hi": hi,
            "p": paired_test(sw.delta.values, sw.rho_null.values),
            "cliffs_delta": cliffs_delta(sw.delta.values, sw.rho_null.values),
            "preserved_frac": 0.67, "agree_treat": float(sw.agree_treat.mean()),
        })

    heavy = RESULTS / "scale_text_heavy.csv"
    if heavy.exists() and not swept.exists():
        h = pd.read_csv(heavy)
        for name, s in h.groupby("explainer"):
            r, lo, hi = ratio_ci(s.delta.values, s.rho_null.values, n_boot=20000)
            rows.append({
                "model": "distilbert(heavy)", "n_params": 66_955_010, "explainer": name,
                "eps": 0.05, "seeds": int(s.seed.nunique()),
                "delta": float(s.delta.mean()), "rho_null": float(s.rho_null.mean()),
                "ratio": r, "ratio_lo": lo, "ratio_hi": hi,
                "p": paired_test(s.delta.values, s.rho_null.values),
                "cliffs_delta": cliffs_delta(s.delta.values, s.rho_null.values),
                "preserved_frac": float(s.preserved_frac.mean()),
                "agree_treat": float(s.agree_treat.mean()),
            })

    tab = pd.DataFrame(rows).sort_values(["n_params", "explainer"])
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "T15_scale.csv", index=False)
    print("\n" + tab.round(4).to_string(index=False))

    use_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    # DistilBERT is drawn detached. It differs from the others in architecture and in starting from
    # a pretrained checkpoint, not only in parameter count, so a connecting line would assert a
    # smooth trend across a gap that confounds three things at once.
    for name, sub in tab.groupby("explainer"):
        sub = sub.dropna(subset=["n_params"]).sort_values("n_params")
        scratch = sub[sub.model != "distilbert"]
        if not scratch.empty:
            ax.errorbar(scratch.n_params, scratch.ratio,
                        yerr=[scratch.ratio - scratch.ratio_lo,
                              scratch.ratio_hi - scratch.ratio],
                        marker="o", ms=4, capsize=2, lw=1.2, label=label(name))
        pre = sub[sub.model == "distilbert"]
        if not pre.empty:
            ax.errorbar(pre.n_params, pre.ratio,
                        yerr=[pre.ratio - pre.ratio_lo, pre.ratio_hi - pre.ratio],
                        marker="D", ms=7, capsize=2, lw=0, color=PALETTE["delta"],
                        markerfacecolor="white", markeredgewidth=1.5,
                        label="DistilBERT (pretrained)")
            ax.annotate("pretrained\ntransformer", ha="center", fontsize=6.8,
                        color=PALETTE["delta"], xytext=(0, 30), textcoords="offset points",
                        xy=(float(pre.n_params.iloc[0]), float(pre.ratio.iloc[0])))
    ax.axhline(1.0, color="0.4", ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("model parameters")
    ax.set_ylabel(r"$\Delta / \rho_{\mathrm{null}}$")
    ax.set_title("Flat across 630$\\times$ in models trained from scratch; "
                 "absent in a pretrained transformer", fontsize=8.5)
    ax.legend(ncol=2, fontsize=6.8, loc="lower left")
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
