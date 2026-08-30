"""Renders every figure from the result parquets. Each figure writes its own source data beside
the image so no number in the paper is unbacked."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.data.groundtruth import gt_attributions  # noqa: E402
from uec.data.support import shared_support_probe  # noqa: E402
from uec.data.synthetic import BLOCKS, D, make_pair  # noqa: E402
from uec.explain.cache import attribute  # noqa: E402
from uec.metrics.normalise import l1_abs  # noqa: E402
from uec.metrics.uec import change, preserved_mask  # noqa: E402
from uec.metrics.distances import d_spearman  # noqa: E402
from uec.models.mlp import probabilities  # noqa: E402
from uec.paths import FIGURES, RESULTS  # noqa: E402
from uec.plots import figures as F  # noqa: E402
from uec.plots.style import PALETTE, save, use_style  # noqa: E402
from uec.rng import pin_threads  # noqa: E402
from uec.train.harness import TrainConfig, UpdateConfig, build_checkpoints  # noqa: E402

BASE = np.zeros(D)


def _load(name):
    p = RESULTS / name
    return pd.read_parquet(p) if p.exists() else None


def fig_concept(seed=0, magnitude=1.5, explainer="integrated_gradients"):
    """Fig 1, built from real checkpoints rather than drawn by hand.

    Three prediction-preserved probe points from two real experiments: one where the explanation
    holds still under covariate shift (as it should, omega = 0), one where it moves although
    omega = 0 (the phenomenon), and one where it moves because the mechanism moved (warranted).
    """
    cfg, ucfg = TrainConfig(epochs=60), UpdateConfig(lr=2e-4, epochs=2)
    panels = []

    for family, title in (("covariate", "covariate shift"), ("shortcut", "shortcut removal")):
        src, tgt = make_pair(family, magnitude=magnitude)
        probe = shared_support_probe(src, tgt, 400, np.random.default_rng(700 + seed))
        ck = build_checkpoints(src, tgt, seed, 8000, 4000,
                               cfg, UpdateConfig(lr=5e-4, epochs=60) if family == "shortcut" else ucfg,
                               regimes=("treatment",))
        f0, f1 = ck["source"][0], ck["treatment"][0]
        p0, p1 = probabilities(f0, probe), probabilities(f1, probe)
        keep = preserved_mask(p0, p1, 0.05)
        A0, A1 = attribute(f0, probe, explainer), attribute(f1, probe, explainer)
        delta = change(A0, A1, l1_abs, d_spearman)
        omega = np.zeros(len(probe)) if family == "covariate" else change(
            gt_attributions(src, probe, BASE), gt_attributions(tgt, probe, BASE), l1_abs, d_spearman
        )
        idx = np.flatnonzero(keep)
        if family == "covariate":
            panels.append(("stable: $\\Delta\\approx0=\\omega$", A0, A1,
                           idx[np.argmin(delta[idx])], 0.0))
            panels.append(("unwarranted: $\\Delta>0=\\omega$", A0, A1,
                           idx[np.argmax(delta[idx])], 0.0))
        else:
            j = idx[np.argmax(omega[idx])] if idx.size else 0
            panels.append(("warranted: $\\Delta\\approx\\omega>0$", A0, A1, j, omega[j]))

    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.5), sharey=True)
    rows = []
    x = np.arange(D)
    for ax, (title, A0, A1, j, w) in zip(axes, panels):
        a, b = l1_abs(A0[j]), l1_abs(A1[j])
        ax.bar(x - 0.2, a, 0.4, color=PALETTE["rho_null"], label="$f_t$")
        ax.bar(x + 0.2, b, 0.4, color=PALETTE["delta"], label="$f_{t+1}$")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([np.mean(v) for v in BLOCKS.values()])
        ax.set_xticklabels(list(BLOCKS), rotation=20, fontsize=6.5)
        for edge in [b_[-1] + 0.5 for b_ in list(BLOCKS.values())[:-1]]:
            ax.axvline(edge, color="0.8", lw=0.6)
        rows += [{"panel": title, "feature": int(i), "f_t": float(a[i]),
                  "f_t1": float(b[i]), "omega": float(w)} for i in range(D)]
    axes[0].set_ylabel("attribution mass")
    axes[0].legend()
    fig.suptitle("Predictions preserved on all three inputs; only the middle panel is a defect",
                 y=1.04)
    return fig, pd.DataFrame(rows)


def main():
    use_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    pin_threads(4)

    syn = _load("synthetic_metrics.parquet")
    sweep = _load("sweep_regime.parquet")
    theory = _load("synthetic_theory.parquet")
    folk = _load("folktables_metrics.parquet")
    vis = _load("vision_metrics.parquet")
    perpoint = dict(np.load(RESULTS / "synthetic_perpoint.npz")) if (
        RESULTS / "synthetic_perpoint.npz"
    ).exists() else {}

    made = []
    try:
        fig, data = fig_concept()
        made.append(save(fig, FIGURES / "fig1_concept.png", data))
    except Exception as e:  # a figure that cannot be built must say so, not vanish
        print(f"fig1 skipped: {e}")

    if syn is not None:
        fig, data = F.fig_headline(syn)
        made.append(save(fig, FIGURES / "fig2_headline.png", data))
        fig, data = F.fig_eps_sweep(syn)
        made.append(save(fig, FIGURES / "fig4_eps_sweep.png", data))
        fig, data = F.fig_invisibility(syn)
        made.append(save(fig, FIGURES / "fig6_invisibility.png", data))
        fig, data = F.fig_distance_agreement(syn)
        made.append(save(fig, FIGURES / "fig7_distance_agreement.png", data))
    if sweep is not None:
        fig, data = F.fig_update_strength(sweep)
        made.append(save(fig, FIGURES / "fig2b_update_strength.png", data))
    if perpoint:
        fig, data = F.fig_warranted_alignment(perpoint)
        made.append(save(fig, FIGURES / "fig3_warranted.png", data))
    if theory is not None:
        fig, data = F.fig_theory(theory)
        made.append(save(fig, FIGURES / "fig5_theory.png", data))
    if folk is not None:
        fig, data = F.fig_headline(folk, family="covariate_state")
        made.append(save(fig, FIGURES / "fig8_folktables.png", data))
    if vis is not None:
        fig, data = F.fig_headline(vis, family="corruption", eps=0.05)
        made.append(save(fig, FIGURES / "fig9_vision.png", data))

    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
