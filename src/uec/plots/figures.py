"""Figure builders. Each returns (fig, data) so the numbers behind every panel are saved beside
the image and nothing in the paper is unbacked."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..stats.inference import bootstrap_ci, ratio_ci
from .style import FAMILY_LABEL, PALETTE, label


def _primary(df, family="covariate", eps=0.05, distance="spearman", phi="abs", features="all"):
    q = df[
        (df.family == family) & (df.eps == eps) & (df.distance == distance)
        & (df.phi == phi) & (df.features == features)
    ]
    return q


def fig_headline(df, family="covariate", eps=0.05, order=None):
    """Fig 2. Per explainer: noise floor, matched null, seed floor, and shift-induced change."""
    q = _primary(df, family=family, eps=eps)
    names = order or sorted(q.explainer.unique())
    rows = []
    for name in names:
        s = q[q.explainer == name]
        if s.empty:
            continue
        for key in ("nu", "rho_null", "rho_seed", "delta"):
            m, lo, hi = bootstrap_ci(s[key].values)
            rows.append({"explainer": name, "quantity": key, "mean": m, "lo": lo, "hi": hi})
        r, rlo, rhi = ratio_ci(s.delta.values, s.rho_null.values)
        rows.append({"explainer": name, "quantity": "ratio", "mean": r, "lo": rlo, "hi": rhi})
    data = pd.DataFrame(rows)

    plotted = [n for n in names if n in set(data.explainer)]
    fig, ax = plt.subplots(figsize=(1.15 * len(plotted) + 2.2, 3.0))
    width = 0.2
    for i, key in enumerate(("nu", "rho_null", "rho_seed", "delta")):
        sub = data[data.quantity == key].set_index("explainer").reindex(plotted)
        x = np.arange(len(plotted)) + (i - 1.5) * width
        ax.bar(x, sub["mean"], width, color=PALETTE[key], label={
            "nu": r"explainer noise $\nu$",
            "rho_null": r"matched null $\rho_{\mathrm{null}}$",
            "rho_seed": r"seed floor $\rho_{\mathrm{seed}}$",
            "delta": r"shift-induced $\Delta$",
        }[key])
        ax.errorbar(x, sub["mean"], yerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]],
                    fmt="none", ecolor="0.25", elinewidth=0.7, capsize=1.8)

    for i, name in enumerate(plotted):
        r = data[(data.explainer == name) & (data.quantity == "ratio")]["mean"].values
        if r.size:
            top = data[(data.explainer == name)]["hi"].max()
            ax.text(i, top * 1.04, f"{r[0]:.2f}$\\times$", ha="center", fontsize=7.2, color=PALETTE["delta"])

    ax.set_xticks(np.arange(len(plotted)))
    ax.set_xticklabels([label(n) for n in plotted], rotation=18, ha="right")
    ax.set_ylabel(r"attribution change  $1-\rho_s$ (scaled)")
    ax.set_title(f"Explanation change against its floors — {FAMILY_LABEL.get(family, family)} shift"
                 f" ($\\epsilon={eps}$)")
    ax.legend(ncol=2, loc="upper left")
    return fig, data


def fig_update_strength(sweep, explainers=None):
    """The result the matched null buys: which control you choose reverses the conclusion."""
    df = sweep.copy()
    explainers = explainers or sorted(df.explainer.unique())
    fig, axes = plt.subplots(1, len(explainers), figsize=(3.3 * len(explainers), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    rows = []

    for ax, name in zip(axes, explainers):
        s = df[df.explainer == name]
        for mag, grp in s.groupby("magnitude"):
            g = grp.groupby("update_epochs").agg(
                ratio=("ratio", "mean"), ratio_seed=("ratio_seed", "mean")
            ).reset_index()
            ax.plot(g.update_epochs, g.ratio, marker="o", ms=3,
                    label=f"shift {mag:g}")
            rows += [{"explainer": name, "magnitude": mag, **r} for r in g.to_dict("records")]
        ref = s.groupby("update_epochs")["ratio_seed"].mean().reset_index()
        ax.plot(ref.update_epochs, ref.ratio_seed, ls="--", color=PALETTE["rho_seed"],
                marker="s", ms=3, label="vs seed floor")
        ax.axhline(1.0, color="0.4", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("update epochs")
        ax.set_title(label(name))
    axes[0].set_ylabel(r"$\Delta$ / floor")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Shift-induced change relative to the matched null and to the seed floor", y=1.03)
    return fig, pd.DataFrame(rows)


def fig_warranted_alignment(perpoint, families=("concept", "shortcut")):
    """Fig 3. Measured change against the warranted reference."""
    fig, axes = plt.subplots(1, len(families), figsize=(3.2 * len(families), 2.9))
    axes = np.atleast_1d(axes)
    rows = []
    for ax, fam in zip(axes, families):
        xs, ys = [], []
        for key, arr in perpoint.items():
            seed, family, name = key.split("|")
            if family != fam or name != "integrated_gradients":
                continue
            delta, _, _, _, omega = arr
            xs.append(omega)
            ys.append(delta)
        if not xs:
            ax.set_visible(False)
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        ax.scatter(x, y, s=4, alpha=0.25, color=PALETTE["omega"], edgecolors="none")
        lim = [0, max(x.max(), y.max()) * 1.05]
        ax.plot(lim, lim, ls="--", color="0.4", lw=0.8, label=r"$\Delta=\omega$")
        r = np.corrcoef(x, y)[0, 1] if x.std() > 0 else np.nan
        ax.set_title(f"{FAMILY_LABEL.get(fam, fam)}  ($r={r:.2f}$)")
        ax.set_xlabel(r"warranted change $\omega$")
        ax.legend(loc="upper left")
        rows.append({"family": fam, "pearson_r": r, "n": len(x)})
    axes[0].set_ylabel(r"measured change $\Delta$")
    return fig, pd.DataFrame(rows)


def fig_eps_sweep(df, family="covariate"):
    """Fig 4. Exceedance and preserved fraction across the epsilon grid."""
    q = df[(df.family == family) & (df.distance == "spearman") & (df.phi == "abs")
           & (df.features == "all")]
    g = q.groupby(["explainer", "eps"]).agg(
        exceedance=("exceedance", "mean"), preserved=("preserved_frac", "mean"),
        ratio=("ratio", "mean")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for name, s in g.groupby("explainer"):
        axes[0].plot(s.eps, s.exceedance, marker="o", ms=3, label=label(name))
        axes[1].plot(s.eps, s.ratio, marker="o", ms=3, label=label(name))
    axes[0].axhline(0.05, color="0.4", ls="--", lw=0.8)
    axes[0].set_ylabel("exceedance rate")
    axes[1].axhline(1.0, color="0.4", ls="--", lw=0.8)
    axes[1].set_ylabel(r"$\Delta/\rho_{\mathrm{null}}$")
    for ax in axes:
        ax.set_xlabel(r"prediction-preservation threshold $\epsilon$")
        ax.set_xscale("log")
    axes[1].legend(ncol=2, fontsize=6.5)
    fig.suptitle(f"Sensitivity to the preservation threshold — {FAMILY_LABEL.get(family, family)}",
                 y=1.02)
    return fig, g


def fig_theory(theory):
    """Fig 5. IG's aggregate is pinned by output change; gradient x input's is not."""
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7))

    axes[0].hist(theory.ig_slack_max, bins=20, color=PALETTE["rho_null"])
    axes[0].axvline(1.0, color=PALETTE["delta"], ls="--")
    axes[0].set_xlabel("max IG slack per run")
    axes[0].set_title("Prop. 1(i): IG aggregate bounded")

    axes[1].hist(theory.gi_slack_median, bins=20, color=PALETTE["accent"])
    axes[1].axvline(1.0, color=PALETTE["delta"], ls="--")
    axes[1].set_xlabel(r"median Grad$\times$Input slack")
    axes[1].set_title("Prop. 2: no aggregate bound")

    axes[2].hist(np.log10(theory.coal_ratio_median.clip(lower=1e-3)), bins=20,
                 color=PALETTE["omega"])
    axes[2].axvline(0.0, color=PALETTE["delta"], ls="--")
    axes[2].set_xlabel(r"$\log_{10}(\epsilon_{\mathrm{coal}}/\epsilon_{\mathrm{data}})$")
    axes[2].set_title("Prop. 3 premise fails off-manifold")
    for ax in axes:
        ax.set_ylabel("runs")
    return fig, theory


def fig_invisibility(df, family="covariate", eps=0.05):
    """Fig 6. Unwarranted change is not predicted by the metrics practitioners watch."""
    q = _primary(df, family=family, eps=eps)
    q = q[q.explainer == "integrated_gradients"]
    cols = [("agree_treat", "prediction agreement"), ("acc_treat_tgt", "target accuracy"),
            ("ece_treat", "target ECE")]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 2.7))
    rows = []
    for ax, (c, name) in zip(np.atleast_1d(axes), cols):
        if c not in q:
            ax.set_visible(False)
            continue
        ax.scatter(q[c], q.delta, s=14, color=PALETTE["delta"], alpha=0.8, edgecolors="none")
        r = np.corrcoef(q[c], q.delta)[0, 1] if q[c].std() > 0 else np.nan
        ax.set_xlabel(name)
        ax.set_title(f"$r={r:.2f}$")
        rows.append({"covariate": c, "pearson_r": r, "n": len(q)})
    np.atleast_1d(axes)[0].set_ylabel(r"$\Delta$")
    fig.suptitle("Explanation change against the quantities practitioners monitor", y=1.03)
    return fig, pd.DataFrame(rows)


def fig_distance_agreement(df, family="covariate", eps=0.05):
    """Fig 7. Do the explainer rankings survive the choice of distance?"""
    from scipy.stats import kendalltau

    q = df[(df.family == family) & (df.eps == eps) & (df.phi == "abs") & (df.features == "all")]
    piv = q.groupby(["distance", "explainer"])["ratio"].mean().unstack()
    dists = list(piv.index)
    M = np.ones((len(dists), len(dists)))
    for i, a in enumerate(dists):
        for j, b in enumerate(dists):
            ok = piv.loc[a].notna() & piv.loc[b].notna()
            if ok.sum() >= 3:
                M[i, j] = kendalltau(piv.loc[a][ok], piv.loc[b][ok]).statistic

    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(dists)), dists, rotation=45, ha="right")
    ax.set_yticks(range(len(dists)), dists)
    for i in range(len(dists)):
        for j in range(len(dists)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"Kendall $\tau$")
    ax.set_title("Explainer-ranking agreement across distances")
    return fig, pd.DataFrame(M, index=dists, columns=dists).reset_index()
