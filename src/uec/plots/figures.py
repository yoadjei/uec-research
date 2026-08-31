"""Figure builders. Each returns (fig, data) so the numbers behind every panel are saved beside
the image and nothing in the paper is unbacked."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..stats.inference import bootstrap_ci, ratio_ci
from .style import (
    BAR_EDGE,
    FAMILY_LABEL,
    HATCH,
    MARKER,
    PALETTE,
    QUANTITY_LABEL,
    explainer_ticks,
    label,
)


def _primary(df, family="covariate", eps=0.05, distance="l1", phi="abs", features="all"):
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
    fig, ax = plt.subplots(figsize=(1.05 * len(plotted) + 2.0, 3.4))
    width = 0.2
    # Order matters: the headline quantity is Delta/rho_null, so those two bars sit adjacent. With
    # rho_seed between them the ratio bracket had to span a third bar and read as a property of the
    # whole group rather than of the pair it actually describes.
    for i, key in enumerate(("nu", "rho_seed", "rho_null", "delta")):
        sub = data[data.quantity == key].set_index("explainer").reindex(plotted)
        x = np.arange(len(plotted)) + (i - 1.5) * width
        ax.bar(x, sub["mean"], width, color=PALETTE[key], hatch=HATCH[key],
               edgecolor=BAR_EDGE, linewidth=0.4, label=QUANTITY_LABEL[key])
        ax.errorbar(x, sub["mean"], yerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]],
                    fmt="none", ecolor="0.2", elinewidth=0.7, capsize=1.8)
        # A deterministic explainer has nu = 0 exactly. Drawing nothing makes that look like a
        # missing measurement, so it is stated.
        if key == "nu":
            for xi, v in zip(x, sub["mean"].values):
                if v < 1e-9:
                    ax.text(xi, ax.get_ylim()[1] * 0.012, "0", ha="center", va="bottom",
                            fontsize=8, color="0.35")

    # The ratio is Delta/rho_null, so it is drawn over those two bars rather than over the group,
    # where it read as a property of the tallest bar (the seed floor) instead.
    bars = data[data.quantity != "ratio"]
    ymax = bars["hi"].max() * 1.24
    for i, name in enumerate(plotted):
        r = data[(data.explainer == name) & (data.quantity == "ratio")]["mean"].values
        if not r.size:
            continue
        pair = data[(data.explainer == name) & data.quantity.isin(["rho_null", "delta"])]
        top = pair["hi"].max()
        xc = i + width                # midpoint of the adjacent rho_null and delta bars
        ax.annotate("", xy=(i + 0.5 * width, top * 1.05), xytext=(i + 1.5 * width, top * 1.05),
                    arrowprops=dict(arrowstyle="-", color=PALETTE["delta"], lw=0.8))
        ax.text(xc, top * 1.08, f"{r[0]:.2f}$\\times$", ha="center", va="bottom",
                fontsize=9, color=PALETTE["delta"], fontweight="bold")
    ax.set_ylim(0, ymax)

    explainer_ticks(ax, plotted)
    # No figure-level title: ICLR figures carry their description in the caption below, and a
    # title here would repeat it and cost vertical space. Panel identifiers are kept elsewhere.
    ax.set_ylabel(r"attribution change (normalised $\ell_1$)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    return fig, data


def fig_update_strength(sweep, explainers=None):
    """The result the matched null buys: which control you choose reverses the conclusion."""
    df = sweep.copy()
    explainers = explainers or sorted(df.explainer.unique())
    fig, axes = plt.subplots(1, len(explainers), figsize=(2.9 * len(explainers), 3.0),
                             sharey=True)
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
    fig.tight_layout(w_pad=1.2)
    return fig, pd.DataFrame(rows)


def fig_warranted_alignment(perpoint, families=("concept", "shortcut")):
    """Fig 3. Measured change against the warranted reference."""
    fig, axes = plt.subplots(1, len(families), figsize=(3.0 * len(families), 3.1))
    axes = np.atleast_1d(axes)
    rows = []
    for ax, fam in zip(axes, families):
        xs, ys, per_seed = [], [], []
        for key, arr in perpoint.items():
            seed, family, name = key.split("|")
            if family != fam or name != "integrated_gradients":
                continue
            delta, _, _, _, omega = arr
            xs.append(omega)
            ys.append(delta)
            # Correlate within seed: pooling first would ignore the clustering and gives a
            # different number from the one the text reports (T17b).
            if np.std(omega) > 1e-12:
                per_seed.append(np.corrcoef(delta, omega)[0, 1])
        if not xs:
            ax.set_visible(False)
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        ax.scatter(x, y, s=4, alpha=0.18, color=PALETTE["omega"], edgecolors="none",
                   rasterized=True)
        lim = [0, max(x.max(), y.max()) * 1.05]
        ax.plot(lim, lim, ls="--", color="0.35", lw=0.9,
                label=r"$\Delta=\omega$ (perfect tracking)")
        # Both panels share a y-scale. With independent scales the right panel's flat band looked
        # flatter than the left's purely because omega ranges twice as far.
        ax.set_ylim(0, 0.16)
        r_seed = np.array(per_seed)
        m = r_seed.mean()
        se = r_seed.std(ddof=1) / np.sqrt(len(r_seed))
        ax.set_title(f"{FAMILY_LABEL.get(fam, fam)}  "
                     f"($r={m:+.2f}$ [{m - 1.96 * se:+.2f}, {m + 1.96 * se:+.2f}])")
        ax.set_xlabel(r"warranted change $\omega$")
        if ax is axes[0]:
            ax.legend(loc="upper left")
        # State the reference's range, not a max/min ratio: the minimum is essentially zero, so a
        # ratio reports the smallest observed omega rather than anything about the spread.
        ax.text(x.max() * 0.60, 0.118,
                "measured change stays flat while\n"
                rf"the reference spans ${np.percentile(x, 1):.2f}$–${np.percentile(x, 99):.2f}$",
                fontsize=8, color="0.35", ha="center", va="center")
        rows.append({"family": fam, "pearson_r": m, "ci_lo": m - 1.96 * se,
                     "ci_hi": m + 1.96 * se, "pooled_r": np.corrcoef(x, y)[0, 1],
                     "seeds": len(r_seed), "n": len(x)})
    axes[0].set_ylabel(r"measured change $\Delta$")
    return fig, pd.DataFrame(rows)


def fig_eps_sweep(df, family="covariate"):
    """Fig 4. Exceedance and preserved fraction across the epsilon grid."""
    q = df[(df.family == family) & (df.distance == "l1") & (df.phi == "abs")
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
    axes[1].legend(ncol=2, fontsize=8.5)
    return fig, g


def fig_theory(theory):
    """Fig 5. IG's aggregate is pinned by output change; gradient x input's is not."""
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))

    # Plotting raw slack put every run just above the 1.0 line and read as a universal
    # violation, with tick labels colliding under matplotlib's "+1" offset. The bound is tested
    # against each run's own quadrature tolerance, so that is what the axis shows: <1 means the
    # excess is numerical, which is why the violation count is 0.
    excess = (theory.ig_slack_max - 1.0) / theory.ig_quad_tol
    axes[0].hist(excess, bins=20, color=PALETTE["rho_null"], edgecolor=BAR_EDGE, linewidth=0.4)
    axes[0].axvline(1.0, color=PALETTE["delta"], ls="--", lw=1.2)
    axes[0].set_xlabel("IG bound excess / quadrature tolerance")
    axes[0].set_title("Prop. 1(i): within tolerance")
    axes[0].text(0.03, 0.96, f"{int(theory.ig_violations.sum())} violations\nacross all runs",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=8.5,
                 color=PALETTE["rho_null"])

    axes[1].hist(theory.gi_slack_median, bins=20, color=PALETTE["accent"],
                 edgecolor=BAR_EDGE, linewidth=0.4)
    axes[1].axvline(1.0, color=PALETTE["delta"], ls="--", lw=1.2)
    axes[1].set_xlabel(r"median Grad$\times$Input slack")
    axes[1].set_title("Prop. 2: bound broken")
    axes[1].text(0.03, 0.96, f"{theory.gi_exceeds_one.mean():.0%} of points\nexceed the bound",
                 transform=axes[1].transAxes, ha="left", va="top", fontsize=8.5,
                 color=PALETTE["accent"])

    axes[2].hist(np.log10(theory.coal_ratio_median.clip(lower=1e-3)), bins=20,
                 color=PALETTE["omega"], edgecolor=BAR_EDGE, linewidth=0.4)
    axes[2].axvline(0.0, color=PALETTE["delta"], ls="--", lw=1.2)
    axes[2].set_xlabel(r"$\log_{10}(\epsilon_{\mathrm{coal}}/\epsilon_{\mathrm{data}})$")
    axes[2].set_title("Prop. 3: premise fails")
    axes[2].text(0.03, 0.96, "coalitions move\nfurther than the data",
                 transform=axes[2].transAxes, ha="left", va="top", fontsize=8.5,
                 color=PALETTE["omega"])
    for ax in axes:
        ax.set_ylabel("runs")
        # Headroom so the annotation never sits on top of a bar: the tallest bar in each panel
        # reaches the axis top otherwise, and the text was being struck through by it.
        ax.set_ylim(0, ax.get_ylim()[1] * 1.42)
    fig.tight_layout(w_pad=1.4)
    return fig, theory


def fig_invisibility(df, family="covariate", eps=0.05):
    """Fig 6. Unwarranted change is not predicted by the metrics practitioners watch."""
    q = _primary(df, family=family, eps=eps)
    q = q[q.explainer == "integrated_gradients"]
    cols = [("agree_treat", "prediction agreement"), ("acc_treat_tgt", "target accuracy"),
            ("ece_treat", "target ECE")]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.8 * len(cols), 3.0))
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
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"Kendall $\tau$")
    return fig, pd.DataFrame(M, index=dists, columns=dists).reset_index()


def fig_distributions(perpoint, family="covariate", explainers=None):
    """Audit §19.4: report the *distribution* of the per-point change, not only its mean.

    Left: violins of the four quantities. Right: ECDFs, where the separation between the shift
    curve and the matched-null curve is the effect, and the gap to the seed floor is what a
    seed-floor control would have reported instead.
    """
    keys = [k for k in perpoint if k.split("|")[1] == family]
    explainers = explainers or ["integrated_gradients", "gradient_x_input", "kernel_shap"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
    rows = []

    stacked = {}
    for name in explainers:
        arrs = [perpoint[k] for k in keys if k.split("|")[2] == name]
        if not arrs:
            continue
        stacked[name] = np.concatenate(arrs, axis=1)

    order = ["nu", "rho_null", "rho_seed", "delta"]
    idx = {"delta": 0, "nu": 1, "rho_null": 2, "rho_seed": 3}
    positions, data, colors = [], [], []
    for i, name in enumerate(stacked):
        for j, q in enumerate(order):
            v = stacked[name][idx[q]]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            positions.append(i * (len(order) + 1) + j)
            data.append(v)
            colors.append(PALETTE[q])
            rows.append({"explainer": name, "quantity": q, "median": float(np.median(v)),
                         "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75)),
                         "n": int(v.size)})
    parts = axes[0].violinplot(data, positions=positions, widths=0.85, showextrema=False,
                               showmedians=True)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.85)
    parts["cmedians"].set_color("0.2")
    axes[0].set_xticks([i * (len(order) + 1) + 1.5 for i in range(len(stacked))])
    axes[0].set_xticklabels([label(n) for n in stacked], rotation=15, ha="right")
    axes[0].set_ylabel("per-point attribution change")
    axes[0].set_title("Distributions, not just means")

    for q in order:
        v = np.concatenate([stacked[n][idx[q]] for n in stacked])
        v = np.sort(v[np.isfinite(v)])
        axes[1].plot(v, np.linspace(0, 1, v.size), color=PALETTE[q], label={
            "nu": r"$\nu$", "rho_null": r"$\rho_{\mathrm{null}}$",
            "rho_seed": r"$\rho_{\mathrm{seed}}$", "delta": r"$\Delta$"}[q])
    axes[1].set_xlabel("per-point attribution change")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title("Pooled over explainers")
    axes[1].legend(loc="lower right")
    return fig, pd.DataFrame(rows)


def fig_faithfulness_quadrant(faith, family="covariate"):
    """Audit §9.11: stability against faithfulness on two axes.

    The deflation being tested is "the explainer is just bad on one checkpoint". If that were the
    story, the ratio would collapse when restricted to points where the explainer is faithful to
    *both*. The diagonal is where it does not move at all.
    """
    q = faith[faith.family == family]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))

    axes[0].scatter(q.faith_source, q.faith_treat, s=18, c=PALETTE["rho_null"], edgecolors="none")
    lim = [min(q.faith_source.min(), q.faith_treat.min()) - 0.2,
           max(q.faith_source.max(), q.faith_treat.max()) + 0.2]
    axes[0].plot(lim, lim, ls="--", lw=0.8, color="0.4")
    axes[0].set_xlabel("faithfulness on $f_t$")
    axes[0].set_ylabel("faithfulness on $f_{t+1}$")
    axes[0].set_title("Faithfulness is not lost systematically")

    axes[1].scatter(q.ratio_all, q.ratio_faithful, s=18, c=PALETTE["delta"], edgecolors="none")
    lo = min(q.ratio_all.min(), q.ratio_faithful.min()) * 0.95
    hi = max(q.ratio_all.max(), q.ratio_faithful.max()) * 1.05
    axes[1].plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="0.4", label="unchanged")
    axes[1].axhline(1.0, color="0.7", lw=0.7)
    axes[1].axvline(1.0, color="0.7", lw=0.7)
    axes[1].set_xlabel(r"$\Delta/\rho_{\mathrm{null}}$, all preserved points")
    axes[1].set_ylabel("restricted to faithful-to-both")
    axes[1].set_title("The effect does not depend on faithfulness")
    axes[1].legend(loc="upper left")
    return fig, q[["seed", "explainer", "faith_source", "faith_treat",
                   "ratio_all", "ratio_faithful", "corr_delta_faith"]]


def fig_adaptation(adapt, semisynth_point=(1.09, 0.807)):
    """Fig 13. Per-point tracking against how far the update actually travelled.

    The relationship is peaked at completeness 1, not monotone, so the panel is the argument: a
    reader can see that tracking is recovered where the model finishes adapting and degrades again
    once it overshoots. The semi-synthetic run is marked because it is the point that motivated the
    sweep -- it lands on the curve rather than beside it.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.set_ylim(-0.32, 1.16)

    # Shade the two regimes. This is the claim of the panel, so it should be visible before any
    # marker is read.
    ax.axvspan(0, 1.0, color=PALETTE["rho_seed"], alpha=0.10, zorder=0)
    ax.axvline(1.0, color="0.35", ls="--", lw=0.9, zorder=1)
    ax.text(0.99, 1.11, "under-adapted", ha="right", va="top", fontsize=8.5, color="0.35")
    ax.text(1.03, 1.11, "overshoot", ha="left", va="top", fontsize=8.5, color="0.35")

    # Points bunch near completeness 1, so a single offset per family collides. Offsets are given
    # per point, placed away from the neighbouring marker rather than uniformly.
    label_offsets = {
        "shortcut": {2: (6, 13), 20: (-8, 13), 100: (18, 7), 400: (21, 2)},
        "concept": {2: (-6, -15), 20: (8, -16), 100: (-19, 5), 400: (-10, -15)},
    }
    for fam in ("shortcut", "concept"):
        q = adapt[adapt.family == fam]
        if q.empty:
            continue
        # Sort by completeness, not by epochs: the x-axis is completeness, and joining points in
        # epoch order drew a zigzag that implied a non-monotone path the data does not show.
        g = q.groupby("update_epochs").agg(
            completeness=("completeness", "mean"), r=("r", "mean"),
            sd=("r", "std"), n=("r", "size")).reset_index().sort_values("completeness")
        err = 1.96 * g.sd / np.sqrt(g.n)
        colour = PALETTE["delta"] if fam == "shortcut" else PALETTE["rho_null"]
        ax.errorbar(g.completeness, g.r, yerr=err, marker=MARKER["delta" if fam == "shortcut"
                                                                 else "rho_null"],
                    ms=5, lw=1.4, capsize=2.5, color=colour, zorder=3,
                    label=FAMILY_LABEL.get(fam, fam))
        for _, x in g.iterrows():
            dx, dy = label_offsets[fam].get(int(x.update_epochs), (0, 11))
            ax.annotate(f"{int(x.update_epochs)}ep", (x.completeness, x.r),
                        textcoords="offset points", xytext=(dx, dy), ha="center",
                        fontsize=8, color=colour)

    # The semi-synthetic run motivated the sweep, so it is marked -- offset from the curve and
    # leadered, because at completeness 1.09 it sits directly on top of the 100ep marker.
    sx, sy = semisynth_point
    # Dark edge, not white: in greyscale the green star and the orange line sit at nearly the same
    # luminance, so the outline plus the star glyph carry the distinction.
    ax.scatter(sx, sy, marker="*", s=170, facecolor=PALETTE["omega"], edgecolor="0.15",
               linewidth=0.7, zorder=6, label="semi-synthetic (real covariates)")
    ax.annotate("real covariates\nland on the curve", xy=(sx, sy), xytext=(1.72, 0.74),
                fontsize=8.5, color=PALETTE["omega"], ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=PALETTE["omega"], lw=0.7,
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_xlabel(r"adaptation completeness  $\mathbb{E}[\Delta]\,/\,\mathbb{E}[\omega]$")
    ax.set_ylabel(r"per-point $r(\Delta,\omega)$")
    ax.axhline(0.0, color="0.75", lw=0.7, zorder=1)
    ax.set_xlim(0.12, 3.32)
    # Below the axes: every in-panel position collided with either the concept curve's 400ep
    # marker or the leader line to the semi-synthetic star.
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    rows = adapt.groupby(["family", "update_epochs"]).agg(
        completeness=("completeness", "mean"), r=("r", "mean"), seeds=("r", "size")).reset_index()
    return fig, rows


def fig_share_model(rows):
    """Fig 14. What predicts the optimiser share, and where DistilBERT sits relative to it."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    scratch = rows[rows.pretrained == 0]

    w = scratch[scratch.source == "mlp_width"]
    g = w.groupby("n_params").share.agg(["mean", "std", "count"]).reset_index()
    axes[0].errorbar(g.n_params, g["mean"], yerr=1.96 * g["std"] / np.sqrt(g["count"]),
                     marker=MARKER["rho_null"], ms=4.5, lw=1.3, capsize=2.5,
                     color=PALETTE["rho_null"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("parameters")
    axes[0].set_ylabel(r"optimiser share  $\rho_{\mathrm{null}}/\Delta$")
    axes[0].set_title("capacity: flat over $630\\times$  ($p=0.94$)")
    axes[0].set_ylim(0, 1.05)
    axes[0].annotate("", xy=(g.n_params.iloc[0], 0.86), xytext=(g.n_params.iloc[-1], 0.86),
                     arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.7))
    axes[0].text(np.sqrt(g.n_params.iloc[0] * g.n_params.iloc[-1]), 0.885,
                 "no trend", ha="center", fontsize=8.5, color="0.4")

    fit = scratch[scratch.agree.notna()]
    axes[1].scatter(fit.agree, fit.share, s=7, alpha=0.22, color=PALETTE["rho_null"],
                    edgecolors="none", label="from scratch (612 runs)", zorder=2)

    # The panel's claim is that DistilBERT sits *above* the from-scratch relationship, which cannot
    # be read off a bare scatter. Draw the fit the text quotes, on the logit scale it was fitted on.
    import statsmodels.api as sm
    z = np.log(np.clip(fit.share, 0.02, 0.98) / (1 - np.clip(fit.share, 0.02, 0.98)))
    m = sm.OLS(z, sm.add_constant(fit.agree.astype(float))).fit()
    xs = np.linspace(fit.agree.min(), 1.0, 100)
    pr = m.get_prediction(sm.add_constant(xs, has_constant="add")).summary_frame(alpha=0.05)
    inv = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
    axes[1].plot(xs, inv(pr["mean"]), color=PALETTE["rho_null"], lw=1.6, zorder=4,
                 label="fitted from-scratch curve")
    axes[1].fill_between(xs, inv(pr.mean_ci_lower), inv(pr.mean_ci_upper),
                         color=PALETTE["rho_null"], alpha=0.28, lw=0, zorder=3)

    pre = rows[rows.pretrained == 1]
    axes[1].scatter(pre.agree, pre.share, s=90, marker="*", facecolor=PALETTE["delta"],
                    edgecolor="white", linewidth=0.5, zorder=6, label="DistilBERT (66.9M)")
    axes[1].annotate(f"+{0.173:.2f} above the curve\nat matched update size",
                     xy=(pre.agree.mean(), pre.share.mean()), xytext=(0.76, 1.18),
                     fontsize=8.5, color=PALETTE["delta"], ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", color=PALETTE["delta"], lw=0.8,
                                     connectionstyle="arc3,rad=0.15"))
    axes[1].set_xlabel("prediction agreement (update size)")
    axes[1].set_ylabel(r"optimiser share  $\rho_{\mathrm{null}}/\Delta$")
    axes[1].set_title("update strength: monotone")
    axes[1].set_ylim(0, 1.38)
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout(w_pad=2.0)
    return fig, rows.groupby(["source", "pretrained"]).share.agg(["mean", "size"]).reset_index()


def fig_faithfulness(fa, family="covariate"):
    """Fig 11. The change is not faithfulness loss.

    If attribution movement were the explainers becoming *wrong* rather than merely different, the
    ratio would collapse once unfaithful points are dropped, and the movement would correlate with
    the faithfulness drop. Neither happens, so the panel exists to show two near-identical bars and
    a correlation cloud centred on zero.
    """
    q = fa[fa.family == family]
    g = q.groupby("explainer").agg(
        ratio_all=("ratio_all", "mean"), ratio_faithful=("ratio_faithful", "mean"),
        corr=("corr_delta_faith", "mean")).reset_index().sort_values("explainer")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    x = np.arange(len(g))
    axes[0].bar(x - 0.19, g.ratio_all, 0.38, color=PALETTE["delta"], hatch=HATCH["delta"],
                edgecolor=BAR_EDGE, linewidth=0.4, label="all preserved points")
    axes[0].bar(x + 0.19, g.ratio_faithful, 0.38, color=PALETTE["rho_null"],
                hatch=HATCH["rho_null"], edgecolor=BAR_EDGE, linewidth=0.4,
                label="faithful points only")
    axes[0].axhline(1.0, color="0.4", ls="--", lw=0.8)
    explainer_ticks(axes[0], list(g.explainer))
    axes[0].set_ylabel(r"$\Delta/\rho_{\mathrm{null}}$")
    axes[0].set_title("restricting to faithful points changes nothing\n"
                      f"({FAMILY_LABEL.get(family, family)} shift)")
    axes[0].legend(fontsize=8)

    # The right panel spans *all* families, not just `family`: the text quotes the per-seed range
    # over every row, and showing only one family here would put a different number in the panel
    # from the one in the sentence beside it.
    axes[1].axhline(0.0, color="0.4", ls="--", lw=0.8)
    for i, (name, s) in enumerate(fa.groupby("explainer")):
        axes[1].scatter(np.full(len(s), i), s.corr_delta_faith, s=10, alpha=0.5,
                        color=PALETTE["omega"], edgecolors="none")
    explainer_ticks(axes[1], sorted(fa.explainer.unique()))
    axes[1].set_ylabel(r"corr($\Delta$, faithfulness drop)")
    axes[1].set_title("no consistent association\n"
                      f"per seed, all families ({len(fa)} rows)")
    axes[1].set_ylim(-0.6, 0.6)
    return fig, fa.reset_index(drop=True)
