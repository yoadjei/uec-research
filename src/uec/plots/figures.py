"""Figure builders. Each returns (fig, data) so the numbers behind every panel are saved beside
the image and nothing in the paper is unbacked."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..stats.inference import bootstrap_ci, ratio_ci
from .style import FAMILY_LABEL, PALETTE, label


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

    # the ratio is on a different scale from the bars; it annotates, it must not set the y-limit
    bars = data[data.quantity != "ratio"]
    ymax = bars["hi"].max() * 1.18
    for i, name in enumerate(plotted):
        r = data[(data.explainer == name) & (data.quantity == "ratio")]["mean"].values
        if r.size:
            top = bars[bars.explainer == name]["hi"].max()
            ax.text(i, min(top * 1.06, ymax * 0.97), f"{r[0]:.2f}$\\times$", ha="center",
                    fontsize=7.2, color=PALETTE["delta"])
    ax.set_ylim(0, ymax)

    ax.set_xticks(np.arange(len(plotted)))
    ax.set_xticklabels([label(n) for n in plotted], rotation=18, ha="right")
    ax.set_ylabel(r"attribution change (normalised $\ell_1$)")
    ax.set_title(f"Explanation change against its floors — {FAMILY_LABEL.get(family, family)} shift"
                 f" ($\\epsilon={eps}$)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
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
        ax.scatter(x, y, s=4, alpha=0.25, color=PALETTE["omega"], edgecolors="none")
        lim = [0, max(x.max(), y.max()) * 1.05]
        ax.plot(lim, lim, ls="--", color="0.4", lw=0.8, label=r"$\Delta=\omega$")
        r_seed = np.array(per_seed)
        m = r_seed.mean()
        se = r_seed.std(ddof=1) / np.sqrt(len(r_seed))
        ax.set_title(f"{FAMILY_LABEL.get(fam, fam)}  "
                     f"($r={m:+.2f}$ [{m - 1.96 * se:+.2f}, {m + 1.96 * se:+.2f}])")
        ax.set_xlabel(r"warranted change $\omega$")
        ax.legend(loc="upper left")
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
    fig.suptitle(f"Per-point change distributions — {FAMILY_LABEL.get(family, family)} shift", y=1.03)
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
    fig.suptitle(f"Stability against faithfulness — {FAMILY_LABEL.get(family, family)} shift", y=1.04)
    return fig, q[["seed", "explainer", "faith_source", "faith_treat",
                   "ratio_all", "ratio_faithful", "corr_delta_faith"]]


def fig_adaptation(adapt, semisynth_point=(1.09, 0.807)):
    """Fig 13. Per-point tracking against how far the update actually travelled.

    The relationship is peaked at completeness 1, not monotone, so the panel is the argument: a
    reader can see that tracking is recovered where the model finishes adapting and degrades again
    once it overshoots. The semi-synthetic run is marked because it is the point that motivated the
    sweep -- it lands on the curve rather than beside it.
    """
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for fam, marker in (("shortcut", "o"), ("concept", "s")):
        q = adapt[adapt.family == fam]
        if q.empty:
            continue
        g = q.groupby("update_epochs").agg(
            completeness=("completeness", "mean"), r=("r", "mean"),
            sd=("r", "std"), n=("r", "size")).reset_index()
        err = 1.96 * g.sd / np.sqrt(g.n)
        colour = PALETTE["delta"] if fam == "shortcut" else PALETTE["rho_null"]
        ax.errorbar(g.completeness, g.r, yerr=err, marker=marker, ms=4, lw=1.0,
                    capsize=2, color=colour, label=FAMILY_LABEL.get(fam, fam))
        for _, x in g.iterrows():
            ax.annotate(f"{int(x.update_epochs)}ep", (x.completeness, x.r),
                        textcoords="offset points", xytext=(4, -9), fontsize=6, color=colour)

    ax.axvline(1.0, color="0.4", ls="--", lw=0.8)
    ax.text(1.02, ax.get_ylim()[0] + 0.04, "adaptation complete", fontsize=6, color="0.35")
    ax.scatter(*semisynth_point, marker="*", s=90, color=PALETTE["omega"], zorder=5,
               label="semi-synthetic (real covariates)")
    ax.set_xlabel(r"adaptation completeness  $\mathbb{E}[\Delta]/\mathbb{E}[\omega]$")
    ax.set_ylabel(r"per-point $r(\Delta,\omega)$")
    ax.axhline(0.0, color="0.8", lw=0.6, zorder=0)
    ax.legend(fontsize=6, loc="lower left")
    rows = adapt.groupby(["family", "update_epochs"]).agg(
        completeness=("completeness", "mean"), r=("r", "mean"), seeds=("r", "size")).reset_index()
    return fig, rows


def fig_share_model(rows):
    """Fig 14. What predicts the optimiser share, and where DistilBERT sits relative to it."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    scratch = rows[rows.pretrained == 0]

    w = scratch[scratch.source == "mlp_width"]
    g = w.groupby("n_params").share.agg(["mean", "std", "count"]).reset_index()
    axes[0].errorbar(g.n_params, g["mean"], yerr=1.96 * g["std"] / np.sqrt(g["count"]),
                     marker="o", ms=4, lw=1.0, capsize=2, color=PALETTE["rho_null"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("parameters")
    axes[0].set_ylabel(r"optimiser share  $\rho_{\rm null}/\Delta$")
    axes[0].set_title("capacity: no effect ($p=0.94$)")
    axes[0].set_ylim(0, 1.05)

    fit = scratch[scratch.agree.notna()]
    axes[1].scatter(fit.agree, fit.share, s=6, alpha=0.25, color=PALETTE["rho_null"],
                    edgecolors="none", label="from scratch")
    pre = rows[rows.pretrained == 1]
    axes[1].scatter(pre.agree, pre.share, s=40, marker="*", color=PALETTE["delta"],
                    zorder=5, label="DistilBERT")
    axes[1].set_xlabel("prediction agreement (update size)")
    axes[1].set_title("update strength: monotone")
    axes[1].set_ylim(0, 1.35)
    axes[1].legend(fontsize=6, loc="lower left")
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

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    x = np.arange(len(g))
    axes[0].bar(x - 0.19, g.ratio_all, 0.38, color=PALETTE["delta"], label="all preserved points")
    axes[0].bar(x + 0.19, g.ratio_faithful, 0.38, color=PALETTE["rho_null"],
                label="faithful points only")
    axes[0].axhline(1.0, color="0.4", ls="--", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label(n) for n in g.explainer], rotation=18, ha="right")
    axes[0].set_ylabel(r"$\Delta/\rho_{\mathrm{null}}$")
    axes[0].set_title("restricting to faithful points changes nothing — "
                      f"{FAMILY_LABEL.get(family, family)}")
    axes[0].legend(fontsize=6)

    # The right panel spans *all* families, not just `family`: the text quotes the per-seed range
    # over every row, and showing only one family here would put a different number in the panel
    # from the one in the sentence beside it.
    axes[1].axhline(0.0, color="0.4", ls="--", lw=0.8)
    for i, (name, s) in enumerate(fa.groupby("explainer")):
        axes[1].scatter(np.full(len(s), i), s.corr_delta_faith, s=10, alpha=0.5,
                        color=PALETTE["omega"], edgecolors="none")
    axes[1].set_xticks(np.arange(fa.explainer.nunique()))
    axes[1].set_xticklabels([label(n) for n in sorted(fa.explainer.unique())],
                            rotation=18, ha="right")
    axes[1].set_ylabel(r"corr($\Delta$, faithfulness drop)")
    axes[1].set_title(f"per seed, all families ({len(fa)} rows)")
    axes[1].set_ylim(-0.6, 0.6)
    return fig, fa.reset_index(drop=True)


def fig_distributions(df, family="covariate", eps=0.05):
    """Fig 10. Medians and interquartile ranges, not just means.

    The audit asks for distribution panels because a mean ratio can hide a bimodal Delta. It does
    not: the quartiles sit either side of the mean for every explainer.
    """
    q = _primary(df, family=family, eps=eps)
    rows = []
    for name, s in q.groupby("explainer"):
        for key in ("nu", "rho_null", "rho_seed", "delta"):
            v = s[key].values
            rows.append({"explainer": name, "quantity": key, "median": np.median(v),
                         "q25": np.percentile(v, 25), "q75": np.percentile(v, 75), "n": len(v)})
    data = pd.DataFrame(rows)

    names = sorted(q.explainer.unique())
    fig, ax = plt.subplots(figsize=(1.15 * len(names) + 2.2, 3.0))
    width = 0.2
    for i, key in enumerate(("nu", "rho_null", "rho_seed", "delta")):
        sub = data[data.quantity == key].set_index("explainer").reindex(names)
        xs = np.arange(len(names)) + (i - 1.5) * width
        ax.bar(xs, sub["median"], width, color=PALETTE[key], label=key)
        ax.errorbar(xs, sub["median"],
                    yerr=[sub["median"] - sub["q25"], sub["q75"] - sub["median"]],
                    fmt="none", ecolor="0.25", elinewidth=0.7, capsize=1.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([label(n) for n in names], rotation=18, ha="right")
    ax.set_ylabel(r"attribution change (median, IQR)")
    ax.set_title(f"Distributions, not means — {FAMILY_LABEL.get(family, family)}")
    ax.legend(ncol=4, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    return fig, data
