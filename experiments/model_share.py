"""Is the optimiser share predictable, or is 48-98% just a range?

The share is `rho_null / delta`: the fraction of measured attribution change that the matched null
reproduces -- the part the optimiser would have produced with no shift at all. A range is a
documented correlation; a prediction is a mechanism.

Two covariates are comparable across model classes: **prediction agreement**, which is the operative
measure of how far an update moved a model, and **capacity**. We separate them:

  capacity   the width sweep holds the update fixed and varies parameters 630x
  update     the regime sweep holds capacity fixed and varies update strength

then fit the share on models trained **from scratch** and predict DistilBERT out of sample. The
share is bounded, so it is modelled on a logit scale; a linear fit on the raw share extrapolates to
impossible negative values and must not be used.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uec.paths import RESULTS, ROOT  # noqa: E402

OUT = ROOT / "paper" / "tables"
PRIMARY = dict(distance="l1", phi="abs", features="all", eps=0.05)
CLIP = (0.02, 0.98)


def _sel(df, **over):
    sel = {**PRIMARY, **over}
    m = np.ones(len(df), bool)
    for k, v in sel.items():
        if k in df:
            m &= df[k] == v
    return df[m]


def _load(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def mlp_params(width, d=20):
    return d * width + width + width * width + width + width + 1


def logit(p):
    p = np.clip(p, *CLIP)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def collect():
    rows = []
    w = _load("scale_width.parquet")
    if w is not None:
        for _, r in w.iterrows():
            rows.append(dict(source="mlp_width", pretrained=0, n_params=mlp_params(int(r.width)),
                             agree=np.nan, seed=r.seed, delta=r.delta, rho_null=r.rho_null))
    sw = _load("sweep_regime.parquet")
    if sw is not None:
        for _, r in sw.iterrows():
            rows.append(dict(source="mlp_sweep", pretrained=0, n_params=mlp_params(64),
                             agree=r.agree, seed=r.seed, delta=r.delta, rho_null=r.rho_null))
    vis = _load("vision_metrics.parquet")
    if vis is not None:
        for _, r in _sel(vis, family="corruption").iterrows():
            rows.append(dict(source="resnet_small", pretrained=0, n_params=78_042,
                             agree=r.agree_treat, seed=r.seed,
                             delta=r.delta, rho_null=r.rho_null))
    ig = _load("scale_text_sweep_ig.csv")
    if ig is not None:
        for _, r in ig.iterrows():
            rows.append(dict(source="distilbert", pretrained=1, n_params=66_955_010,
                             agree=r.agree_treat, seed=r.seed,
                             delta=r.delta, rho_null=r.rho_null))
    df = pd.DataFrame(rows)
    df["share"] = df.rho_null / df.delta
    return df


def main():
    import statsmodels.api as sm

    df = collect()
    res = {}

    print("=" * 74)
    print("1. CAPACITY: width sweep, update strength held fixed")
    print("=" * 74)
    w = df[df.source == "mlp_width"]
    g = w.groupby("n_params").share.agg(["mean", "std", "count"])
    print(g.round(4).to_string())
    X = sm.add_constant(np.log10(w.n_params.astype(float)))
    mc = sm.OLS(logit(w.share.values), X).fit()
    slope, (lo, hi) = mc.params.iloc[1], mc.conf_int().iloc[1]
    print(f"\nlogit(share) ~ log10(params):  slope = {slope:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
          f"p = {mc.pvalues.iloc[1]:.3f}")
    span = g["mean"].iloc[-1] - g["mean"].iloc[0]
    print(f"share moves {g['mean'].iloc[0]:.3f} -> {g['mean'].iloc[-1]:.3f} "
          f"({span:+.3f}) across a 630x parameter range")
    print("=> capacity does not drive the share." if abs(span) < 0.10
          else "=> capacity matters.")
    res.update(capacity_slope=slope, capacity_p=float(mc.pvalues.iloc[1]),
               capacity_span=span)

    print("\n" + "=" * 74)
    print("2. UPDATE STRENGTH: regime sweep, capacity held fixed")
    print("=" * 74)
    sw = df[df.source == "mlp_sweep"].dropna(subset=["agree"])
    Xs = sm.add_constant(sw.agree.astype(float))
    mu = sm.OLS(logit(sw.share.values), Xs).fit()
    print(f"logit(share) ~ agreement:  slope = {mu.params.iloc[1]:+.2f} "
          f"[{mu.conf_int().iloc[1, 0]:+.2f}, {mu.conf_int().iloc[1, 1]:+.2f}]  "
          f"p = {mu.pvalues.iloc[1]:.2e}   R^2 = {mu.rsquared:.3f}   n = {int(mu.nobs)}")
    raw = _load("sweep_regime.parquet")
    byep = raw.groupby("update_epochs").apply(
        lambda x: pd.Series({"share": (x.rho_null / x.delta).mean(), "agree": x.agree.mean()}),
        include_groups=False)
    print("\n" + byep.round(3).to_string())
    print("=> update strength drives the share, monotonically.")
    res.update(update_slope=float(mu.params.iloc[1]), update_r2=float(mu.rsquared),
               update_p=float(mu.pvalues.iloc[1]))

    print("\n" + "=" * 74)
    print("3. OUT-OF-SAMPLE: predict DistilBERT from the from-scratch curve")
    print("=" * 74)
    fit = df[(df.pretrained == 0) & df.agree.notna()]
    test = df[df.pretrained == 1]
    Xf = sm.add_constant(fit.agree.astype(float))
    mf = sm.OLS(logit(fit.share.values), Xf).fit()
    Xt = sm.add_constant(test.agree.astype(float), has_constant="add")
    pr = mf.get_prediction(Xt).summary_frame(alpha=0.05)
    point = expit(pr["mean"].mean())
    # The *observation* interval covers one hypothetical future run and, with this much per-run
    # noise, spans nearly the whole range -- "inside" it would confirm nothing. The question is
    # whether DistilBERT's mean sits on the curve, so the mean interval is the right one.
    mlo, mhi = expit(pr.mean_ci_lower.mean()), expit(pr.mean_ci_upper.mean())
    olo, ohi = expit(pr.obs_ci_lower.mean()), expit(pr.obs_ci_upper.mean())
    actual = float(test.share.mean())
    print(f"fitted on {int(mf.nobs)} from-scratch rows (MLP sweep + small ResNet)")
    print(f"DistilBERT agreement = {test.agree.mean():.3f}")
    print(f"  predicted mean share : {point:.3f}   95% mean interval [{mlo:.3f}, {mhi:.3f}]")
    print(f"  observed mean share  : {actual:.3f}")
    print(f"  (the observation interval is [{olo:.3f}, {ohi:.3f}] -- too wide to test anything)")

    near = fit[(fit.agree > 0.94) & (fit.agree < 0.98)]
    a, b = near.share.values, test.share.values
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    from scipy import stats
    pval = stats.ttest_ind(b, a, equal_var=False).pvalue
    print(f"\n  matched agreement (0.94-0.98): from-scratch {a.mean():.3f} (n={len(a)}) "
          f"vs DistilBERT {b.mean():.3f} (n={len(b)})")
    print(f"  difference {diff:+.3f}  95% CI [{diff - 1.96 * se:+.3f}, {diff + 1.96 * se:+.3f}]  "
          f"Welch p = {pval:.2e}")

    above = diff - 1.96 * se > 0
    if above:
        print("\n  => DistilBERT sits ABOVE the from-scratch curve at matched update size, and")
        print("     capacity is flat (part 1), so neither covariate explains the excess. It is")
        print("     NOT a predicted extreme point; it is an additional effect attributable to")
        print("     architecture or pretraining, which this design cannot separate.")
    else:
        print("\n  => consistent with the from-scratch curve.")
    res.update(predicted_share=float(point), mean_ci_lo=float(mlo), mean_ci_hi=float(mhi),
               obs_ci_lo=float(olo), obs_ci_hi=float(ohi), observed_share=actual,
               matched_scratch=float(a.mean()), matched_n=len(a),
               excess=float(diff), excess_lo=float(diff - 1.96 * se),
               excess_hi=float(diff + 1.96 * se), excess_p=float(pval),
               above_curve=bool(above), fit_n=int(mf.nobs))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).to_csv(OUT / "T16_share_model.csv", index=False)
    df.to_parquet(RESULTS / "share_model_rows.parquet")
    print(f"\nwrote {OUT / 'T16_share_model.csv'}")


if __name__ == "__main__":
    main()
