"""Builds every table in the paper from the result parquets. Missing inputs are skipped with a
note rather than silently producing an empty table."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.paths import RESULTS, ROOT  # noqa: E402
from uec.plots.style import label  # noqa: E402
from uec.stats.inference import (  # noqa: E402
    bootstrap_ci,
    cliffs_delta,
    holm,
    paired_test,
    ranking_agreement,
    ratio_ci,
)

OUT = ROOT / "paper" / "tables"
PRIMARY = dict(distance="l1", phi="abs", features="all", eps=0.05)


def _load(name):
    p = RESULTS / name
    return pd.read_parquet(p) if p.exists() else None


def _primary(df, **over):
    sel = {**PRIMARY, **over}
    m = np.ones(len(df), bool)
    for k, v in sel.items():
        if k in df:
            m &= df[k] == v
    return df[m]


def table_grid(syn, folk, vis, sweep):
    rows = []
    for name, df, dataset in (("synthetic", syn, "synthetic (SEM, d=20)"),
                              ("folktables", folk, "ACS Income"),
                              ("vision", vis, "CIFAR-10")):
        if df is None or df.empty:
            continue
        rows.append({
            "dataset": dataset,
            "shift families": ", ".join(sorted(df.family.unique())),
            "explainers": len(df.explainer.unique()),
            "seeds": df.seed.nunique(),
            "distances": len(df.distance.unique()),
            "rows": len(df),
        })
    if sweep is not None and not sweep.empty:
        rows.append({
            "dataset": "synthetic (regime sweep)",
            "shift families": "covariate",
            "explainers": len(sweep.explainer.unique()),
            "seeds": sweep.seed.nunique(),
            "distances": 1,
            "rows": len(sweep),
        })
    return pd.DataFrame(rows)


def table_uec(syn):
    q = _primary(syn)
    rows = []
    for (family, name), s in q.groupby(["family", "explainer"]):
        d, dlo, dhi = bootstrap_ci(s.delta.values)
        r, rlo, rhi = ratio_ci(s.delta.values, s.rho_null.values)
        u, ulo, uhi = bootstrap_ci(s.uec.values)
        rows.append({
            "shift": family, "explainer": label(name),
            "nu": s.nu.mean(), "rho_null": s.rho_null.mean(), "rho_seed": s.rho_seed.mean(),
            "omega": s.omega.mean(), "delta": d, "delta_lo": dlo, "delta_hi": dhi,
            "ratio": r, "ratio_lo": rlo, "ratio_hi": rhi,
            "ratio_seed": s.ratio_seed.mean(),
            "uec": u, "uec_lo": ulo, "uec_hi": uhi,
            "exceedance": s.exceedance.mean(),
            "preserved": s.preserved_frac.mean(),
            "n_seeds": s.seed.nunique(),
        })
    return pd.DataFrame(rows).sort_values(["shift", "explainer"])


def table_significance(syn):
    """H1: paired test of shift-induced change against the matched null, Holm-corrected."""
    q = _primary(syn)
    rows = []
    for family, fam in q.groupby("family"):
        names, ps, ds = [], [], []
        for name, s in fam.groupby("explainer"):
            if s.seed.nunique() < 5:
                continue
            names.append(name)
            ps.append(paired_test(s.delta.values, s.rho_null.values))
            ds.append(cliffs_delta(s.delta.values, s.rho_null.values))
        if not names:
            continue
        for name, p, adj, d in zip(names, ps, holm(ps), ds):
            rows.append({"shift": family, "explainer": label(name), "p": p,
                         "p_holm": adj, "cliffs_delta": d,
                         "significant": bool(adj < 0.05)})
    return pd.DataFrame(rows)


def table_invisibility(syn):
    """H3: is unwarranted change predicted by the metrics practitioners watch?"""
    q = _primary(syn, family="covariate")
    rows = []
    for name, s in q.groupby("explainer"):
        rec = {"explainer": label(name), "n": len(s)}
        for col in ("acc_treat_tgt", "ece_treat", "agree_treat", "acc_source_src"):
            if col in s and s[col].std() > 1e-12 and s.delta.std() > 1e-12:
                rec[col] = float(np.corrcoef(s[col], s.delta)[0, 1])
            else:
                rec[col] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def table_differentiation(diff):
    cols = ["fass_distance", "delta_audit_jsd", "delta_audit_spurious", "ros_mean", "ros_max",
            "omega", "uec", "ratio"]
    return diff.groupby(["family", "update_epochs", "explainer"])[cols].mean().reset_index()


def table_ablations(syn):
    rows = []
    base = _primary(syn, family="covariate")
    ranking = base.groupby("explainer")["ratio"].mean().to_dict()

    for axis, values in (("distance", syn.distance.unique()),
                         ("phi", syn.phi.unique()),
                         ("eps", sorted(syn.eps.unique())),
                         ("features", syn.features.unique())):
        for v in values:
            q = _primary(syn, family="covariate", **{axis: v})
            if q.empty:
                continue
            other = q.groupby("explainer")["ratio"].mean().to_dict()
            rows.append({
                "axis": axis, "value": v,
                "mean_ratio": float(q.ratio.mean()),
                "kendall_tau_vs_primary": ranking_agreement(ranking, other),
                "n_rows": len(q),
            })
    return pd.DataFrame(rows)


def table_theory(theory):
    return pd.DataFrame([{
        "family": fam,
        "IG max slack": s.ig_slack_max.max(),
        "IG violations": int(s.ig_violations.sum()),
        "IG quad tol": s.ig_quad_tol.max(),
        "GxI median slack": s.gi_slack_median.median(),
        "GxI frac > 1": s.gi_exceeds_one.mean(),
        "GxI frac > 1 (preserved)": s.gi_exceeds_one_preserved.mean(),
        "coalition ratio (median)": s.coal_ratio_median.median(),
    } for fam, s in theory.groupby("family")])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    syn = _load("synthetic_metrics.parquet")
    folk = _load("folktables_metrics.parquet")
    vis = _load("vision_metrics.parquet")
    sweep = _load("sweep_regime.parquet")
    diff = _load("differentiation.parquet")
    theory = _load("synthetic_theory.parquet")

    built = {}
    if syn is not None:
        built["T1_grid"] = table_grid(syn, folk, vis, sweep)
        built["T2_uec"] = table_uec(syn)
        built["T3_significance"] = table_significance(syn)
        built["T3b_invisibility"] = table_invisibility(syn)
        built["T5_ablations"] = table_ablations(syn)
    if diff is not None:
        built["T4_differentiation"] = table_differentiation(diff)
    if theory is not None:
        built["T7_theory"] = table_theory(theory)
    if folk is not None:
        built["T8_folktables"] = table_uec(folk)
    if vis is not None:
        built["T9_vision"] = table_uec(vis)

    md = ["# Tables\n"]
    for name, t in built.items():
        t.to_csv(OUT / f"{name}.csv", index=False)
        md.append(f"\n## {name}\n")
        md.append(t.round(4).to_markdown(index=False))
        md.append("")
        print(f"{name}: {len(t)} rows")
    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {OUT / 'tables.md'}")


if __name__ == "__main__":
    main()
