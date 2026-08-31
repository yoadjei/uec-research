"""Check the paper's headline numbers against the produced artefacts.

"Reproducible" means more than "the code runs". Every quantitative claim in the paper should be
recoverable from a file in results/ or paper/tables/. This re-derives each headline number from
the data and compares it to what the text asserts, so a stale number cannot survive a rerun.

    python experiments/verify_paper.py

Exit code 1 if any claim fails.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uec.paths import RESULTS, ROOT  # noqa: E402

TABLES = ROOT / "paper" / "tables"
PRIMARY = dict(distance="l1", phi="abs", features="all", eps=0.05)


def _primary(df, **over):
    sel = {**PRIMARY, **over}
    m = np.ones(len(df), bool)
    for k, v in sel.items():
        if k in df:
            m &= df[k] == v
    return df[m]


class Check:
    def __init__(self):
        self.rows = []

    def __call__(self, claim, actual, expected, tol=0.02, source=""):
        if actual is None:
            self.rows.append((claim, "SKIP", "input absent", source))
            return
        ok = abs(float(actual) - float(expected)) <= tol
        self.rows.append((claim, "ok" if ok else "MISMATCH",
                          f"paper {expected} vs data {float(actual):.4f}", source))

    def report(self):
        w = max(len(r[0]) for r in self.rows)
        bad = 0
        for claim, status, detail, source in self.rows:
            flag = {"ok": "  ", "SKIP": "~ ", "MISMATCH": "! "}[status]
            print(f"{flag}{claim:{w}s}  {detail:34s} {source}")
            bad += status == "MISMATCH"
        skipped = sum(r[1] == "SKIP" for r in self.rows)
        print(f"\n{len(self.rows) - bad - skipped} verified, {skipped} skipped, {bad} mismatched")
        return bad


def load(name, folder=RESULTS):
    p = folder / name
    if not p.exists():
        return None
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def main():
    c = Check()
    syn = load("synthetic_metrics.parquet")

    if syn is not None:
        q = _primary(syn, family="covariate")
        for name, expected in [("saliency", 1.73), ("smoothgrad", 1.73),
                               ("gradient_x_input", 1.67), ("integrated_gradients", 1.52),
                               ("kernel_shap", 1.40), ("lime", 1.25), ("expected_gradients", 1.02)]:
            s = q[q.explainer == name]
            c(f"7.4 H1 ratio, {name}", s.delta.mean() / s.rho_null.mean(), expected,
              source="T2_uec")

        p = _primary(syn, family="none")
        c("7.1 placebo max ratio", (p.groupby("explainer").delta.mean()
                                    / p.groupby("explainer").rho_null.mean()).max(), 1.10,
          tol=0.05, source="T2_uec")

        cov = _primary(syn, family="covariate", explainer="integrated_gradients")
        sc = _primary(syn, family="shortcut", explainer="integrated_gradients")
        c("7.5 IG delta, covariate", cov.delta.mean(), 0.0398, tol=0.003, source="T2_uec")
        c("7.5 IG delta, shortcut", sc.delta.mean(), 0.0386, tol=0.003, source="T2_uec")
        c("7.5 omega, shortcut", sc.omega.mean(), 0.326, tol=0.01, source="T2_uec")
        c("7.5 omega, covariate", cov.omega.mean(), 0.0, tol=1e-9, source="T2_uec")

    tr = load("trees_metrics.parquet")
    if tr is not None:
        t = _primary(tr, family="none")
        c("7.2 tree placebo, matched null", t.delta.mean() / t.rho_null.mean(), 0.98,
          tol=0.03, source="T10_trees")
        c("7.2 tree placebo, SEED floor", t.delta.mean() / t.rho_seed.mean(), 1.90,
          tol=0.06, source="T10_trees")
        tc = _primary(tr, family="covariate")
        c("7.2 tree covariate ratio", tc.delta.mean() / tc.rho_null.mean(), 2.08,
          tol=0.05, source="T10_trees")

    th = load("synthetic_theory.parquet")
    if th is not None:
        c("7.8 IG bound violations", th.ig_violations.sum(), 0, tol=0, source="T7_theory")
        c("7.8 GxI exceeds bound", th.gi_exceeds_one.mean(), 0.608, tol=0.01, source="T7_theory")
        by_fam = th.groupby("family").coal_ratio_median.mean()
        c("7.8 coalition ratio, min family", by_fam.min(), 1.49, tol=0.02,
          source="T7_theory")
        c("7.8 coalition ratio, max family", by_fam.max(), 2.02, tol=0.02,
          source="T7_theory")

    v = load("reaudit_verdicts.parquet")
    if v is not None:
        c("7.7 re-audit clears floor", v.clears_floor.sum(), 15, tol=0, source="T11_reaudit")
        c("7.7 re-audit below floor", (v.ratio < 1).sum(), 29, tol=0, source="T11_reaudit")
        c("7.7 re-audit median ratio", v.ratio.median(), 0.861, tol=0.01, source="T11_reaudit")

    fa = load("faithfulness.parquet")
    if fa is not None:
        f = fa[(fa.family == "covariate") & (fa.explainer == "integrated_gradients")]
        c("7.9 E6 ratio, all points", f.ratio_all.mean(), 1.563, tol=0.01, source="faithfulness")
        c("7.9 E6 ratio, faithful only", f.ratio_faithful.mean(), 1.567, tol=0.01,
          source="faithfulness")
        cell = fa.groupby(["family", "explainer"]).corr_delta_faith.mean()
        c("7.9 E6 |corr| max, seed-averaged", cell.abs().max(), 0.084, tol=0.005,
          source="faithfulness")
        c("7.9 E6 |corr| max, per seed", fa.corr_delta_faith.abs().max(), 0.389,
          tol=0.005, source="faithfulness")

    w = load("scale_width.parquet")
    if w is not None:
        g = w[w.explainer == "integrated_gradients"].groupby("width")
        c("8.1 width 32 ratio", g.get_group(32).ratio.mean(), 1.573, tol=0.02, source="T14")
        c("8.1 width 1024 ratio", g.get_group(1024).ratio.mean(), 1.360, tol=0.02, source="T14")
        c("8.1 width 1024 params", w[w.width == 1024].n_params.iloc[0], 1_070_080, tol=0,
          source="T14")

    sw = load("scale_text_sweep_ig.csv")
    if sw is not None:
        c("7.13 DistilBERT pooled ratio", sw.delta.sum() / sw.rho_null.sum(), 1.023, tol=0.01,
          source="scale_text_sweep_ig")
        c("7.13 DistilBERT agreement", sw.agree_treat.mean(), 0.959, tol=0.005,
          source="scale_text_sweep_ig")

    ab = load("T5_ablations.csv", TABLES)
    if ab is not None:
        core = ab[ab.value.isin(["spearman", "cosine", "l1", "abs", "all", "reliable"])]
        c("8 ablation Kendall tau (min over core axes)",
          core.kendall_tau_vs_primary.min(), 1.0, tol=1e-9, source="T5_ablations")

    bad = c.report()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
