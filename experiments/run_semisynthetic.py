"""Does omega behave the same way off the synthetic generator?

The generator supplies both the covariates and the mechanism. That makes omega exact but leaves the
question of whether anything here survives contact with real inputs. Here the covariates are real
ACS rows and only the labels are regenerated, so the mechanism is still known exactly while the
input distribution is not ours to choose.

Three things are checked, in order of what they would cost us if they failed:

  1. the closed form matches quadrature on real rows -- if this breaks, omega is simply wrong
  2. omega is exactly 0 under a covariate tilt, and the ratio still exceeds 1 -- H1 off-generator
  3. per-point measured change barely tracks omega under concept shift -- H2 off-generator

    python experiments/run_semisynthetic.py --seeds 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.data.semisynthetic import (  # noqa: E402
    concept_shift,
    covariate_tilt,
    load_semisynthetic,
)
from uec.explain.cache import attribute  # noqa: E402
from uec.metrics.distances import DISTANCES, PRIMARY  # noqa: E402
from uec.metrics.normalise import NORMALISERS  # noqa: E402
from uec.metrics.uec import change, preserved_mask  # noqa: E402
from uec.models.mlp import probabilities  # noqa: E402
from uec.paths import RESULTS  # noqa: E402
from uec.rng import pin_threads  # noqa: E402
from uec.train.harness import TrainConfig, UpdateConfig, agreement_rate, train_source, update  # noqa: E402

N_SOURCE, N_UPDATE, N_PROBE, EPS = 8000, 4000, 600, 0.05
# Calibrated, not chosen: a cross-fitted domain classifier separates the tilted sample from the
# untilted one at AUC 0.906, matching the synthetic covariate shift's 0.902. At strength 1.0 the
# tilt reaches only 0.759, so a null result there would have measured a weaker shift rather than a
# weaker effect. Sample sizes above are the synthetic defaults, unchanged.
TILT = 2.0
EXPLAINER_SET = ["integrated_gradients", "saliency", "gradient_x_input"]
# Perturb the two features the interactions touch, plus one they do not, so the warranted change is
# not concentrated in a single coordinate.
DELTA_BETA = {0: 0.8, 3: -0.6, 6: 0.5}


def build(Z, y, mech, mech_t, seed, tilt_strength, ucfg=None):
    """Same operator on both arms; only the sampling domain and the mechanism differ."""
    cfg, ucfg = TrainConfig(), ucfg or UpdateConfig()
    rng = np.random.default_rng([seed, 7])

    i_src = rng.choice(len(Z), N_SOURCE, replace=False)
    ys = mech.label(Z[i_src], np.random.default_rng([seed, 10]))
    f0, _ = train_source(Z[i_src], ys, cfg, seed)

    # Both arms draw with replacement. Drawing the null *without* replacement and the tilted
    # treatment *with* it would hand the treatment fewer distinct rows, so the ratio would partly
    # measure effective sample size rather than the shift -- the exact confound this paper is
    # about. Identical procedure, weights the only difference.
    rest = np.setdiff1d(np.arange(len(Z)), i_src)
    i_null = np.random.default_rng([seed, 11]).choice(rest, N_UPDATE, replace=True)
    yn = mech.label(Z[i_null], np.random.default_rng([seed, 12]))
    f_null, mn = update(f0, Z[i_null], yn, ucfg, seed + 1)

    if tilt_strength:
        i_tgt = rest[covariate_tilt(Z[rest], np.random.default_rng([seed, 13]), N_UPDATE,
                                    strength=tilt_strength)]
    else:
        i_tgt = np.random.default_rng([seed, 13]).choice(rest, N_UPDATE, replace=True)
    yt = mech_t.label(Z[i_tgt], np.random.default_rng([seed, 14]))
    f_treat, mt = update(f0, Z[i_tgt], yt, ucfg, seed + 1)

    assert mn["n_steps"] == mt["n_steps"] and mn["n_train"] == mt["n_train"]
    # Distinct-row counts are reported so the residual diversity gap is visible, not assumed away.
    return f0, f_null, f_treat, i_src, len(np.unique(i_null)), len(np.unique(i_tgt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    pin_threads()

    Z, y_real, mech = load_semisynthetic()
    baseline = np.zeros(Z.shape[1])
    norm, dist = NORMALISERS["abs"], DISTANCES[PRIMARY]

    print(f"ACS CA 2018: {len(Z)} real rows, {Z.shape[1]} features")
    print(f"mechanism beta fitted to real labels, |beta| in "
          f"[{np.abs(mech.beta).min():.3f}, {np.abs(mech.beta).max():.3f}]\n")

    print("=" * 72)
    print("1. Is the closed form correct on real rows?")
    print("=" * 72)
    probe = Z[np.random.default_rng(0).choice(len(Z), 2000, replace=False)]
    exact = mech.ig(probe, baseline)
    quad = mech.ig_quadrature(probe, baseline, steps=512)
    err = np.abs(exact - quad).max()
    comp = np.abs(exact.sum(1) - (mech.logodds(probe) - mech.logodds(baseline[None]))).max()
    print(f"max |closed form - quadrature| = {err:.3e}")
    print(f"max completeness residual      = {comp:.3e}")
    assert err < 1e-8 and comp < 1e-8, "closed form disagrees with quadrature on real covariates"
    print("=> exact on real covariates, as on synthetic ones. omega is not a Gaussian artefact.\n")

    rows, perpoint = [], {}
    for family, tilt, mech_t in [("covariate", TILT, mech),
                                 ("concept", 0.0, concept_shift(mech, DELTA_BETA))]:
        t0 = time.time()
        for seed in range(args.seeds):
            f0, f_null, f_treat, i_src, uniq_n, uniq_t = build(
                Z, y_real, mech, mech_t, seed, tilt)

            rp = np.random.default_rng([seed, 20])
            i_probe = rp.choice(np.setdiff1d(np.arange(len(Z)), i_src), N_PROBE, replace=False)
            Xp = Z[i_probe]

            keep = preserved_mask(probabilities(f0, Xp), probabilities(f_treat, Xp), EPS)
            if keep.sum() < 30:
                continue
            Xk = Xp[keep]
            omega = dist(norm(mech.ig(Xk, baseline)), norm(mech_t.ig(Xk, baseline)))

            for name in EXPLAINER_SET:
                # The zero baseline the explainers default to is the same one omega integrates
                # from, which is what makes the two comparable at all.
                A0 = attribute(f0, Xk, name)
                An = attribute(f_null, Xk, name)
                At = attribute(f_treat, Xk, name)
                delta = change(A0, At, norm, dist)
                rho = change(A0, An, norm, dist)
                rows.append(dict(family=family, explainer=name, seed=seed,
                                 delta=delta.mean(), rho_null=rho.mean(),
                                 omega=omega.mean(), n_probe=int(keep.sum()),
                                 agree=agreement_rate(f0, f_treat, Xp),
                                 uniq_null=uniq_n, uniq_treat=uniq_t))
                if name == "integrated_gradients":
                    perpoint[f"{seed}|{family}"] = np.vstack([delta, omega])
        print(f"[{family}] {args.seeds} seeds in {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    df["ratio"] = df.delta / df.rho_null
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "semisynthetic_metrics.parquet")
    np.savez_compressed(RESULTS / "semisynthetic_perpoint.npz", **perpoint)

    print("\n" + "=" * 72)
    print("2. omega = 0 under covariate tilt, and does the ratio still exceed 1?")
    print("=" * 72)
    cov = df[df.family == "covariate"]
    print(f"max omega under covariate tilt = {cov.omega.abs().max():.3e} (must be exactly 0)")
    print(f"distinct update rows: null {cov.uniq_null.mean():.0f}, "
          f"treatment {cov.uniq_treat.mean():.0f}, of {N_UPDATE} draws")
    for name, g in cov.groupby("explainer"):
        r = g.ratio.values
        lo, hi = r.mean() - 1.96 * r.std(ddof=1) / np.sqrt(len(r)), \
            r.mean() + 1.96 * r.std(ddof=1) / np.sqrt(len(r))
        verdict = "above" if lo > 1 else ("below" if hi < 1 else "covers 1")
        print(f"  {name:22s} ratio = {r.mean():.3f} [{lo:.3f}, {hi:.3f}]  {verdict:9s} "
              f"agreement = {g.agree.mean():.3f}")

    print("\n" + "=" * 72)
    print("3. Does measured change track omega, per point, under concept shift?")
    print("=" * 72)
    rs = []
    for k, arr in perpoint.items():
        if not k.endswith("concept"):
            continue
        d, o = arr
        if np.std(o) > 1e-12:
            rs.append(np.corrcoef(d, o)[0, 1])
    r = np.array(rs)
    m, se = r.mean(), r.std(ddof=1) / np.sqrt(len(r))
    print(f"per-seed r = {m:+.3f} [{m - 1.96 * se:+.3f}, {m + 1.96 * se:+.3f}]   "
          f"R^2 = {m ** 2:.3f} ({m ** 2 * 100:.1f}% of variance)   seeds = {len(r)}")
    con = df[df.family == "concept"]
    print(f"mean omega under concept shift = {con.omega.mean():.3f}")
    if abs(m) > 0.3:
        print("\n  This is the opposite of the synthetic result (r = +0.12, 1.5% of variance), so")
        print("  it needs a cause rather than a footnote. The update configuration is identical to")
        print("  the synthetic one, so update strength cannot be the explanation on its own --")
        print("  testing it directly rather than assuming:")
        sweep = []
        for ep in (2, 20, 100):
            ucfg = UpdateConfig(epochs=ep)
            rr = []
            for seed in range(min(args.seeds, 5)):
                f0, _, f_treat, i_src, _, _ = build(
                    Z, y_real, mech, concept_shift(mech, DELTA_BETA), seed, 0.0, ucfg)
                rp = np.random.default_rng([seed, 20])
                ip = rp.choice(np.setdiff1d(np.arange(len(Z)), i_src), N_PROBE, replace=False)
                keep = preserved_mask(probabilities(f0, Z[ip]),
                                      probabilities(f_treat, Z[ip]), EPS)
                if keep.sum() < 30:
                    continue
                Xk = Z[ip][keep]
                o = dist(norm(mech.ig(Xk, baseline)),
                         norm(concept_shift(mech, DELTA_BETA).ig(Xk, baseline)))
                d = change(attribute(f0, Xk, "integrated_gradients"),
                           attribute(f_treat, Xk, "integrated_gradients"), norm, dist)
                if np.std(o) > 1e-12:
                    rr.append(np.corrcoef(d, o)[0, 1])
            sweep.append((ep, float(np.mean(rr)), len(rr)))
            print(f"    update epochs {ep:3d}:  r = {np.mean(rr):+.3f}  (n = {len(rr)} seeds)")
        pd.DataFrame(sweep, columns=["update_epochs", "r", "seeds"]).to_csv(
            RESULTS / "semisynthetic_strength_sweep.csv", index=False)
        spread = max(s[1] for s in sweep) - min(s[1] for s in sweep)
        print(f"    r moves {spread:.3f} across a 50x range of update strength "
              f"=> {'update strength explains it' if spread > 0.3 else 'NOT update strength'}")

    out = Path(__file__).resolve().parents[1] / "paper" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    summ = df.groupby(["family", "explainer"]).agg(
        delta=("delta", "mean"), rho_null=("rho_null", "mean"),
        omega=("omega", "mean"), ratio=("ratio", "mean"), agree=("agree", "mean")).reset_index()
    summ["quadrature_err"] = err
    summ["per_point_r"] = m
    summ.to_csv(out / "T18_semisynthetic.csv", index=False)
    print(f"\nwrote {out / 'T18_semisynthetic.csv'}")


if __name__ == "__main__":
    main()
