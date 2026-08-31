"""Does unwarranted explanation change need redundant features to live in?

On real ACS covariates the effect vanishes: the ratio covers 1 and per-point measured change tracks
omega closely (r = 0.81). On the synthetic generator it does not. The two settings differ in many
ways, but one difference is structural and measurable -- the generator carries a block of near-copies
of its informative features (causal-redundant correlation 0.94) and ACS does not (largest
off-diagonal correlation 0.48).

If that is the cause, then adding collinear copies to the *same real covariates* should bring the
effect back. The copies carry zero mechanism weight, so omega is unchanged by construction; all they
supply is a direction in which attribution mass can move without moving predictions.

This is a single manipulated variable with a prediction made before running it, which is the only
form in which the explanation is worth anything.

    python experiments/run_redundancy.py --seeds 8
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.data.semisynthetic import concept_shift, load_semisynthetic  # noqa: E402
from uec.explain.cache import attribute  # noqa: E402
from uec.metrics.distances import DISTANCES, PRIMARY  # noqa: E402
from uec.metrics.normalise import NORMALISERS  # noqa: E402
from uec.metrics.uec import change, preserved_mask  # noqa: E402
from uec.models.mlp import probabilities  # noqa: E402
from uec.paths import RESULTS  # noqa: E402
from uec.rng import pin_threads  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_semisynthetic import DELTA_BETA, EPS, N_PROBE, TILT, build  # noqa: E402

# sigma is the noise added to each copy: smaller sigma means a tighter copy. 0.35 reproduces the
# synthetic generator's redundancy exactly; the larger values interpolate down to none at all.
# (copies, sigma, noise features). Two hypotheses, one manipulation each: collinear directions to
# hide in, versus uninformative directions to wander into. The last row rebuilds the synthetic
# generator's structure -- 4 tight copies plus 8 noise features -- on top of real covariates.
GRID = [(0, 0.0, 0), (4, 1.20, 0), (4, 0.35, 0), (0, 0.0, 8), (4, 0.35, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    pin_threads()

    norm, dist = NORMALISERS["abs"], DISTANCES[PRIMARY]
    rows = []

    print("prediction, fixed before running: if redundancy is the cause, the covariate ratio")
    print("rises above 1 and the per-point correlation falls as the copies get tighter.\n")
    print(f"{'copies':>6} {'sigma':>6} {'noise':>6} {'max|corr|':>10} {'ratio (cov)':>22} "
          f"{'r (concept)':>13}")
    print("-" * 74)

    for n_red, sigma, n_noise in GRID:
        t0 = time.time()
        Z, y_real, mech = load_semisynthetic(n_redundant=n_red, sigma_r=sigma,
                                             n_noise=n_noise)
        baseline = np.zeros(Z.shape[1])
        C = np.corrcoef(Z.T)
        maxcorr = np.abs(C[~np.eye(len(C), dtype=bool)]).max()

        ratios, rs, agrees = [], [], []
        for seed in range(args.seeds):
            # covariate tilt: omega is exactly 0, so any movement above the null is unwarranted
            f0, f_null, f_treat, i_src, _, _ = build(Z, y_real, mech, mech, seed, TILT)
            rp = np.random.default_rng([seed, 20])
            ip = rp.choice(np.setdiff1d(np.arange(len(Z)), i_src), N_PROBE, replace=False)
            keep = preserved_mask(probabilities(f0, Z[ip]), probabilities(f_treat, Z[ip]), EPS)
            if keep.sum() >= 30:
                Xk = Z[ip][keep]
                d = change(attribute(f0, Xk, "integrated_gradients"),
                           attribute(f_treat, Xk, "integrated_gradients"), norm, dist)
                r_ = change(attribute(f0, Xk, "integrated_gradients"),
                            attribute(f_null, Xk, "integrated_gradients"), norm, dist)
                ratios.append(d.mean() / r_.mean())

            # concept shift: omega varies per point, so the correlation is measurable
            mech_t = concept_shift(mech, DELTA_BETA)
            f0, _, f_treat, i_src, _, _ = build(Z, y_real, mech, mech_t, seed, 0.0)
            ip = np.random.default_rng([seed, 20]).choice(
                np.setdiff1d(np.arange(len(Z)), i_src), N_PROBE, replace=False)
            keep = preserved_mask(probabilities(f0, Z[ip]), probabilities(f_treat, Z[ip]), EPS)
            if keep.sum() >= 30:
                Xk = Z[ip][keep]
                o = dist(norm(mech.ig(Xk, baseline)), norm(mech_t.ig(Xk, baseline)))
                d = change(attribute(f0, Xk, "integrated_gradients"),
                           attribute(f_treat, Xk, "integrated_gradients"), norm, dist)
                if np.std(o) > 1e-12:
                    rs.append(np.corrcoef(d, o)[0, 1])

        ra, rr = np.array(ratios), np.array(rs)
        rlo = ra.mean() - 1.96 * ra.std(ddof=1) / np.sqrt(len(ra))
        rhi = ra.mean() + 1.96 * ra.std(ddof=1) / np.sqrt(len(ra))
        print(f"{n_red:>6} {sigma:>6.2f} {n_noise:>6} {maxcorr:>10.3f} "
              f"{ra.mean():>10.3f} [{rlo:.3f}, {rhi:.3f}] {rr.mean():>+13.3f}"
              f"  ({time.time() - t0:.0f}s)")
        rows.append(dict(n_redundant=n_red, sigma=sigma, n_noise=n_noise, max_corr=maxcorr,
                         ratio=ra.mean(), ratio_lo=rlo, ratio_hi=rhi,
                         per_point_r=rr.mean(), seeds=args.seeds))

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "redundancy_sweep.parquet")
    out = Path(__file__).resolve().parents[1] / "paper" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "T19_redundancy.csv", index=False)

    print("\n" + "=" * 72)
    base, tight = df.iloc[0], df.iloc[-1]
    print(f"ratio      {base.ratio:.3f} -> {tight.ratio:.3f}   "
          f"(bare real covariates -> {int(tight.n_redundant)} copies + "
          f"{int(tight.n_noise)} noise features)")
    print(f"per-point r {base.per_point_r:+.3f} -> {tight.per_point_r:+.3f}")
    if tight.ratio_lo > 1 and tight.per_point_r < base.per_point_r - 0.2:
        print("\nBoth move as predicted. Feature redundancy is sufficient to produce unwarranted")
        print("explanation change on real covariates that otherwise show none, and it does so")
        print("without touching omega. That is a mechanism, not a correlation.")
    else:
        print("\nThe prediction is not borne out. Redundancy alone does not account for the gap")
        print("between the synthetic and semi-synthetic settings, and the paper must say so.")
    print(f"\nwrote {out / 'T19_redundancy.csv'}")


if __name__ == "__main__":
    main()
