# Warranted and Unwarranted Explanation Change Under Model Updates

Research code for measuring how much of the attribution change caused by a model update is
*warranted* by a change in the underlying predictive mechanism, and how much is not.

The central claim is a measurement design rather than a metric. Prior work measures how far
explanations move after an update and calls the residual instability. That has no reference for how
far they *should* have moved, and no control for how far they would have moved anyway. This
repository supplies both:

- **ω, a warranted-change reference.** Defined against the Bayes-optimal predictor of each
  environment, in closed form on generators where that predictor is known. Exactly zero under
  covariate shift; non-zero and computable under concept shift and shortcut removal.
- **ρ_null, a matched-operator null.** The identical update applied to fresh data from the *source*
  distribution — same learning rate, epochs, step count and sample size. Only the distribution
  differs. Prior work's implicit control is an independent retrain, which is a different operator on
  a different amount of data, and it orders the update regimes in the opposite direction.

## Layout

```
src/uec/
  data/        generators, closed-form Bayes reference, shared-support sampling
  models/      MLP, XGBoost, small ResNet (smooth activations; Prop. 1 assumes C^1)
  train/       source training and the matched update operators
  explain/     explainers and a checkpoint-hashed attribution cache
  metrics/     normalisers, distances, the UEC decomposition, prior-work baselines
  theory/      per-point measurement of Propositions 1-3
  stats/       paired tests, effect sizes, seed bootstrap
  plots/       figure builders
experiments/   runners; each writes a parquet to results/
docs/          spec.md (the contract), theory.md (proofs), lit_matrix.csv, terminology_map.md
tests/         101 tests, including numeric verification of every proposition
```

## Install

```bash
python -m pip install -e .          # every dependency is declared in pyproject.toml
python -m pip install -e ".[dev]"   # + pytest
```

CPU is sufficient. Only the transformer arm needs a GPU (`.[gpu]`), and its measured output is
committed under `results/` because it cannot be regenerated locally.

## Reproduce

```bash
pytest tests/ -q                          # 101 tests, including the proposition checks

python experiments/reproduce.py --list    # the plan, with per-stage timings
python experiments/reproduce.py --quick   # headline result only, ~15 min
python experiments/reproduce.py           # everything, ~4.5 h on 8 CPU cores
```

`reproduce.py` runs all fourteen stages in dependency order, skips any whose output already exists
(so an interrupted run resumes), then rebuilds every table and figure. Individual stages still run
directly — `run_synthetic.py`, `sweep_regime.py`, `run_trees.py`, `run_differentiation.py`,
`run_faithfulness.py`, `run_ablations.py`, `run_budget_sweep.py`, `run_reaudit.py`,
`run_folktables.py`, `run_vision.py`, `run_semisynthetic.py`, `run_redundancy.py`,
`run_adaptation.py`.

Figures and tables are committed, so a fresh clone already has them. `make_tables.py` and
`make_figures.py` **refuse to run** when `results/` is empty rather than silently overwriting the
committed versions with nothing.

Every figure writes its source data beside the image. `results/registry.csv` maps each run to its
regime, seed and checkpoint weight hash, and `checkpoints/` holds the headline models with a
manifest so those hashes can be verified.

## What it found

- **Unwarranted change exists.** Under covariate shift, where the warranted change is exactly zero,
  all seven explainers move more than their matched null (ratios 1.02–1.73, Holm-adjusted p = 0.014,
  Cliff's δ up to +0.98). The no-shift placebo returns 1.00–1.10 with every interval covering 1.
- **The control decides the answer.** As updates get heavier, change relative to the matched null
  *falls* (1.81 → 1.09 for IG at shift 1.5, lr 2e-4) while change relative to the seed floor *rises*
  (0.11 → 1.07). The opposite ordering holds in **24 of 24** explainer × magnitude × lr cells.
- **On a no-shift placebo the seed floor's bias flips sign with the model class.** Nothing shifted at
  all: the matched null reports 1.02 (MLP) and 0.98 (trees, exact TreeSHAP), while the seed floor
  reports **0.32 and 1.90** — a sixfold self-disagreement, so no fixed correction recovers it.
  Against that seed floor our own headline effect is inverted (median 0.31), which we state in §7.3
  rather than leave to be found.
- **Not a gradient artefact.** The effect is largest for trees with exact TreeSHAP (2.08), and holds
  for MLPs, trees and a small CIFAR ResNet across seven attribution methods.
- **But it is bounded.** Flat at ~1.4 across a 630x parameter range within models trained from
  scratch (1.7k to 1.07M), and **absent** in a pretrained DistilBERT at 66.9M: 1.02 [0.97, 1.09],
  unchanged across a tenfold range of update strength. Whether that is scale or pretraining is
  unresolved.
- **Magnitude says little about legitimacy — on the generator.** Per probe point, how far an
  explanation moved explains ~1% of the variance in how far it *should* have moved (r = +0.12
  [0.08, 0.16] under concept shift, −0.10 [−0.19, −0.02] under shortcut removal, 7.5k points). The
  per-explainer version is underpowered at ten seeds and is not read as equivalence; LIME differs.
- **…and that holds only under light updates.** On real ACS covariates with a mechanism known by
  construction the same measurement gives r = +0.81. The cause is not the covariates but how far the
  model travelled: sweeping the update budget on the generator carries r from +0.15 to **+0.882** as
  Δ/ω passes through 1, then back to +0.51 on overshoot. Magnitude becomes informative about
  legitimacy only once the update is heavy enough to move predictions visibly — it fails exactly in
  the regime auditors work in.
- **ω itself transfers.** Exact to 1.1e-14 against quadrature on real rows, and exactly 0 under a
  covariate tilt calibrated to the synthetic shift's domain AUC (0.906 vs 0.902).
- **Two published metrics get it backwards.** Reimplemented on the same checkpoints, a FASS-style
  filtered distance and Delta-Audit's spurious residual rank correct adaptation as 45% and 89%
  *worse* than unwarranted drift. UEC gives them opposite signs.
- **The theory holds.** Proposition 1(i): 0 violations in 20,000 probe points. Proposition 2:
  Grad×Input breaks the same bound on 60.8%.
- **It replicates on real data.** ACS Income, CA → four states: 27 of 28 explainer×state cells
  significant after Holm.

## Reading the numbers

- `delta` — attribution change across the update, on prediction-preserved probe points
- `nu` — the explainer's own noise floor (zero for deterministic explainers)
- `rho_null` — matched-operator null (**the** control)
- `rho_seed` — independent-retrain floor, reported for comparability with prior work
- `omega` — warranted change; exactly zero under covariate shift
- `uec` — `delta − omega − max(nu, rho_null)`
- `ratio` — `delta / rho_null`, the quantity the conclusions turn on

## Caveats the code enforces

- Attributions target the **logit**, never the probability, so theory and reference share a space.
- The attribution baseline is fixed across checkpoints; a per-checkpoint baseline makes ΔIG
  meaningless.
- The activation is smooth by default. On a ReLU network IG completeness holds only to O(1/n_steps)
  and would contaminate the Proposition 1 check.
- Probe points come from the shared support, so neither checkpoint is extrapolating.
- ω is exact only where the generative process is known. On real data only the covariate-shift null
  is claimed, and even then it is *checked* via a calibration-transfer test, not proved.
