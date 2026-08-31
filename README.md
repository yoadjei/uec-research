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
tests/         89 tests, including numeric verification of every proposition
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
pytest tests/ -q                          # 97 tests, including the proposition checks

python experiments/reproduce.py --list    # the plan, with per-stage timings
python experiments/reproduce.py --quick   # headline result only, ~15 min
python experiments/reproduce.py           # everything, ~4.5 h on 8 CPU cores
```

`reproduce.py` runs all twelve stages in dependency order, skips any whose output already exists
(so an interrupted run resumes), then rebuilds every table and figure. Individual stages still run
directly — `run_synthetic.py`, `sweep_regime.py`, `run_trees.py`, `run_differentiation.py`,
`run_faithfulness.py`, `run_ablations.py`, `run_budget_sweep.py`, `run_reaudit.py`,
`run_folktables.py`, `run_vision.py`.

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
  *falls* (1.81 → 1.09) while change relative to the seed floor *rises* (0.11 → 1.07). Prior work's
  implicit control orders the regimes backwards.
- **On a no-shift placebo the seed floor invents an effect.** With gradient-boosted trees and exact
  TreeSHAP, and nothing shifted at all, the matched null reports 0.98 (not significant) while the
  seed floor reports 1.90.
- **Not a gradient artefact.** The effect is largest for trees with exact TreeSHAP (2.08), and holds
  for MLPs, trees and a small CIFAR ResNet across seven attribution methods.
- **But it is bounded.** Flat at ~1.4 across a 630x parameter range within models trained from
  scratch (1.7k to 1.07M), and **absent** in a pretrained DistilBERT at 66.9M: 1.02 [0.97, 1.09],
  unchanged across a tenfold range of update strength. Whether that is scale or pretraining is
  unresolved.
- **Magnitude says nothing about legitimacy.** For six of seven explainers, change relative to the
  matched null is statistically indistinguishable between a shift that leaves the Bayes-optimal
  predictor untouched and one that rewrites it (IG: p = 0.56).
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
