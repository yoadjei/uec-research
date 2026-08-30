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
  models/      MLP and small ResNet (smooth activations; Prop. 1 assumes C^1)
  train/       source training and the matched update operators
  explain/     explainers and a checkpoint-hashed attribution cache
  metrics/     normalisers, distances, the UEC decomposition, prior-work baselines
  theory/      per-point measurement of Propositions 1-3
  stats/       paired tests, effect sizes, seed bootstrap
  plots/       figure builders
experiments/   runners; each writes a parquet to results/
docs/          spec.md (the contract), theory.md (proofs), lit_matrix.csv, terminology_map.md
tests/         74+ tests, including numeric verification of every proposition
```

## Install

```bash
python -m pip install -e .
python -m pip install torch captum shap lime xgboost folktables quantus torchvision \
                      scikit-learn statsmodels pandas pyarrow matplotlib
```

CPU is sufficient; nothing here needs a GPU.

## Reproduce

```bash
pytest tests/ -q                       # includes the proposition checks

python experiments/run_synthetic.py    # E0-E7: floors, treatments, theory
python experiments/sweep_regime.py     # shift magnitude x update strength
python experiments/run_differentiation.py   # what ROS / FASS / Delta-Audit report instead
python experiments/run_folktables.py   # ACS state shifts
python experiments/run_vision.py       # CIFAR-10 sanity check

python experiments/make_tables.py
python experiments/make_figures.py
```

The headline figure alone:

```bash
python experiments/run_synthetic.py --seeds 5 --families none covariate --tag synthetic
python experiments/make_figures.py
# -> figures/fig2_headline.png (+ fig2_headline.csv, the numbers behind it)
```

Every figure writes its source data beside the image. `results/registry.csv` maps each run to its
regime, seed and checkpoint weight hash.

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
