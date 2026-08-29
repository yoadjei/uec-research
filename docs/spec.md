# Specification: Warranted and Unwarranted Explanation Change

Formal contract for the implementation. A second reader should be able to reimplement UEC from this
file alone.

## 1. Objects

| Symbol | Definition |
|---|---|
| `f: X → R` | the **explained scalar output** — the logit (log-odds of class 1). Everything is defined on logits, matching the theory and the analytic Bayes reference. |
| `E_f: X → R^d` | an explainer applied to `f`; possibly stochastic with law `P_E` |
| `e ∈ {S, T}` | environment (source, target) |
| `h_e(x)` | **Bayes-optimal logit** in environment `e`: `logit P_e(Y = 1 | X = x)` |
| `U` | update operator; `f_{t+1} = U(f_t, D)` for a training set `D` |
| `P` | probe set, drawn from the **shared support** of `P_S(X)` and `P_T(X)` |
| `b` | attribution baseline, fixed across checkpoints and environments |

## 2. The category error, stated once

For a frozen `f` and deterministic `E`, `E_f(x)` is a fixed function of `x`. Changing `P(X)` does not
change `E_f(x)` at any `x`; it changes only which `x` are observed. Instance-level explanation change
under pure data shift with a frozen model is therefore **identically zero**, and the only measurable
object is the shift of the explanation *distribution* — which is Mougan et al.'s explanation shift.

Genuine instance-level change requires exactly one of:
1. the model changed (`f_t → f_{t+1}`),
2. the input changed (`x → T(x)`),
3. the explainer is stochastic.

This work studies (1). (2) is owned by FASS and the adversarial-fragility line. (3) is a nuisance we
measure and subtract as `ν_E`. Experiment E8 demonstrates the vacuity of the frozen-model case
empirically so the point is made with evidence rather than assertion.

## 3. Shift typology

Generated, never assumed:

| Family | Formal condition | Environment change |
|---|---|---|
| **Covariate** | `P_S(Y\|X) = P_T(Y\|X)`, `P_S(X) ≠ P_T(X)` | mean/covariance of `X_C`, `X_N` |
| **Concept** | `P_S(Y\|X) ≠ P_T(Y\|X)` | coefficients of `g` on a known feature subset |
| **Shortcut removal** | `P_S(Y\|X) ≠ P_T(Y\|X)` via a feature that was predictive but never causal | `S₀ ⟂ Y` in target |
| **Label** | `P(X\|Y)` fixed, `P(Y)` changes | not used as a treatment; noted for completeness |

## 4. The warranted-change reference ω

**Departure D1 from the audit.** ω is referenced to the **Bayes-optimal predictor per environment**,
not the causal generating mechanism.

```
ω(x) = d( φ(IG[h_S](x, b)), φ(IG[h_T](x, b)) )
ω_S  = mean over the probe set
```

Justification: the object of study is what a *correctly behaving model* should do. A correctly
behaving model approximates `h_e`, not the causal mechanism. Under shortcut removal the causal
mechanism `g(X_C)` is unchanged, yet a correct model *must* stop using `S₀` — so a causal reference
would label that change unwarranted, contradicting the audit's own §9 table. The Bayes reference
gives the intended verdicts:

| Family | `h` changes? | ω | Verdict on a matching Δ |
|---|---|---|---|
| Covariate | no | **exactly 0** | any change is unwarranted |
| Concept | yes | closed form, > 0 | change aligned with ω is warranted |
| Shortcut removal | yes (the `S₀` term vanishes) | closed form, > 0 | change toward causal features is warranted |

ω is exact only on synthetic data. On real data we report UEC **only for covariate-shift updates**,
where `ω = 0` is the defensible null; for real concept shift we report Δ and decline to label it.

## 5. Probe set and prediction preservation

`P` is drawn from the shared support so that both checkpoints are in-distribution on it. Synthetic:
exact, by rejection on the closed-form log density ratio, `|log p_T(x) − log p_S(x)| ≤ τ`. Real data:
approximate, by a calibrated domain classifier with propensity clipping — documented as an
approximation, not presented as exact.

```
P_ε = { x ∈ P : |p_t(x) − p_{t+1}(x)| ≤ ε },     ε ∈ {0.01, 0.02, 0.05, 0.10}
```

`p` is the predicted probability. Preservation is *not* required to hold anywhere else; in
particular it is not assumed on masked coalition inputs, which is the subject of §8 Proposition 3.
Report `|P_ε| / |P|` for every ε. If `P_ε` is small, widen ε and say so.

## 6. Normalisation, distances, and the estimator

`φ` (normalisation, primary): `a ↦ |a| / ‖a‖₁`. Signed variant `a ↦ a / ‖a‖₁` reported separately.
Both are invariant to positive rescaling of attributions, which makes cross-explainer comparison of
*floors-relative* quantities legitimate.

Distances, all mapped to `[0, 1]`, larger = more different:

| id | definition |
|---|---|
| `spearman` | `(1 − ρ_s(a, a')) / 2` |
| `topk` | `1 − Jaccard(top-k(a), top-k(a'))`, `k = 5` |
| `cosine` | `(1 − cos(φ(a), φ(a'))) / 2` |
| `l1` | `½‖φ(a) − φ(a')‖₁` |

`spearman` is primary (comparable with RSP and FASS); the other three are the required ablation.

```
Δ_E(x)      = d( φ(E_{f_t}(x)), φ(E_{f_{t+1}}(x)) )                     x ∈ P_ε
ν_E         = mean_x d( φ(E_{f_t}^{(1)}(x)), φ(E_{f_t}^{(2)}(x)) )       two runs, same checkpoint
ρ_null,E    = mean_x d( φ(E_{f_t}(x)), φ(E_{U(f_t, D_S')}(x)) )          matched operator, fresh SOURCE data
ρ_seed,E    = mean_x d( φ(E_{f_t}(x)), φ(E_{f_t^{(s)}}(x)) )             independent retrain on source

UEC_E(S)    = mean_{x∈P_ε}[Δ_E(x)] − ω_S − max(ν_E, ρ_null,E)
ratio_E(S)  = mean Δ_E(S) / ρ_null,E
exceed_E(S) = |{ x ∈ P_ε : Δ_E(x) > q₉₅(floor pool) }| / |P_ε|
```

`floor pool` = the pooled per-point distribution of the ν and ρ_null samples for that explainer.
`exceed` is scale-free and comparable across explainers; `ratio` is the number the paper turns on.

**Departure D2 from the audit.** `ρ_null` is the primary floor. The audit's floor applies a
*from-scratch retrain* as the control for a *fine-tune* treatment — different operators on different
amounts of data, so the comparison confounds operator with distribution. `ρ_null` applies the
identical operator (same learning rate, epochs, step count, and training-set size) to a fresh draw
from the **source** distribution. Only the distribution differs between treatment and control.
`ρ_seed` is retained and reported because it is what prior work (EvoXplain, Laberge et al.) measures.

### Estimator properties and known pathologies

- Invariant to positive rescaling of attributions (via `φ`); permutation-equivariant.
- Cost: 2 explanations per probe point per update, plus 2 per point per floor.
- If `P_ε` is small, `Δ` is estimated on few points — report `|P_ε|` beside every number.
- If `ρ_null,E` is large, UEC is uninformative for that explainer. **That is a finding about the
  explainer, not a failure of the estimator**, and it is reported as such.
- `ω` is exact only where the generative process is known.
- UEC is a *difference*, deliberately not a ratio: a ratio to output change (ROS) is undefined and
  explosive exactly in the prediction-preserving regime we care about.

## 7. Feature partition under collinearity

Following the Attribution Impossibility (arXiv:2605.21492), rankings between collinear features are
unreliable independently of any shift. We compute their Z-statistic per feature pair on the probe
set and partition features into `Z ≥ 1.96` (reliable) and `Z < 1.96` (unreliable). All headline
results are reported on the full feature set **and** on the reliable partition. Our redundant block
`R` is collinear with `C` by construction and is expected to fall in the unreliable partition; the
claim survives only if unwarranted change persists on the reliable partition.

## 8. Theory (proofs in `theory.md`)

With `δ = f_t − f_{t+1}`, baseline `b`, `u = x − b`:

- **P1(i)** `Σ_j ΔIG_j(x) = δ(x) − δ(b)` exactly, hence `|Σ_j ΔIG_j(x)| ≤ 2ε`.
- **P1(ii)** `‖ΔIG(x)‖₁ ≤ ‖u‖₁ · sup_γ‖∇δ‖_∞`.
- **P1(iii)** Sharpness: output preservation alone does **not** bound individual IG components.
- **P2** For gradient×input and saliency, not even the aggregate is bounded.
- **P3** `‖Δφ^Shapley‖_∞ ≤ 2ε_coal` if all coalition values are preserved within `ε_coal`.
- **Remark** `ε_coal` is a sup over off-manifold masked inputs and is *not* controlled by data-level
  preservation; we measure it.

**Departure D3/D4 from the audit.** The audit's Proposition 1 claims IG *inherits output stability*.
It does not — only the aggregate does. The real asymmetry, and the one we prove and test, is:

> Completeness makes the **aggregate attribution mass** of path-integrated explainers inherit output
> stability; local-gradient explainers inherit nothing, not even in aggregate. **Neither inherits
> stability of the allocation across features** — which is what practitioners actually read.

## 8b. Update strength is a primary axis, not an ablation (D5)

The null and the treatment apply the *same* operator, so both contain the same optimisation
transient; only the treatment additionally contains the distribution signal. The ratio therefore
depends on update strength in a way that is itself a result, and the pilot sweep confirms it:

| update | `Δ/ρ_null` | `Δ/ρ_seed` | prediction agreement |
|---|---|---|---|
| 1 epoch, lr 2e-4 | **≈ 3.1** | ≈ 0.5 | 0.95 |
| 5 epochs, lr 5e-4 | ≈ 1.6 | ≈ 0.7 | 0.92 |
| 150 epochs, lr 2e-3 | ≈ 1.0 | **≈ 2.3** | 0.84 |

Two things follow, and both are load-bearing.

1. **Light updates isolate the shift.** The gentler the update, the larger the ratio *and* the
   better predictions are preserved — the two desiderata align rather than trade off. Light
   incremental fine-tuning is also the realistic deployment regime.
2. **The choice of control reverses the conclusion.** Against the seed floor that prior work uses,
   a heavy update looks like the *most* shift-driven condition (2.3×); against the matched null it
   is the *least* (1.0×). The seed floor is fixed while `Δ` grows with training, so the ratio to it
   measures how hard you trained, not how far the data moved. This is the empirical justification
   for D2, and it means a substantial part of the attribution movement reported after fine-tuning
   in prior work is attributable to the optimisation rather than to the new distribution.

Consequently the experiment grid crosses **shift magnitude × update strength**, and the headline
figure reports the ratio surface rather than a single number.

### Shared support bounds the studiable shift

Overlap at `τ = 2` falls as the covariate shift grows: 0.84 at magnitude 0.5, 0.54 at 0.75, 0.32 at
1.0, 0.10 at 1.5, 0.028 at 2.0. Beyond magnitude ≈ 2 there is essentially no region where both
models are in-distribution, so instance-level comparison stops being meaningful. Magnitude 1.5 is
the headline (10% overlap); 2.0 is reported as a marginal-overlap sensitivity check. The overlap
fraction is reported beside every number.

## 9. Hypotheses

| ID | Statement | Test |
|---|---|---|
| H1 | Under covariate-shift updates (`ω = 0`), `Δ` exceeds `ρ_null` on prediction-preserved probes, by a margin that grows as the update gets lighter and the shift larger | paired Wilcoxon, Cliff's δ, seed-bootstrap CI on `ratio` over the magnitude × update grid |
| H1b | Ranking the update regimes by `Δ/ρ_seed` inverts the ranking by `Δ/ρ_null` | rank correlation between the two orderings across the grid |
| H2 | Under concept and shortcut shift, `Δ` is larger *and* aligned with ω | Pearson/Spearman `corr(Δ_E(x), ω(x))` |
| H3 | Unwarranted change is not predicted by ID accuracy, ECE, or prediction-agreement rate | partial correlation / OLS across runs |
| H4 | `\|1ᵀΔIG\| / (\|δ(x)\|+\|δ(b)\|) ≤ 1` always; the same ratio for G×I exceeds 1 frequently; `ε_coal ≫ ε_data` | per-point bound check |
| H5 | Unwarranted change is larger for updates that move within the Rashomon set than for accuracy-improving updates | LR/epoch sweep, regression on Δaccuracy |

H6 ("different explainers have different profiles") and H7 ("drift grows with shift magnitude") are
**not hypotheses** — they are known and expected. Reported as context.

## 10. Pseudocode

```
def uec(dataset, shift, explainer, distance, eps, seeds):
    out = []
    for s in seeds:
        f_t   = train(source, seed=s)
        f_null = update(f_t, sample(source, n_target), seed=s)      # matched-operator null
        f_seed = train(source, seed=s + 1000)                       # seed floor
        f_tp1  = update(f_t, target(shift),          seed=s)        # treatment

        P  = shared_support_probe(source, target(shift), n)
        Pe = [x for x in P if abs(prob(f_t,x) - prob(f_tp1,x)) <= eps]

        A_t   = explainer(f_t,   Pe, run=0)
        A_t2  = explainer(f_t,   Pe, run=1)                         # nu
        A_n   = explainer(f_null, Pe)
        A_s   = explainer(f_seed, Pe)
        A_tp1 = explainer(f_tp1, Pe)

        nu    = distance(A_t, A_t2)
        rho_n = distance(A_t, A_n)
        rho_s = distance(A_t, A_s)
        delta = distance(A_t, A_tp1)
        omega = distance(gt(source, Pe), gt(target(shift), Pe))     # 0 for covariate

        out.append(dict(nu=nu, rho_null=rho_n, rho_seed=rho_s, delta=delta, omega=omega,
                        uec=mean(delta) - mean(omega) - max(mean(nu), mean(rho_n)),
                        ratio=mean(delta) / mean(rho_n),
                        exceed=mean(delta > q95(concat(nu, rho_n))),
                        n_preserved=len(Pe), n_probe=len(P)))
    return bootstrap_over_seeds(out)
```

## 11. Assumptions

1. The explained output is the logit; all theory and ω are stated in logit space.
2. The baseline `b` is fixed across checkpoints and environments. A per-checkpoint baseline would
   make ΔIG uninterpretable.
3. Probe points lie in the shared support, so neither checkpoint is extrapolating.
4. Update operators are matched in learning rate, epochs, step count, and training-set size within
   any treatment/control comparison.
5. ω is exact only where the generative process is known; on real data only `ω = 0` under covariate
   shift is claimed.
6. Seeds are the unit of statistical dependence; all confidence intervals bootstrap over seeds.
7. Attribution methods explain class 1; for multiclass vision we explain the predicted class of the
   *source* model, held fixed across checkpoints.
