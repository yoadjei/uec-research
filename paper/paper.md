# Stable Predictions, Shifting Evidence: Warranted and Unwarranted Explanation Change Under Model Updates

*Draft. Sections 7–10 are populated from `paper/tables/` and `figures/`; every number in them is
produced by `experiments/` and traceable through `results/registry.csv`.*

---

## Abstract

When a deployed model is updated in response to a distribution shift, its feature attributions
move. Existing work measures how much they move and treats whatever remains after conditioning on
prediction change as instability. That is not a criterion: it has no account of how much an
explanation *should* have moved. We supply one. Using shift families whose Bayes-optimal predictor
is known in closed form, we define a warranted-change reference ω and decompose attribution change
across a model update into warranted, unwarranted, and explainer-noise components. The design turns
on a control that prior work does not use: a **matched-operator null**, in which the identical
update is applied to fresh data from the *source* distribution, so that treatment and control
differ only in the distribution and not in the amount of training. This control changes the
conclusion. Measured against the seed-retraining floor that prior work uses implicitly, heavy
fine-tuning looks maximally shift-driven; measured against the matched null it is not shift-driven
at all, and most of the attribution movement reported after a model update is attributable to the
optimisation rather than to the new data. Unwarranted change is largest for *light* updates, where
predictions are best preserved. We accompany this with three propositions establishing which
explainer classes inherit stability from prediction stability: completeness pins the *aggregate*
attribution mass of path-integrated explainers to within the output change, local-gradient
explainers inherit no such bound even in aggregate, and Shapley-type explainers are bounded only
under a coalition-level premise that fine-tuning does not deliver. None of the three protects the
*allocation* across features — which is the part practitioners read.

---

## 1. Introduction

Post-hoc feature attributions are consumed as evidence: a practitioner reads which features a model
relied on and acts on it. Models, meanwhile, do not stand still. They are fine-tuned when the data
drifts, retrained on refreshed samples, and re-deployed. A natural question follows: after such an
update, is the evidence still the same evidence?

The question has been asked, in pieces. Mougan et al. (2025) compare explanation *distributions*
across domains for a frozen model. FASS (2026) filters to prediction-preserved pairs before
measuring attribution stability under input perturbation. Delta-Audit (2025) differences
attributions across a model update. RSP (2026) tracks attribution change across fine-tuning epochs.
Each measures a magnitude. None answers the prior question: **how much should the explanation have
changed?** Absent that, every method is forced to treat change as damage, and each does so by fiat —
calling the residual "instability", "semantic drift", or "risky reliance redistribution".

Treating all change as damage is wrong, and provably so in a case that occurs in practice. A model
fine-tuned on data where a spurious shortcut has been removed *should* stop attributing to the
shortcut. Its predictions on most inputs are unchanged; its explanation changes a great deal; and
that change is exactly what we want. Any metric that scores this as instability is measuring the
wrong thing.

Our contributions:

1. **A warranted-change reference.** We define ω as the change in the attributions of the
   *Bayes-optimal predictor* of each environment — not of the causal generating mechanism. On
   generators where that predictor is available in closed form, ω is exact: identically zero under
   covariate shift, and non-zero and computable under concept shift and shortcut removal. This is
   the definitional choice that makes the shortcut case come out right (§4).
2. **A matched-operator null.** The control for a fine-tune-on-target treatment is the identical
   fine-tune on fresh *source* data, matched in learning rate, epochs, step count and sample size.
   Only the distribution differs. Prior work's implicit control — an independent retrain — is a
   different operator on a different amount of data, and we show it inverts the conclusion (§7.2).
3. **A decomposition, not a metric.** UEC subtracts ω and the floors from the measured change. Every
   distance we use already exists; the contribution is the reference and the controls.
4. **Three propositions** on which explainer classes inherit stability from prediction stability,
   each with a sharpness construction and a per-point empirical check (§5, §7.4).
5. **An empirical finding** that survives its own controls: unwarranted change exists, it is largest
   where predictions are best preserved, and it is invisible to accuracy, calibration and prediction
   agreement.

**What this paper does not claim.** We do not propose a new stability metric — the distances are
standard (Alvarez-Melis & Jaakkola 2018; Yeh et al. 2019; Agarwal et al. 2022; Quantus 2023). We do
not claim to detect distribution shift; that is Mougan et al.'s contribution and their frozen-model
setting is not ours. We were not first to condition on prediction preservation; FASS was, and we
adopt their conditioning. We do not use the phrase "explanation drift", which now denotes three
different objects in the 2025–26 literature (Appendix A).

## 2. Problem formulation

### 2.1 A category error worth stating once

For a frozen model `f` and a deterministic explainer `E`, the explanation `E_f(x)` is a fixed
function of `x`. Changing `P(X)` does not change `E_f(x)` at any `x`; it changes only which `x` are
observed. **Instance-level explanation change under pure data shift with a frozen model is
identically zero.** The only measurable object is the shift of the explanation *distribution*, which
is precisely and exclusively what Mougan et al. measure.

Genuine instance-level change requires exactly one of: (i) the model changed, (ii) the input
changed, or (iii) the explainer is stochastic. This paper studies (i). Case (ii) is owned by FASS
and the adversarial-fragility line (Ghorbani et al. 2019; Dombrowski et al. 2019). Case (iii) is a
nuisance we measure and subtract. We verify (i)–(iii) empirically rather than asserting them
(§7.1).

### 2.2 Objects

`f: X → R` is the explained scalar output, always the logit, so that the theory, the explainers and
the closed-form reference all live in the same space. `E_f: X → R^d` is an explainer, possibly
stochastic. An update operator produces `f_{t+1} = U(f_t, D)`. A probe set `P` is drawn from the
**shared support** of source and target, so that neither checkpoint is extrapolating on it; the
prediction-preserved subset is `P_ε = {x ∈ P : |p_t(x) − p_{t+1}(x)| ≤ ε}`.

Conditioning on preservation is not optional. Shapley efficiency implies that if two inputs'
predictions differ, their explanation vectors must differ in at least one coordinate: measuring
explanation change without conditioning partly measures a mathematical necessity rather than a
property of the model. This is the formal reason FASS's filtering step is right.

### 2.3 Shift typology

Shifts are *generated*, never assumed. Covariate shift moves `P(X)` with `P(Y|X)` fixed; concept
shift moves `P(Y|X)`; shortcut removal severs a feature that was predictive but never causal.

## 3. Related work, organised by what is measured

| What changes | Work | What is missing |
|---|---|---|
| the data, model frozen (population) | Mougan et al., TMLR 2025 | no legitimacy question; instance-level change is vacuous here |
| the input | FASS 2026; Ghorbani 2019; Dombrowski 2019; Rethinking Robustness 2025 | single model; residual called instability by fiat |
| the seed | EvoXplain 2025; Laberge et al., JMLR 2023 | no shift; this is our *floor*, not our treatment |
| the model class | Hypothesis Class Determines Explanation 2026 | architecture varies, distribution does not |
| the checkpoint | Delta-Audit 2025; RSP 2026; chest X-ray 2026 | closest to us; no reference, no matched control, no theory |

The two nearest works deserve specifics. **Delta-Audit** differences occlusion attributions across
A/B model pairs and labels movement uncorrelated with behaviour change "spurious" — a heuristic, not
a reference, and it misfires precisely on the shortcut case. **FASS** establishes that without
conditioning on prediction preservation up to 99% of compared pairs have changed predictions, then
measures raw distance on what remains, with no floor to compare it against.

## 4. Warranted change

**The reference.** ω is defined against the Bayes-optimal predictor `h_e(x) = logit P_e(Y=1|X=x)` of
each environment:

```
ω(x) = d( φ(IG[h_S](x, b)), φ(IG[h_T](x, b)) )
```

Referencing the *causal mechanism* instead would be a mistake. Under shortcut removal the causal
function is unchanged, yet a correct model must stop using the shortcut; a causal reference would
label that change unwarranted. The Bayes reference gives the intended verdicts:

| shift | `h` changes? | ω | a matching Δ is |
|---|---|---|---|
| covariate | no | **exactly 0** | unwarranted |
| concept | yes | closed form | warranted |
| shortcut removal | yes | closed form | warranted |

**The floors.** `ν` is the explainer's own noise, measured between two runs on the same checkpoint.
`ρ_null` is the matched-operator null. `ρ_seed` is the independent-retrain floor that prior work
uses. The decomposition is

```
UEC = mean_{P_ε}[Δ] − ω − max(ν, ρ_null),     ratio = mean Δ / ρ_null
```

UEC is deliberately a difference rather than a ratio to output change: a ratio (ROS) is undefined
and explosive exactly in the prediction-preserving regime this paper is about.

**Why the matched null is the load-bearing choice.** The null and the treatment contain the same
optimisation transient; only the treatment additionally contains the distribution signal. Comparing
a *fine-tune* against a *from-scratch retrain*, as the seed floor does, confounds the operator with
the distribution. §7.2 shows this is not a fastidious distinction: the two controls order the update
regimes in opposite directions.

## 5. Theory

Let `δ = f_t − f_{t+1}`, baseline `b`, path `γ(α) = b + α(x − b)`. All explainers below are linear
in the explained function.

**Proposition 1 (path-integrated: the aggregate is inherited, the allocation is not).**
(i) `Σ_j ΔIG_j(x) = δ(x) − δ(b)` exactly, hence `|Σ_j ΔIG_j(x)| ≤ 2ε` when the checkpoints agree to
ε at `x` and at `b`. Note this requires agreement only at the two endpoints, not along the path.
(ii) `‖ΔIG(x)‖₁ ≤ ‖x − b‖₁ · sup_γ‖∇δ‖_∞`.
(iii) *Sharpness:* output preservation alone does **not** bound individual IG components. With
`δ(z) = ε·sin(ω(z₁ − z₂))`, `b = 0`, `x = 1`: `|δ| ≤ ε` everywhere, yet `ΔIG₁ = εω` is unbounded
while the aggregate is exactly zero.

**Proposition 2 (local gradients inherit nothing, not even the aggregate).** With
`δ(z) = ε·sin(ωz₁)`, the aggregate gap `Σ_j Δ(G×I)_j(x) = x₁εω` is unbounded under the same premise
that pins IG's aggregate to `2ε`.

**Proposition 3 (Shapley, under a coalition-level premise).** If `|v_f(S) − v_{f'}(S)| ≤ ε_coal` for
every coalition `S`, then `‖φ^f − φ^{f'}‖_∞ ≤ 2ε_coal`.
*Remark.* `ε_coal` is a supremum over **masked, off-manifold** inputs. Fine-tuning optimises a loss
over the data distribution and leaves the model unconstrained there, so data-level preservation
implies nothing about `ε_coal`. We measure the gap rather than assume it (§7.4).

**Corollary.** Under a prediction-preserving update: IG's aggregate attribution mass is bounded;
gradient×input's is not; Shapley's is bounded only if the coalition premise holds. **No class is
protected in its allocation across features.** Completeness buys aggregate stability and nothing
else — and the allocation is what is read off the page.

The sharpness construction is not exotic in kind. It is a difference that oscillates *transverse* to
the integration path while vanishing on it: two checkpoints that agree on the data manifold but
curve differently off it, which is what fine-tuning produces. This is the mechanism Dombrowski et
al. identified in input space, transposed to model space.

Every proposition is verified numerically against closed-form functions in
`tests/test_theory_numeric.py`. A proof that fails its check is a wrong proof, not a flaky test.

### 5.1 A precondition the theory imposes on the experiments

Proposition 1 assumes `C¹`. A ReLU network is not `C¹`: its gradient is piecewise constant, so
Riemann quadrature for IG converges as `O(1/n_steps)` rather than `O(1/n_steps²)`, and completeness
holds only to ≈5×10⁻³ at 512 steps — the same order as the quantity the bound check measures. We
therefore use a smooth activation throughout and report ReLU as an ablation. This is not a
technicality: testing a completeness-based claim on a ReLU network measures quadrature error.

## 6. Experimental protocol

**Generators.** A structural model with `d = 20` in four blocks — causal, shortcut, redundant (noisy
copies of causal, collinear by construction) and noise. The Bayes logit is closed form because the
shortcut is drawn from the label and is conditionally independent of the causal block given it, so
ω is exact. Real data: ACS Income with state shifts. Vision: CIFAR-10 with locally generated
corruptions.

**Shared support.** On synthetic data the density ratio is closed form and probe membership is
exact. On real data a cross-fitted domain classifier estimates it — the classifier's logit *is* the
log density ratio up to the class-prior offset — and a within-probe domain AUC near 0.5 confirms the
screen worked. This substitution is the approximation, and it is stated as one.

The shared-support requirement bounds what can be studied. Overlap falls from 0.84 at covariate
magnitude 0.5 to 0.10 at 1.5 and 0.028 at 2.0; beyond ≈2 there is no region where both models are
in-distribution and instance-level comparison stops being meaningful. Magnitude 1.5 is the headline;
2.0 is reported as a marginal-overlap sensitivity check.

**Vision has no shared support under corruption.** A corrupted image is not a clean image. We
therefore make the vision update *additive* — the model is fine-tuned on extra data that is either
clean (null) or corrupted (treatment) — and keep the probe on the clean distribution where both
checkpoints are in-distribution by construction. This is also what deployment looks like.

**Collinearity partition.** Following the Attribution Impossibility (2026), rankings between
collinear features are unreliable independently of any shift, with a Z-threshold of 1.96. Our
redundant block is collinear by construction. Every headline result is reported on the full feature
set *and* on the reliable partition, so "this is just collinearity" is not an available deflation.

**Statistics.** Seeds are the unit of dependence; all intervals bootstrap over seeds, and ratios use
a paired bootstrap since numerator and denominator come from the same seed. Paired Wilcoxon with
Cliff's δ compares treatment to null per probe point; Holm corrects across explainers.

---

## 7. Results

*Populated from `paper/tables/tables.md` and `figures/`.*

## 8. Ablations

*Populated from `paper/tables/T5_ablations.csv`.*

## 9. Limitations

## 10. Conclusion

---

## Appendix A — Terminology

## Appendix B — Novelty deltas
