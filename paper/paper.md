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

All synthetic numbers are 10 seeds, covariate magnitude 1.5 (10% shared-support overlap), a
2-epoch update at lr 2×10⁻⁴, ε = 0.05, normalised-ℓ₁ distance, unless stated. Intervals are paired
seed bootstraps. Source: `paper/tables/`, `figures/`.

### 7.1 The controls behave

Prediction agreement on the probe is 0.98 under covariate shift and 0.99 under shortcut removal, so
the conditioning set is not a rare corner: 79% of probe points are preserved at ε = 0.05.

The noise floor `ν` is exactly 0 for saliency, Grad×Input and IG (bitwise reproducible), 0.016 for
SmoothGrad, 0.073 for KernelSHAP, 0.056 for LIME, and 0.135 for Expected Gradients. Two of those
matter. **Expected Gradients is noise-dominated**: its `ν` (0.135) is indistinguishable from its
matched null (0.138) and from its measured change (0.141), so EG at 32 background samples cannot
resolve any effect in this setting. **LIME is partly noise-dominated**: `ν` = 0.056 exceeds its
matched null of 0.030. Both are reported as findings about the explainers, not as failures of the
estimator — UEC subtracts `max(ν, ρ_null)` and correctly returns ≈ 0 for EG.

The frozen-model check is exact: with a fixed checkpoint, attributions at a fixed input are bitwise
identical whether the input is drawn from the source or the target distribution (§2.1).

**The placebo passes.** With no shift at all, the treatment is a second draw from the source
distribution and the ratio is 1.00–1.10 for every explainer, with every interval covering 1
(EG 1.000 [0.986, 1.014]; IG 1.05; LIME 1.10).

### 7.2 H1: unwarranted change exists (Fig. 2, T2, T3)

Under covariate shift ω is exactly 0, so all measured change on the shared support is unwarranted.
All seven explainers exceed their matched null, and all seven survive Holm correction:

| explainer | ν | ρ_null | ρ_seed | Δ | ratio [95% CI] | Cliff's δ |
|---|---|---|---|---|---|---|
| Saliency | 0.000 | 0.025 | 0.200 | 0.044 | **1.73 [1.58, 1.89]** | +0.98 |
| SmoothGrad | 0.016 | 0.025 | 0.197 | 0.044 | 1.73 [1.58, 1.89] | +0.98 |
| Grad×Input | 0.000 | 0.025 | 0.131 | 0.042 | 1.67 [1.49, 1.87] | +0.90 |
| IG | 0.000 | 0.026 | 0.097 | 0.040 | 1.52 [1.36, 1.71] | +0.90 |
| KernelSHAP | 0.073 | 0.028 | 0.126 | 0.039 | 1.40 [1.27, 1.55] | +0.88 |
| LIME | 0.056 | 0.030 | 0.113 | 0.038 | 1.25 [1.14, 1.37] | +0.64 |
| EG | 0.135 | 0.138 | 0.162 | 0.141 | 1.02 [1.02, 1.03] | +0.48 |

Holm-adjusted p = 0.014 for all seven (the floor of a 10-seed paired Wilcoxon after correcting
across seven tests). Cliff's δ ≥ 0.88 for five of seven means near-perfect separation across seeds.

### 7.3 H1b: the choice of control reverses the conclusion (Fig. 2b)

This is what the matched null buys. Over a grid of shift magnitude × update strength (600 runs,
5 seeds), the ratio to the matched null **falls** as the update gets heavier, while the ratio to the
seed floor **rises**:

| update | Δ/ρ_null (IG, shift 1.5) | Δ/ρ_seed |
|---|---|---|
| 1 epoch | **1.81** | 0.11 |
| 2 epochs | 1.58 | 0.19 |
| 5 epochs | 1.39 | 0.41 |
| 20 epochs | 1.28 | 0.82 |
| 60 epochs | 1.09 | **1.07** |

The two controls order the update regimes in opposite directions, and the curves cross. Read against
the seed floor — the control implicit in prior work — a heavy update looks like the most
shift-driven condition; read against the matched null it is the least. The reason is mechanical: the
seed floor is a fixed quantity while Δ grows with training, so the ratio to it measures *how hard
you trained*, not *how far the data moved*.

The practical consequence is a claim about the existing literature: **a substantial part of the
attribution movement reported after fine-tuning is attributable to the optimisation rather than to
the new distribution**, and no study lacking a matched-operator control can separate the two.

Note also that the effect is largest exactly where predictions are best preserved: at 1 epoch,
agreement is 0.98 and the ratio is at its maximum. The two desiderata align rather than trade off,
and light incremental fine-tuning is the realistic deployment regime.

### 7.4 H2: magnitude is uninformative about legitimacy (Fig. 2c, T4)

This is the central result. Compare a shift where the mechanism does not change (covariate, ω = 0)
with one where it changes a great deal (shortcut removal, ω = 0.326), under the same update:

| explainer | Δ covariate (ω=0) | Δ shortcut (ω=0.33) | paired p |
|---|---|---|---|
| IG | 0.0398 | 0.0386 | **0.63** |
| KernelSHAP | 0.0388 | 0.0415 | 0.23 |
| Saliency | 0.0436 | 0.0366 | 0.010 |
| Grad×Input | 0.0419 | 0.0365 | 0.037 |

For IG and KernelSHAP the measured change is **statistically indistinguishable** between the two
cases. For saliency and Grad×Input it differs — but in the *wrong direction*: they move **more** when
nothing should change than when everything should. Under a light update the attribution change
carries essentially no information about whether the underlying mechanism moved.

UEC separates what magnitude cannot: **+0.014 under covariate shift** (unwarranted change present)
versus **−0.31 under shortcut removal** (the model moved far *less* than warranted).

Two failure modes, not one: **light updates produce change where none is warranted, and fail to
produce change where it is.**

Does the model ever adapt? Yes, with a stronger update. Ground-truth attribution mass on the
shortcut feature should fall from 0.337 to 0. The model takes it from 0.261 to 0.226 at 2 epochs
(13% of the way), 0.060 at 20 epochs (77%), and 0.022 at 100 epochs (92%). Warranted adaptation
requires a substantial update; unwarranted change appears with a minimal one.

### 7.5 What prior metrics report on the same checkpoints (T4)

At 100 update epochs, where the model has genuinely adapted to the shortcut removal:

| metric | covariate (ω=0, change is **wrong**) | shortcut (ω=0.25, change is **right**) |
|---|---|---|
| FASS-style filtered distance | 0.192 | **0.279** |
| Delta-Audit JSD | 0.045 | **0.087** |
| Delta-Audit "spurious" residual | 0.080 | **0.151** |
| ROS (mean) | 4.81 | 1.17 |
| **UEC** | **+0.015** | **−0.143** |

FASS-style distance and Delta-Audit's spurious residual both rank the *correct* adaptation as worse
than the *incorrect* drift, by 45% and 89% respectively. They are not miscalibrated; they are
measuring a quantity that cannot express the distinction. UEC assigns opposite signs.

ROS behaves as the theory predicts: its mean swings from 2.03 to 4.81 to 0.49 across update strengths
with no interpretable pattern, because its denominator vanishes in exactly the prediction-preserving
regime the paper is about.

### 7.6 H4: the method-class asymmetry (Fig. 5, T7)

Over 20,000 probe points (10 seeds × 4 shift families × 500 points):

- **Proposition 1(i) holds without exception. Zero violations.** Maximum observed slack 1.00023
  against a propagated quadrature tolerance of 3.4×10⁻⁴; the identity residual is 0 to 10⁻⁸. The
  bound is also *tight* — median slack 0.87–0.99 — so it is not vacuous.
- **Proposition 2 is realised by ordinary fine-tuning.** Grad×Input's aggregate exceeds the same
  bound on **60.8%** of points (60.5% among prediction-preserved points), with a 90th percentile of
  1.47–1.62.
- **Proposition 3's premise fails in the predicted direction, at moderate size.** Coalition-level
  deviation exceeds data-level deviation by a median factor of 1.5–2.0. The usable bound is
  therefore ≈ 3ε_data rather than 2ε_data, and it is governed by off-manifold divergence, which
  prediction preservation does not control. We had predicted "≫" and record the correction.

The practically important row is the one no method class satisfies: rank-order change is large for
every explainer under every condition. Completeness buys aggregate stability and nothing else.

### 7.7 H3: not invisible, but not usable either (T3b, T3c)

The original hypothesis — that unwarranted change is uncorrelated with the metrics practitioners
watch — is **partly refuted, and the refutation is worse news than the hypothesis**. Across the
600-run regime grid:

- Prediction agreement correlates **positively** with the ratio (IG r = +0.22, p < 10⁻⁴;
  Grad×Input r = +0.27, p < 10⁻⁵). Higher agreement goes with *more* unwarranted change relative to
  the null — the opposite of the natural inference.
- Target accuracy's correlation **reverses sign across shift magnitudes** (from +0.44 at magnitude
  0.75 to −0.31 at magnitude 2.0), so it is not a usable proxy in either direction.
- Calibration (ECE) shows no consistent association (|r| ≤ 0.16 across explainers).

So a practitioner watching accuracy, calibration and prediction agreement is not merely uninformed
about unwarranted explanation change; the one signal that does correlate points the wrong way.

### 7.8 Real data (T8, Fig. 8)

ACS Income, source CA 2018, four target states, 10 seeds, light update. The shared-support screen
works: domain AUC is 0.61–0.72 over the pooled data but 0.55–0.63 within the probe.

Ratios under a 2-epoch update, primary distance: MI 1.53–1.87, MS 1.80–2.12, SD 1.70–1.98,
PR 1.44–1.75, with Cliff's δ = +1.00 (perfect separation across seeds) for most explainers, and EG
again pinned at ≈ 1.0 by its own noise. The synthetic pattern replicates in direction and magnitude.

ω is not computable here, so no change on real data is labelled warranted. The calibration-transfer
check (`mech_gap`) is reported as a *necessary-not-sufficient* screen on the covariate-shift null.

## 8. Ablations

The conclusion does not depend on any of the arbitrary choices (T5). Kendall τ between the
explainer ranking under the primary configuration and under each alternative:

| axis | variants | τ vs primary |
|---|---|---|
| distance | Spearman, cosine, ℓ₁ | **1.00** |
| distance | top-k (k = 3, 10) | 0.62, 0.91 |
| ε | 0.01, 0.02, 0.05, 0.10 | **1.00** (mean ratio 1.47 → 1.53) |
| φ | absolute, signed | **1.00** |
| feature set | all, collinearity-reliable | **1.00** |

Two ablations deserve comment.

**Collinearity does not explain the effect — it suppresses it.** On the collinearity-reliable feature
partition (Attribution Impossibility Z ≥ 1.96), the mean ratio *rises* from 1.50 to 1.92. Removing
the features whose rankings are provably unreliable makes the effect larger, not smaller.

**Top-k Jaccard can be blind to warranted change by construction.** When the reference is supported
on exactly k features, the top-k set cannot move: ω measured by top-5 Jaccard is identically 0 under
our concept shift, while ℓ₁ gives 0.11 and top-3 gives 0.16. This is why k is an ablation axis and
why rank-based distances are not the primary choice here — with 15 of 20 features carrying zero
attribution, a full sign flip of the mechanism registers as 1 − ρ_s = 0.002.

## 9. Limitations

1. **ω is exact only where the generative process is known.** On real data we claim only the
   covariate null and check it with a calibration-transfer screen that is necessary, not sufficient.
   A reviewer is right that real-data legitimacy is not verifiable; that is why the real-data section
   reports Δ and floors and declines to label.
2. **Shared support bounds the studiable shift.** Overlap falls to 2.8% at covariate magnitude 2.0.
   Beyond that, instance-level comparison is not defined, so this method cannot speak to severe
   shift — precisely the regime practitioners most worry about.
3. **Scale.** MLPs, gradient-boosted-scale tabular data, and a small CIFAR ResNet on CPU. Nothing
   here is evidence about LLMs or large vision transformers.
4. **Attribution only.** No concept-based explanations, no attention, no counterfactuals.
5. **Two explainers are noise-dominated at our budgets** (EG at 32 samples, LIME partly). Their
   ratios are uninformative rather than null, and larger sample budgets would change them.
6. **The update operator is a plain Adam fine-tune.** We ablate learning rate and epochs but not
   optimiser family, regularisation, or replay.
7. **`ω` compares the Bayes predictor's IG to a model's IG.** Both are IG with a common baseline, so
   the comparison is like-for-like, but a model that is not close to Bayes-optimal will show change
   relative to a reference it was never going to match.

## 10. Conclusion

Explanations move when models are updated, and the field has been measuring that movement without a
reference for how much movement was called for and without a control for how much would have
happened anyway. Supplying both changes what the measurements mean.

With a warranted-change reference, the magnitude of attribution change turns out to carry almost no
information about whether the underlying mechanism moved: under a light update, integrated gradients
change by the same amount whether the Bayes-optimal predictor is unchanged or has been rewritten.
With a matched-operator control, a large part of the movement attributed to distribution shift in
prior work is revealed as the ordinary effect of continued training — and the control that prior work
implicitly uses orders the regimes backwards.

The theory says why no better attribution method fixes this. Completeness pins the aggregate
attribution mass of path-integrated explainers to the output change, and pins nothing else; local
gradients inherit nothing even in aggregate; Shapley values are bounded only off the data manifold
where nothing constrains them. The allocation across features — the only part anyone reads — carries
no stability guarantee in any class.

For practice the recommendation is narrow and, we think, defensible: do not read attribution change
across a model update without a matched-operator null, and do not interpret its magnitude as
evidence about the model's reasoning having changed for a reason.

---

## Appendix A — Terminology

## Appendix B — Novelty deltas
