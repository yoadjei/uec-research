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
predictions are best preserved.

Two further results follow. First, the magnitude of attribution change is **uninformative about
legitimacy**: for six of seven explainers, change relative to the matched null is statistically
indistinguishable between a shift that leaves the Bayes-optimal predictor untouched and one that
rewrites it, and two published metrics consequently rank correct adaptation as 45% and 89% *worse*
than unwarranted drift. Second, the monitored quantity that does correlate with unwarranted change
points the wrong way: higher prediction agreement goes with *more* unwarranted change, while target
accuracy's association reverses sign across shift magnitudes. We accompany this with three propositions establishing which
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

**Distance.** The primary distance is the normalised-ℓ₁ distance between attribution masses, not
rank correlation, and the reason is stated here rather than left to the ablation. Rank correlation
is the natural choice for comparability with RSP and FASS, but it is dominated by tie structure when
many features carry zero attribution: on a reference supported on 5 of 20 features, a full sign flip
of the generating mechanism registers as `1 − ρ_s = 0.002` while the same change is 0.11 in ℓ₁, and
top-5 Jaccard registers it as exactly 0 because the top-5 set cannot move. ℓ₁ on the normalised
absolute attributions is a proper metric on the attribution simplex, magnitude-sensitive, and free of
that pathology. Spearman, cosine and two top-k variants are reported as ablations, and the explainer
ranking is identical under all of them (§8), so nothing turns on the choice.

**Statistics.** Seeds are the unit of dependence; all intervals bootstrap over seeds, and ratios use
a paired bootstrap since numerator and denominator come from the same seed. Paired Wilcoxon with
Cliff's δ compares treatment to null per probe point; Holm corrects across explainers. Note the
floor on a paired Wilcoxon: with 10 seeds the smallest attainable p is 0.002, and after Holm
correction across seven explainers, 0.014. Five seeds cannot reach significance at all after
correction, which is why the main experiments use ten.

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

The two families use different source models (the shortcut source is trained on data where the
shortcut is predictive), so raw Δ is not a controlled contrast. The controlled comparison normalises
each condition by **its own** matched null, which removes that confound:

| explainer | ratio, covariate (ω=0) | ratio, shortcut (ω=0.33) | paired p |
|---|---|---|---|
| IG | 1.52 [1.36, 1.71] | 1.56 [1.39, 1.77] | **0.56** |
| Saliency | 1.73 [1.58, 1.89] | 1.69 [1.54, 1.88] | **0.38** |
| SmoothGrad | 1.73 [1.58, 1.89] | 1.69 [1.54, 1.88] | **0.43** |
| Grad×Input | 1.67 [1.49, 1.87] | 1.57 [1.41, 1.76] | **0.38** |
| KernelSHAP | 1.40 [1.27, 1.55] | 1.59 [1.43, 1.80] | **0.16** |
| EG | 1.02 [1.02, 1.03] | 1.02 [1.01, 1.02] | **0.23** |
| LIME | 1.25 [1.14, 1.37] | 1.86 [1.67, 2.08] | 0.002 |

For **six of seven explainers** the amount of attribution change, relative to what the same update
would have produced on in-distribution data, is **statistically indistinguishable** between a shift
that leaves the Bayes-optimal predictor untouched and one that rewrites it. LIME is the exception,
and it is the explainer whose own noise floor exceeds its matched null.

The raw magnitudes tell the same story (IG: 0.0398 vs 0.0386, p = 0.63; KernelSHAP: 0.0388 vs
0.0415, p = 0.23), and for saliency and Grad×Input they differ in the *wrong direction* — moving
**more** when nothing should change than when everything should.

Under a light update, then, attribution change carries essentially no information about whether the
underlying mechanism moved.

UEC separates what magnitude cannot: **+0.014 under covariate shift** (unwarranted change present)
versus **−0.31 under shortcut removal** (the model moved far *less* than warranted).

Two failure modes, not one: **light updates produce change where none is warranted, and fail to
produce change where it is.**

Does the model ever adapt? Yes, with a stronger update. Ground-truth attribution mass on the
shortcut feature should fall from 0.337 to 0. The model takes it from 0.261 to 0.226 at 2 epochs
(13% of the way), 0.060 at 20 epochs (77%), and 0.022 at 100 epochs (92%). Warranted adaptation
requires a substantial update; unwarranted change appears with a minimal one.

**The retrain-from-scratch regime.** Retraining on the target rather than fine-tuning changes
attributions by 0.19–0.39 — five to ten times the fine-tuning change, and above even the seed floor.
But the shift contributes almost none of it: for IG, retraining from scratch moves attributions by
0.205 with **no shift at all** and 0.252 under covariate shift, a ratio of 1.23. Retraining destroys
explanation stability almost entirely through the retraining, which is the same lesson as §7.3 at a
larger amplitude.

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

ACS Income, source CA 2018, four target states, 10 seeds, light update (2 epochs). The
shared-support screen works: domain AUC is 0.61–0.72 over the pooled data but 0.59–0.63 within the
probe, and the probe is balanced across domains by construction — necessary here, because SD supplies
4,899 rows against CA's 195,665 and an unbalanced pool would be a source-only probe.

**27 of 28 explainer × state combinations are significant after Holm correction**, with prediction
agreement 0.94–0.96:

| target | mech_gap | IG | Saliency | KernelSHAP | LIME | EG |
|---|---|---|---|---|---|---|
| MI | **0.064** | 1.70 [1.50, 1.92] | 1.74 [1.59, 1.90] | 1.59 | 1.59 | 1.03 |
| SD | 0.142 | 1.90 [1.68, 2.12] | 1.91 [1.76, 2.08] | 1.63 | 1.75 | 1.03 |
| MS | 0.203 | 1.95 [1.71, 2.19] | 1.94 [1.78, 2.12] | 1.76 | 1.85 | 1.02 |
| PR | 0.287 | 1.50 [1.32, 1.70] | 1.62 [1.46, 1.78] | 1.34 | 1.36 | 1.01 (n.s.) |

Cliff's δ is +1.00 — perfect separation across all ten seeds — for saliency, SmoothGrad and
Grad×Input on three of four targets. EG is again pinned at ≈ 1.0 by its own noise floor, and is the
single non-significant cell.

ω is not computable here, so **no change on real data is labelled warranted**. The
calibration-transfer check `mech_gap` is a *necessary-not-sufficient* screen on the covariate null,
and it orders the targets sensibly: Michigan's conditional transfers best (0.064) and Puerto Rico's
worst (0.287, consistent with its very different base rate, 0.106 against California's 0.370). The
result that matters for the covariate reading is that **Michigan — the state where the ω = 0 null is
most defensible — still shows a ratio of 1.70**, so the effect is not an artifact of unacknowledged
concept shift riding along with the covariate shift.

### 7.9 Vision sanity check (T9, Fig. 9)

CIFAR-10, a 78k-parameter ResNet, 3 seeds, additive update (the model is fine-tuned on its old data
plus new data that is either clean or corrupted). This is a sanity check, not a benchmark: FASS and
the chest X-ray study own vision attribution stability.

Every explainer exceeds its matched null with **Cliff's δ = +1.00** — perfect separation across all
three seeds — at ratios of 1.20–1.38 under corruption and 1.15–1.40 under the planted shortcut. IG
is highest (1.38) and Grad-CAM behaves like the pixel-space methods despite living on a coarse
feature grid, which it can only be compared to via its own floors. The ratios are smaller than in
the tabular settings and we do not read anything into the difference given three seeds and a weak
(59–63% accurate) backbone.

The shortcut panel reproduces the synthetic asymmetry. Attribution mass inside the planted corner
should fall to zero; under a 1-epoch update it moves from 0.096 to 0.083 for IG and is flat for
saliency and Grad×Input. As on synthetic data, a light update produces unwarranted change while
failing to produce the warranted change.

One design point transfers with a correction. Fine-tuning on the new data *alone* is domain
replacement, not an additive update: it drove accuracy from 0.59 to 0.44 and prediction agreement to
0.54, destroying the prediction-preserved probe. Replaying the old data alongside the new — what
deployment actually does — restores agreement to 0.72–0.82 with accuracy preserved.

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

### A protocol for practitioners

The findings imply a short procedure, and it costs two extra training runs.

1. **Build the matched null.** Before comparing `f_t` and `f_{t+1}`, apply the *same* update — same
   learning rate, epochs, step count and sample size — to a fresh sample from the **old**
   distribution. That checkpoint, not the original model and not an independent retrain, is the
   comparison point.
2. **Measure the explainer's own noise.** Run stochastic explainers twice on the same checkpoint. If
   that spread approaches the null (as it did here for Expected Gradients at 32 samples), the
   comparison cannot resolve anything and the sample budget must go up before any conclusion is
   drawn.
3. **Condition on prediction preservation** and report how many points survive. Without it, most of
   what is measured is the arithmetic consequence of changed predictions.
4. **Do not read the magnitude as evidence about the mechanism.** It does not distinguish a model
   that changed its evidence for no reason from one that correctly began tracking a real change
   (§7.4). If the mechanism's status matters, it has to come from knowledge of the shift, not from
   the attributions.
5. **Report the exceedance rate, not only the mean**, and check the conclusion on the
   collinearity-reliable feature subset.

The recommendation is narrow and, we think, defensible: do not read attribution change across a
model update without a matched-operator null, and do not interpret its magnitude as evidence that
the model's reasoning changed for a reason.

---

## Appendix A — Terminology

"Explanation drift" denotes three different objects in the 2025–26 literature, which is why this
paper does not use it.

| term | owner | object |
|---|---|---|
| **explanation shift** | Mougan et al., TMLR 2025 | divergence between `P(E_f(X_source))` and `P(E_f(X_target))` for a **frozen** model; population level |
| **explanation drift** | Dhayalkar (RSP), 2026 | epoch-to-epoch attribution change on a fixed probe **within one training run** |
| **semantic drift** | Elangovan et al., 2026 | CAM attribution change between transfer-learned and fine-tuned checkpoints |
| **drift explanation** | Hinder, Vaquet & Hammer | explaining **why the data drifted** — the opposite direction |
| **Δ-attribution** | Hemmat & Fatemi, 2025 | `φ_B(x) − φ_A(x)` between two model versions |
| **mechanistic multiplicity** | Bensmail (EvoXplain), 2025 | attribution variation across retraining seeds |

Our vocabulary: **explanation change** `Δ_E(x)` for the neutral quantity; **warranted** and
**unwarranted** change for the two components; `ν`, `ρ_null`, `ρ_seed` for the three floors; and
*drift* reserved for RSP's temporal sequence, which we do not study.

Claims we do not make, restated so they cannot be read into the paper: we propose no new stability
metric; we do not detect distribution shift; we were not first to condition on prediction
preservation (FASS was); we were not first to difference attributions across an update (Delta-Audit
and the chest X-ray study were); we make no vision benchmark claim; and we never use "reliability"
in the human-trust sense.

## Appendix B — Novelty deltas
