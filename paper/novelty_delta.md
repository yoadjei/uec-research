# Appendix B — Novelty deltas

For each of the closest works: what it does, what we do differently, and — where we can — the
number in our results that demonstrates the difference rather than asserting it. Every entry was
verified against the live literature in Aug 2026, not taken from memory; three citation errors found
during that check are noted.

## B.1 The eight closest papers

### Mougan, Broelemann, Kasneci, Tiropanis & Staab — *Explanation Shift* (TMLR 01/2025, arXiv:2303.08081)

**Theirs.** Explanation shift is a two-sample comparison between the distribution of SHAP values on
training data and on new data, for a **frozen** model. A classifier on SHAP vectors detects
model-relevant distribution shift more sensitively than input- or output-based detectors.

**Ours.** Different object. For a frozen model and a deterministic explainer, instance-level change
is *identically zero* — we verify this bitwise (§7.1) — so the only thing that can move is the
distribution, which is exactly what they measure and we do not re-claim. Our object is the model
update. We also inherit a constraint from their theory: Shapley efficiency implies prediction shift
⟹ explanation shift, which is an independent reason to condition on prediction preservation.

### Agarwal, Johnson, Pawelczyk, Krishna, Saxena, Zitnik & Lakkaraju — *Rethinking Stability* / ROS (ICLR 2022 workshop, arXiv:2203.06877)

*(The audit under review attributed this to the OpenXAI author list; corrected here.)*

**Theirs.** Relative Output Stability: `max ‖ΔE‖ / ‖Δlogits‖` over an input neighbourhood of one
model.

**Ours.** We do not use a ratio to output change, because it is undefined and explosive precisely in
the prediction-preserving regime we study. Demonstrated, not argued: our reimplementation of ROS on
the same checkpoints gives mean values of 2.03, 4.81 and 0.49 across three update strengths with no
interpretable ordering (§7.5). UEC is a difference against a reference and two floors instead.

### Subramaniakuppusamy & Gajjar — *FASS* (arXiv:2604.02532, Apr 2026)

**Theirs.** Prediction-invariance filtering for attribution stability under geometric, photometric
and compression perturbations of the **input**, decomposed into SSIM, rank correlation and top-k
Jaccard. Establishes that without conditioning, up to 99% of compared pairs have changed
predictions.

**Ours.** We adopt their conditioning and say so; we were not first to it. We change the object from
input perturbation to model update, and we add what filtering alone cannot supply: a reference for
how much change was warranted, and a floor for how much would have occurred anyway. **The delta is
measurable**: on the shortcut-removal case, where the Bayes-optimal predictor genuinely changes, a
FASS-style filtered distance reports 0.279 against 0.192 for the covariate case — ranking correct
adaptation as 45% *worse* than unwarranted drift (§7.5).

### Hemmat & Fatemi — *Delta-Audit* (arXiv:2508.19589, Aug 2025)

**Theirs.** Differences occlusion attributions between model versions; a quality suite of L1/top-k/
entropy, rank-overlap@10, JSD, Delta Conservation Error and Behaviour–Attribution Coupling; flags
movement uncorrelated with behaviour change as risky reliance redistribution. 45 settings over five
classical learners and three small UCI datasets.

**Ours.** Closest on object — this is also a model-update audit. Three differences. (i) They have no
reference: "spurious" is defined as *residual after regressing on behaviour change*, which is a
heuristic, and it misfires exactly where legitimacy matters — their spurious residual is 0.151 for
warranted shortcut adaptation against 0.080 for unwarranted covariate drift, an 89% inversion
(§7.5). (ii) They have no matched-operator control, so their audit cannot separate the update's
distribution from the update's training. (iii) No shift typology with known mechanism status, and
occlusion only.

### Dhayalkar — *Reasoning Stabilization Point* (arXiv:2601.11625, Jan 2026)

**Theirs.** Defines *explanation drift* as epoch-to-epoch change in normalised token attributions on
a fixed probe set (1 − Spearman), and the earliest epoch after which drift stays low. Shows drift
exposes shortcut adoption while validation accuracy stays competitive.

**Ours.** They own the term and the within-run temporal object; we reserve "drift" for their meaning
and do not use it for ours (Appendix A). Their axis is training time within one distribution; ours
is a change of distribution across an update, with a null that holds training time fixed. Their
finding that drift is visible while accuracy is not is consistent with, and sharpened by, our H3:
the signal that *does* correlate with unwarranted change — prediction agreement — points the wrong
way (§7.7).

### Elangovan et al. — *When Fine-Tuning Changes the Evidence* (arXiv:2604.08513, Apr 2026)

**Theirs.** Semantic drift between transfer-learned and fully fine-tuned chest X-ray checkpoints,
with explicitly **reference-free** metrics, conditioned on true positives; stability rankings reverse
between LayerCAM and Grad-CAM++ at converged accuracy.

**Ours.** Same object (fine-tuning), and they are right that it matters. They describe their metrics
as reference-free; supplying the reference is our contribution. Single task and CAM-family only
versus our shift typology and seven explainers, and no floors.

### Bensmail — *EvoXplain* (arXiv:2512.22240, Dec 2025)

**Theirs.** Mechanistic multiplicity across repeated training runs; explanation samples cluster into
distinct modes even at equal accuracy.

**Ours.** This is our **floor**, not our treatment. We measure it as `ρ_seed` and report it
throughout. The methodological point is that using this floor as the control — which is
what the retraining literature implicitly does — inverts the ordering of update regimes (§7.3).

### Thackshanaramana B — *Hypothesis Class Determines Explanation* (arXiv:2603.15821, Mar 2026)

**Theirs.** Prediction-equivalent models across hypothesis classes disagree on attributions; the
"Explanation Lottery"; an Explanation Reliability Score predicting cross-architecture stability.

**Ours.** They vary the model class at a fixed distribution; we hold architecture fixed and vary the
distribution through an update operator. Complementary axes of the same non-identifiability.

## B.2 Two papers the audit missed

### Kiourti, Singh, Duraipandian, Zhou & Li — *Rethinking Robustness* (arXiv:2512.06665, Dec 2025)

Nearest neighbour of our **critique** — it also objects that existing robustness metrics ignore the
model's output difference — but it stays with input perturbations of a single model and answers with
a better ratio and GAN-generated test cases. We change the object and replace the ratio with a
reference.

### *Missingness Bias Calibration* (MCal, arXiv:2603.04831, Mar 2026)

Independent evidence for the premise of our Proposition 3 remark: models behave anomalously on
ablated, off-manifold inputs, and that anomaly is a property of the output space. MCal corrects it
within one model; we *measure* the deviation **between two checkpoints** (`ε_coal`, median 1.4–1.9×
the data-level gap, §7.6), which is what governs whether Shapley stability can be inferred from
prediction stability. Not in MCal.

## B.3 Contributions that are not in any of the above

| # | Contribution | Evidence |
|---|---|---|
| 1 | A warranted-change reference ω defined against the **Bayes-optimal predictor per environment**, exact in closed form | §4; ω = 0 bitwise under covariate shift, 0.326 under shortcut removal |
| 2 | The **matched-operator null** — a control targeted searches found nowhere in this literature | §7.3; reverses the ordering that the seed floor gives |
| 3 | Magnitude of attribution change is **uninformative about legitimacy** | §7.4; IG: 0.0398 vs 0.0386, p = 0.63, across ω = 0 vs ω = 0.33 |
| 4 | Prior metrics rank **correct adaptation as worse** than unwarranted drift | §7.5; FASS-style +45%, Delta-Audit spurious +89% |
| 5 | Method-class asymmetry with a sharpness construction, and the negative half nobody states: the *allocation* is unprotected in every class | §5, §7.6; 0/20,000 IG violations, 60.8% Grad×Input |
| 6 | Coalition-level deviation between checkpoints measured, not assumed | §7.6 |
| 7 | Collinearity **suppresses** rather than explains the effect, shown with the Attribution Impossibility's own Z-test | §8; ratio 1.50 → 1.92 on the reliable partition |
| 8 | Two failure modes: unwarranted change *and* missing warranted change | §7.4; UEC +0.014 vs −0.31 |

## B.4 Claims deliberately not made

- Not a new stability metric. Every distance is standard.
- Not a shift detector.
- Not the first to condition on prediction preservation.
- Not the first to difference attributions across a model update.
- No vision benchmark claim: FASS and the chest X-ray study own vision attribution stability, and
  our CIFAR experiment is a sanity check that says so.
- No claim about "reliability" in the human-trust sense.
