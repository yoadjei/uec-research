# Internal attack review

Five simulated reviews of the paper **as it now stands**, written against the artefacts that exist
in `figures/`, `paper/tables/` and `results/` — not against the plan. Each ends with *what would
change my score*, because a review that cannot be acted on is decoration.

Then: the AC meta-review, the audit's objection list with each item mapped to an artefact or
conceded, and a rebuttal plan.

Evidence base: 10 seeds on synthetic (13,608 rows) and ACS (17,640 rows), 10 on trees (1,240),
3 on vision (288), a 600-run regime grid, 200 faithfulness rows, 105 ablation rows, 94 tests.

---

## R1 — theory-oriented

**Summary.** A decomposition of attribution change across model updates against a closed-form
warranted-change reference, with three propositions on which explainer classes inherit stability
from prediction stability.

**Strengths.** The category-error argument in §2.1 is correct and is *verified bitwise* rather than
asserted — I checked the test. Proposition 1's sharpness construction is the paper's best technical
moment: the claim that IG "inherits output stability" is widely assumed and false, and the authors
show precisely which part survives (the aggregate, by completeness) and which does not. The
empirical check is unusually disciplined — zero violations in 20,000 points against a *propagated*
quadrature tolerance rather than a guessed epsilon, plus the observation that the bound is tight
(median slack 0.87–0.99) so it is not vacuous. The recognition that Proposition 1 requires C¹ and
that a ReLU network is not one — with the O(1/n) vs O(1/n²) convergence measured to prove it — is
the kind of care that makes me trust the rest of the paper.

I also note the authors *report a failed prediction*: they predicted ε_coal ≫ ε_data and got ≈1.7×,
and corrected it in three files rather than in the reader's head. That is rare.

**Weaknesses.** Propositions 1(i) and 3 are elementary — 1(i) is completeness plus a triangle
inequality, and 3 is linearity plus convexity of the Shapley weights. The genuinely novel content is
1(iii) and 2, and both are existence constructions rather than structural results. The theory tells
me *that* the allocation is unprotected but not *how much* to expect, so it does not predict the
1.5× the experiments find. I would want a statement about which update operators produce unwarranted
change — a flat-minimum or weight-space-distance argument — rather than only an empirical sweep.

**Questions.** (1) Is there a bound on ΔIG in terms of the *weight-space* distance between
checkpoints? (2) Does Proposition 3's premise fail more for larger background sets, and is there a
limit in which it is recovered?

**Score 6, confidence 4.**
**What would raise my score:** a proposition connecting the update operator to the *magnitude* of
unwarranted change, even under strong assumptions (linear model, quadratic loss). That would turn a
measurement paper with theory attached into a theory paper.

---

## R2 — empirical deep learning

**Strengths.** The matched-operator null is the right control and I have not seen it in this
literature. Figure 2b is the paper: the two controls order the update regimes in *opposite*
directions and the curves visibly cross. The negative controls are what convince me — a no-shift
placebo returning 1.00–1.10 with every interval covering 1, and the tree placebo at 0.98 (n.s.)
while the seed floor reports 1.90 on the same checkpoints. That last number is the single most
persuasive thing in the paper and I would put it in the abstract (it is).

The ablations are honest in a way I can check: the authors report that a *target-only* probe would
give 1.81 and the zero baseline 1.50 versus target-mean 2.34, and they use the smaller number in
both cases. Researchers optimising for a headline do not do that.

E6 answers my main methodological worry before I raised it. If the effect were the explainer
failing on one checkpoint, restricting to points where it is faithful to both would shrink it; it
does not move (1.567 vs 1.563), and corr(Δ, faithfulness) is within ±0.08 across all 20 cells.

**Weaknesses.** Scale, and I will not be talked out of it. The largest model is a 78k-parameter
ResNet at 59–63% CIFAR-10 accuracy; the tabular models are 2-layer MLPs and depth-4 trees. The
fine-tuning-stability literature (Mosbach et al.) suggests optimisation transients behave
qualitatively differently at scale, and the paper's central claim is *about* optimisation
transients. I do not think the result is wrong; I think its domain of validity is unestablished.

Vision uses 3 seeds. Cliff's δ = +1.00 with n=3 is not as impressive as it looks — perfect
separation of three pairs happens by chance one time in four under the null.

The effect sizes are moderate (1.4–2.1). The paper is careful not to oversell them, but a reader
skimming the abstract will take "2×" as the headline when the honest headline is the *sign* of the
comparison, not its size.

**Questions.** (1) Does the ratio grow, shrink or vanish with model width beyond 128? The ablation
stops there and shows a mild *decline* (1.57 → 1.45), which slightly cuts against a scale argument.
(2) Why 1 epoch for vision but 2 for tabular?

**Score 6, confidence 4.**
**What would raise my score:** one model at least an order of magnitude larger — a fine-tuned
DistilBERT on a text classification task with token attributions would do — even at 3 seeds. Or a
width sweep to 1024 on synthetic showing the ratio does not vanish.

---

## R3 — XAI specialist

**Strengths.** The related-work organisation *by what changes* is the clearest treatment of this
literature I have read, and the paper engages FASS, Delta-Audit, RSP and Mougan on their own terms
rather than strawmanning them. Crucially the differentiation table is **computed**: the authors
reimplement ROS, FASS-style filtered distance and Delta-Audit's coupling and run them on their own
checkpoints. That both prior metrics rank correct adaptation as *worse* than unwarranted drift
(+45%, +89%) is a real result and the strongest argument for the paper's existence.

The distance ablation (τ = 1.00 across Spearman, cosine, ℓ₁, and across all four ε) pre-empts my
standard objection. The finding that top-k Jaccard is *identically blind* to a mechanism change when
the reference is supported on exactly k features is a genuinely useful methodological note that I
have not seen stated.

I am glad the authors decline to define "reliability", and glad they say plainly that FASS was first
to condition on prediction preservation.

**Weaknesses.** Two of the seven explainers are noise-dominated at the chosen budgets (Expected
Gradients at 32 samples, LIME partly), and the paper reports this honestly but then keeps them in
the headline table where their ratios (1.02, 1.25) drag the reported range down. Either raise the
budgets or move them to an appendix — as it stands the range "1.02–1.73" understates the result for
the explainers that are actually measuring something.

The E6 faithfulness scores expose something the paper mentions only in passing: saliency and LIME
have near-zero or *negative* faithfulness in the covariate setting, meaning their rankings carry
almost no information about the model's output, yet they post among the highest ratios. That
deserves more than a caveat — it suggests the ratio is partly measuring instrument noise for those
methods, and the paper should say which explainers it considers measurement-grade.

**Questions.** (1) What is EG's noise floor at 128 or 512 background samples — does its ratio
recover? (2) Is the ratio for saliency conditional on positive faithfulness still above 1?

**Score 7, confidence 5.**
**What would raise my score:** re-run EG and LIME at a budget where ν < ρ_null, and report the
headline range over measurement-grade explainers only, with the noise-dominated ones shown
separately as a finding about explainer budgets.

---

## R4 — sceptical senior

**Summary.** "The Rashomon effect under fine-tuning, with a better control."

**Weaknesses first.** My standing objection to this literature is that it is underspecification with
extra steps and no actionable consequence. The paper partly disarms me. The matched-null result is
not obtainable from the Rashomon literature, and §7.3 shows the Rashomon-style control gives the
*opposite ordering* — that is a correction, not a refinement. And the tree placebo is the argument I
cannot wave away: on data where nothing shifted, the control that this literature uses reports a
1.90× effect. If that is right, a number of published claims are measuring their own training
procedure.

The result I also cannot dismiss is §7.4: attribution change of statistically indistinguishable
magnitude whether the mechanism moved or not (p = 0.56 controlled, 0.63 raw), because that is a
statement about what these methods can support, not about how much they wobble.

Remaining complaints, and they are real. The headline ratios are 1.4–2.1, not 5; the paper is
disciplined about this but the framing still leans on "×" language. Warranted change on real data is
unverifiable and the authors concede it, which leaves §7.8 as "direction replicates" rather than
evidence for the central claim. The effect is largest for the *lightest* updates, which invites the
reply that a 1-epoch update is barely an update — the paper answers this (predictions are 98%
preserved there, and light incremental fine-tuning is the common deployment pattern) but the answer
should be in the introduction, not §7.3.

And I want to be sure the covariate/shortcut contrast in §7.4 is fair. The two conditions use
different source models. The authors saw this and normalised each by its own matched null, which is
the right fix, and 6 of 7 explainers are then indistinguishable. Good — but say it in the main text
before the raw comparison, not after.

**Novelty.** Moderate-to-good, conditional on §7.4 and the placebo surviving scrutiny.
**Score 6, confidence 4.**
**What would raise my score:** an audit of one *published* result using the matched null — take a
setting from Delta-Audit or the chest X-ray paper, add the control, and show the reported
conclusion changes. That converts a methodological argument into a demonstrated correction of the
record, and it is the highest-value experiment left in this project.

---

## R5 — broad / accessibility

**Strengths.** Clear, well-scoped, and unusually candid about what it does not claim; the list of
six disclaimed claims made me trust the rest. Reproducible on CPU, with a run registry, seeds, and
configs that are *tested* against the runners so they cannot drift. Figure 1 is built from real
checkpoints rather than drawn by hand, and the authors even corrected a panel label that oversold
the effect. Two hypotheses came back partly refuted and are reported as such, including one (H3)
whose refutation is worse news for practitioners than the hypothesis was.

**Weaknesses.** Dense. The three-way distinction between ν, ρ_null and ρ_seed is the crux of the
paper and takes too long to become legible; Figure 2 should be readable before the reader has
internalised the notation. §7 has eleven subsections and the ordering buries the two best results
(§7.3 and §7.4) behind setup. The H3 finding — "the one monitored signal that correlates points the
wrong way" — is a better sentence than most of the abstract and arrives on page 12.

**Score 6, confidence 3.**
**What would raise my score:** restructure §7 to lead with the placebo and the control inversion,
and add a one-line glossary box for the three floors on the same page as Figure 2.

---

## AC meta-review

Scores 6 / 6 / 7 / 6 / 6, confidences 4 / 4 / 5 / 4 / 3. No reviewer disputes the methodological
contribution; all five accept the matched-operator null as new and correct. R3, the domain expert,
is the most positive and the most confident.

The audit's own decision rule — accept if the effect is ≥ 2× the floor, reject to TMLR if within
1.3× — is the wrong rule for the paper that was written. The headline ratio (1.4–2.1, intervals
excluding 1, Cliff's δ ≥ 0.88, placebo at 1.0) is solid but moderate, and a larger ratio would not
strengthen the paper's actual claims, which are qualitative:

1. the control inversion (§7.3) and the tree placebo (§7.8c) — the standard control manufactures a
   1.90× effect where nothing shifted;
2. magnitude is uninformative about legitimacy (§7.4);
3. two published metrics rank correct adaptation as worse than drift (§7.5);
4. the effect is not faithfulness loss (§7.7b) and not collinearity (§8).

Against that, R2's scale objection is the one substantive gap, and it is a limitation rather than an
error. R3's point about noise-dominated explainers in the headline range is a presentational fix.

**Recommendation: accept as poster.** The methodological correction is the contribution; the effect
size is secondary and honestly reported.

**Conditions the authors should meet in revision:** move the §7.4 controlled comparison ahead of the
raw one (R4); separate measurement-grade from noise-dominated explainers in the headline (R3);
restructure §7 to lead with the placebo (R5).

---

## Objection map (audit §10 and §23)

| Objection | Where answered | Status |
|---|---|---|
| "Just another stability metric" | §4, Appendix A — no metric proposed, all distances standard | answered |
| "Metric is arbitrary / distance-dependent" | §8, T5 — τ = 1.00 across distances, ε, φ, feature set | answered |
| "Evaluates XAI rather than advancing ML" | §1, §5 — identifiability framing plus method-class theorem | partly (R1 wants operator-level theory) |
| "Distribution shift insufficiently defined" | §2.3, §6 — shifts generated with known mechanism status | answered |
| "Conflates explanation and model instability" | §4, Fig. 2 — ν, ρ_null, ρ_seed reported separately | answered |
| "Experiments too heterogeneous" | synthetic + trees + ACS + vision sanity check | answered |
| "No theory" | §5, three propositions, all numerically verified | answered |
| "Obvious" | §7.3 inversion, §7.4 magnitude result, §7.8c placebo | answered |
| "No evidence it tracks meaningful reliability" | reliability never claimed | answered |
| "Already in prior work" | Appendix B, eight deltas with numbers | answered |
| "Attention is not explanation" | attention excluded by design | answered |
| "Benchmark too small" | not a benchmark paper | answered |
| "Ratio degenerate at Δf → 0" | §7.5 — ROS reimplemented, swings 0.49–4.81 | answered |
| "Rashomon explains everything" | §7.3, §7.8c, §8 — reliable-partition ratio *rises* to 1.92 | answered |
| "Fine-tuning hyperparameters drive the result" | §7.3 — 600-run lr × epoch grid | answered |
| "Probe from shared support is unrealistic" | §6, §8.1 — it is the only set where neither model extrapolates, and it is the *middle* of the three probe choices | answered |
| "ε-thresholding is arbitrary" | §8 — monotone across 0.01–0.10, τ = 1.00 | answered |
| "Synthetic too simple" | it is the only place ω is exact; ACS and trees are the demonstration | answered |
| "Why fine-tuning and not retraining?" | §7.4 — `delta_scratch` reported; trees are retrain-only | **now answered** (was under-reported) |
| "What should a practitioner do?" | §10 — five-step protocol | **now answered** (was conceded) |
| "Your explainers are just bad" | §7.7b — E6; effect unchanged on faithful-to-both points | **now answered** (was not raised in the audit) |
| "Does it hold at scale?" | §9 limitation 3 | **conceded — a limitation, not an answer** |
| "Real-data legitimacy is unverifiable" | §7.8, §9 limitation 1 — nothing on real data is labelled warranted | **conceded by design** |

Two items remain conceded. Both are stated as limitations rather than argued away.

---

## Rebuttal plan

**Concede immediately and without hedging:** scale (R2), real-data legitimacy (R4), and that the
effect sizes are moderate. Attempting to defend any of these would cost more credibility than it
buys.

**Answer with existing artefacts, no new compute:**
- R4's "1 epoch is barely an update" → agreement is 0.98 there and the ratio *declines* monotonically
  as the update strengthens; the light regime is the deployment-realistic one, and the claim is
  weakest, not strongest, where the update is largest.
- R2's "Cliff's δ = +1.00 with n = 3" → correct; the vision claim rests on consistency with two
  independent 10-seed settings, not on its own significance. Say so.
- R3's "noise-dominated explainers in the headline" → recompute the headline range over the five
  measurement-grade explainers (1.25–1.73) and table EG/LIME separately.

**Status of the four requested experiments (updated after Phase A–C):**

1. **R4's experiment — DONE (§7.5b).** Delta-Audit's 45 published settings re-run with a matched
   null: 64% fall below the resample floor, 33% clear it, median ratio 0.861. Their flagship
   examples survive, which is what makes the result credible rather than a hatchet job. This is now
   the paper's strongest single claim.
2. **R3's budget sweep — DONE (§8.0).** KernelSHAP needs ~16× its default sample budget and LIME
   ~10× theirs before `ν` falls below `ρ_null`; the *estimate* is stable across budgets, only its
   certifiability moves. Expected Gradients remains unresolved even at 512 samples.
3. **R2's scale probe — half done.** The **width sweep to 1024 is complete and ran on CPU** (it
   cost minutes, not GPU-hours): the ratio is stable across a 630x parameter range, 1.7k to 1.07M,
   so R2's specific worry that the ratio vanishes with capacity is answered. The remaining half --
   DistilBERT at 66M parameters and a real ResNet-18 -- is packaged as `kaggle/scale_probe.py` for
   a T4/P100 (not an H100; that assumption was wrong). Until it returns, *architectural* scale
   stays a stated limitation.
4. R1's operator-level bound, under linear-model assumptions — not attempted.
