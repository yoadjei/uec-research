# Internal attack review

Five simulated ICLR reviews of the paper **as produced**, written against the artefacts that
actually exist in `figures/`, `paper/tables/` and `results/`. Then the audit's §10 objection list,
each mapped to a concrete artefact or conceded. An objection with no artefact is a hole to fix, not
a rebuttal to write — three are conceded below.

---

## R1 — theory-oriented

**Summary.** Decomposition of attribution change across model updates against a closed-form
warranted-change reference, plus three propositions on which explainer classes inherit stability
from prediction stability.

**Strengths.** The category-error paragraph (§2.1) is correct and worth stating; it is verified
bitwise rather than asserted. Proposition 1's sharpness construction is the right move — the claim
that IG "inherits output stability" is common and false, and the paper shows exactly which part
survives (the aggregate, via completeness) and which does not (everything else). The empirical check
is unusually honest: 0 violations in 20,000 points against a *propagated* quadrature tolerance
rather than a guessed epsilon, and the authors note the bound is tight so it is not vacuous.
Recognising that Proposition 1 needs C¹ and that a ReLU net is not — with the O(1/n) vs O(1/n²)
measurement to prove the point matters — is the kind of care I rarely see.

**Weaknesses.** Propositions 1(i) and 3 are elementary; 1(i) is completeness plus a triangle
inequality. The genuinely novel content is 1(iii) and 2, and those are constructions rather than
deep results. ω is only exact for synthetic generators, so the theory-to-practice bridge is narrow.
I would like a statement about *which* update operators produce unwarranted change — a flat-minima
or weight-space-distance argument — rather than only an empirical sweep.

**Novelty** moderate. **Score 6, confidence 4.**
**Requested.** State plainly in §5 that 1(i) is elementary and that the content is in the sharpness.
Add the Shapley coalition remark to the main text, not only the appendix.

---

## R2 — empirical deep learning

**Strengths.** The matched-operator null is the right control and I have not seen it in this
literature. Figure 2b is the paper: the two controls order the update regimes in opposite directions
and the curves visibly cross. The placebo (no-shift) condition returning ratio ≈ 1.0 with intervals
covering 1 for all seven explainers is what convinces me the pipeline is not manufacturing an
effect. Reporting that Expected Gradients is noise-dominated rather than quietly dropping it is
good practice.

**Weaknesses.** Scale. MLPs on 20-dimensional synthetic data, ACS Income, and a 78k-parameter CIFAR
ResNet. I do not know whether any of this holds for a ViT or an LLM, and the fine-tuning-stability
literature suggests scale changes optimisation transients qualitatively. The vision experiment is a
sanity check by the authors' own admission and is the weakest section. Five seeds in the regime
sweep is thin, though ten in the main experiment is adequate.

**Score 6, confidence 4.**
**Requested.** Either one larger model or an explicit argument for why the mechanism should be
scale-invariant. State the CPU-only budget in the main text so the scope is not mistaken for a
choice.

---

## R3 — XAI specialist

**Strengths.** The related-work organisation by *what changes* is the clearest I have seen on this
topic, and the paper engages FASS, Delta-Audit, RSP and Mougan honestly rather than strawmanning
them. Crucially, the differentiation table is *computed*: the authors reimplement ROS, FASS-style
filtered distance and Delta-Audit's coupling and run them on their own checkpoints. Finding that
both prior metrics rank correct adaptation as worse than unwarranted drift (+45%, +89%) is a real
result and the strongest argument in the paper. The distance ablation showing τ = 1.0 pre-empts my
usual objection. I appreciate that the authors decline to define "reliability".

**Weaknesses.** The switch of primary distance from Spearman to ℓ₁ mid-project needs to be in the
main text as a decision with a reason, not buried in an ablation — a sceptic will read it as
distance-shopping. (The reason given, tie structure under a sparse reference, is sound and the
ranking is invariant anyway, so say it up front.) The choice of ε and the probe size are still
free parameters. No human study — which I actually think is correct here, but say why.

**Score 7, confidence 5.**
**Requested.** Promote the distance rationale to §6. Report `n_preserved` beside every headline
number.

---

## R4 — sceptical senior

**Summary.** "The Rashomon effect under fine-tuning, with a better control."

**Weaknesses.** My standing objection to this line is that it is underspecification with extra
steps, and I want to know what a practitioner does differently on Monday. The paper partly answers
me: the matched-null result is not obtainable from the Rashomon literature, and §7.3 shows the
Rashomon-style control gives the opposite ordering, which is a genuine correction rather than a
refinement. The result I cannot dismiss is §7.4 — attribution change of the same magnitude whether
the mechanism moved or not (p = 0.63 for IG) — because that is a statement about what these methods
can and cannot support, not about how much they wobble.

Remaining complaints. The headline ratios are 1.4–1.7, not 5; "1.5×" is a real effect but it is not
dramatic, and the paper should not let the reader inflate it. Warranted change on real data is
unverifiable and the authors admit it, which is honest but leaves the real-data section as
"direction replicates" rather than evidence for the central claim. The effect is largest for the
lightest updates, which invites the reply that a 1-epoch update is barely an update.

**Novelty** moderate-to-good, conditional on §7.4 surviving scrutiny.
**Score 6, confidence 4.**
**Requested.** A one-paragraph practitioner protocol. Address the "1 epoch is not an update"
objection head-on — the answer is that light incremental fine-tuning is the common deployment
pattern and predictions are 98% preserved there, but say it.

---

## R5 — broad / AC-facing

**Strengths.** Clear, well-scoped, and unusually candid about what it does not claim — the list of
six disclaimed claims is the kind of thing that makes me trust the rest. Reproducible on CPU with
seeds and a registry. Figure 1 is built from real checkpoints rather than drawn, which I appreciate.

**Weaknesses.** Dense. The three-way distinction between ν, ρ_null and ρ_seed is the crux and takes
too long to become clear; Figure 2 should be readable before the reader has internalised the
notation. H3's result is interesting enough to be promoted — "the one monitored signal that
correlates points the wrong way" is a better sentence than most of the abstract.

**Score 6, confidence 3.**
**Requested.** Move the H3 finding into the abstract. Add a one-line glossary box for the floors.

---

## AC discussion

Scores 6 / 6 / 7 / 6 / 6. All reviewers accept the matched-operator null as a genuine methodological
contribution, and R3 and R4 both single out §7.4 as the result that carries the paper. R2's scale
objection is real but is a limitation, not an error, and the authors state it. R4's "not dramatic"
point is fair and the paper should not oversell 1.5×.

The audit's own decision rule — accept as poster if the effect is ≥ 2× the floor with intervals, or
reject to TMLR if within 1.3× — is not the right rule for the paper that was actually written. The
headline ratio (1.4–1.7, intervals excluding 1, Cliff's δ ≥ 0.88, placebo at 1.0) is solid but
moderate. What carries it is not the ratio: it is (i) the control inversion in §7.3, (ii) the
magnitude-is-uninformative result in §7.4, and (iii) the demonstration that two published metrics
rank correct behaviour as worse. Those are qualitative claims that a larger effect size would not
strengthen.

**Lean accept as poster**, conditional on the §7.4 result surviving a check that the two shift
families really are matched on everything except ω.

---

## Objection map (audit §10 and §23)

| Objection | Where it is answered | Status |
|---|---|---|
| "Just another stability metric" | §4, terminology map; no metric is proposed, every distance is standard | answered |
| "Metric is arbitrary / distance-dependent" | §8, T5: τ = 1.0 across Spearman/cosine/ℓ₁, ε and φ | answered |
| "Evaluates XAI rather than advancing ML" | §1, §5; identifiability framing plus the method-class theorem | partly — R1 wants a statement about which operators |
| "Distribution shift insufficiently defined" | §2.3, §6; shifts are generated with known mechanism status | answered |
| "Conflates explanation and model instability" | §4, Fig. 2; ν, ρ_null and ρ_seed reported separately | answered |
| "Experiments too heterogeneous" | synthetic + ACS + one vision sanity check | answered |
| "No theory" | §5, three propositions, all numerically verified | answered |
| "Obvious" | §7.3 control inversion, §7.4 magnitude result | answered |
| "No evidence the metric tracks meaningful reliability" | reliability is never claimed (terminology map) | answered |
| "Already in prior work" | Appendix B, eight deltas with numbers | answered |
| "Attention is not explanation" | attention excluded by design | answered |
| "Benchmark too small" | not a benchmark paper | answered |
| "Ratio metric degenerate at Δf → 0" | §7.5: ROS reimplemented, swings 0.49–4.81 | answered |
| "Rashomon explains everything" | §7.3 and §8: reliable-partition ratio *rises* to 1.92 | answered |
| "Fine-tuning hyperparameters drive the result" | §7.3 sweep over lr and epochs, 600 runs | answered |
| "Probe set from shared support is unrealistic" | §6; it is the only set where both models are ID | answered |
| "ε-thresholding is arbitrary" | §8; monotone across 0.01–0.10, τ = 1.0 | answered |
| "Synthetic too simple" | it is the only place ω is exact; ACS is the demonstration | answered |
| "Why fine-tuning and not retraining?" | both regimes built (`scratch`); fine-tuning is primary | **partly — the scratch regime is computed but under-reported** |
| "What should a practitioner do?" | — | **conceded — needs the protocol paragraph R4 asks for** |
| "Does it hold at scale?" | §9 limitation 3 | **conceded — a limitation, not an answer** |

Three items are not fully discharged: the practitioner protocol, the under-reported
retrain-from-scratch regime, and scale. The first two are fixable in this draft; the third is a
scope statement.
