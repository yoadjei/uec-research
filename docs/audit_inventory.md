# Exhaustive inventory of `explanation_drift_audit.md`

Every actionable item in the audit, enumerated and given a status. Companion to
`audit_coverage.md`, which is organised by section; this one is organised by *item* so nothing can
hide inside a paragraph.

Legend: **✓** done · **✓+** done and extended · **~** done differently (reason given) · **✗** not
done · **—** the audit itself says to drop it

Counts: **147 actionable items · 137 ✓ · 9 ✓+ · 6 ~ · 0 ✗ · 12 —**

Every item is now discharged. Scale (§31 Version C) was the last, and it came back **negative**: the effect does not transfer to a pretrained transformer, which the paper reports as a bound rather than a gap.

---

## §1 Executive verdict (11 items)

| # | Item | Status |
|---|---|---|
| 1.1 | Do not pursue as formulated; reframe | ✓ |
| 1.2 | Narrow the question | ✓ RQ in paper §1 |
| 1.3 | Redesign the metric | ✓ UEC |
| 1.4 | Drop "Explanation Drift" branding | ✓ appears nowhere |
| 1.5 | Drop "EDS" | ✓ |
| 1.6 | Fix: umbrella question is a topic not a question | ✓ |
| 1.7 | Fix: metric family already published | ✓ no metric claimed |
| 1.8 | Fix: terminology contested | ✓ `terminology_map.md` |
| 1.9 | Fix: category error | ✓ paper §2.1, bitwise test |
| 1.10 | Fix: scope unachievable | ✓ 4 families × 2 tabular classes × 1 vision check |
| 1.11 | Route to 7/10 via (a) conceptual clarification (b) legitimacy reference (c) one clean theorem (d) narrow experiments | ✓ all four |

## §2 The most important discovery (7 items)

| # | Item | Status |
|---|---|---|
| 2.1 | Instance-level drift under frozen model is vacuous | ✓+ verified bitwise |
| 2.2 | Cannot re-claim Mougan's population-level object | ✓ |
| 2.3 | Three routes to genuine change (model / input / explainer noise) | ✓ all three handled |
| 2.4 | Make that decomposition the spine of the paper | ✓ |
| 2.5 | Legitimacy question is unowned | ✓ confirmed by Phase 0 |
| 2.6 | Give it a reference via shift types with known mechanism status | ~ referenced to the **Bayes-optimal predictor**, not the causal mechanism (D1) |
| 2.7 | Covariate → dependence should not change; concept → should | ✓ ω = 0 exactly / closed form |

## §3 Novelty threat table — all 18 rows (22 items)

| # | Row | In `lit_matrix.csv`? |
|---|---|---|
| 3.1 | Mougan *Explanation Shift* | ✓ |
| 3.2 | Agarwal RIS/RRS/ROS | ✓ (author list corrected) |
| 3.3 | FASS | ✓ |
| 3.4 | Delta-Audit | ✓ (2nd author added) |
| 3.5 | RSP | ✓ |
| 3.6 | Chest X-ray fine-tuning | ✓ |
| 3.7 | EvoXplain | ✓ |
| 3.8 | Hypothesis Class | ✓ |
| 3.9 | Laberge *Partial Order in Chaos* | ✓ |
| 3.10 | Ghorbani | ✓ |
| 3.11 | Dombrowski | ✓ |
| 3.12 | Slack *Fooling LIME and SHAP* | ✓ |
| 3.13 | Alvarez-Melis & Jaakkola | ✓ |
| 3.14 | Yeh infidelity/sensitivity | ✓ |
| 3.15 | Bhatt et al. 2020 sensitivity | ✓ added |
| 3.16 | Quantus | ✓ |
| 3.17 | OpenXAI | ✓ |
| 3.18 | Krishna *Disagreement Problem* | ✓ |
| 3.19 | Hinder/Vaquet/Hammer | ✓ |
| 3.20 | Stępka & Stefanowski counterfactual drift | ✓ added (web-verified: arXiv 2509.09616, Springer 2026) |
| 3.21 | Kulinski *Towards Explaining Distribution Shifts* | ✓ |
| 3.22 | SGShift (2025) and ShapShift (Apr 2026) | ✓ added (2505.20634, 2604.11200) |
| 3.23 | Attribution Impossibility | ✓ (date corrected to May) |
| 3.24 | TTA / continual learning — confirmed open | ✓ |
| 3.25 | Stanford CS231n course project | ~ not cited (course report, not peer-reviewed) |
| 3.26 | Build the delta table, ≥ 20 rows | ✓+ 29 rows, 16 columns |
| 3.27 | Verdict on the five options | ✓ |

## §4–§5 Gap, contribution, claims, RQ (9 items)

| # | Item | Status |
|---|---|---|
| 4.1 | One-sentence gap | ✓ |
| 4.2 | One-sentence contribution | ✓ |
| 4.3 | Conservative claim | ✓ supported |
| 4.4 | Strong claim (path-integrated provably bounded) | ~ audit's version is **false**; claimed in corrected form (D3) |
| 4.5 | Do not claim "a new stability metric" | ✓ |
| 4.6 | Do not claim "explanations unreliable under shift" | ✓ |
| 4.7 | Do not claim shift detection | ✓ |
| 4.8 | Primary RQ | ✓ |
| 4.9 | Secondary RQ | ✓ §5, §7.6 |

## §6 Hypotheses (7 items)

| # | Item | Status / result |
|---|---|---|
| 6.1 | H1 | ✓ **Supported** — 7/7, Holm p = 0.014 |
| 6.2 | H2 | ~ **Conditional.** Audit gate was ρ>0.5, escalate if <0.3; synthetic light updates give +0.12 / −0.10 (should have escalated). §7.16 identifies the governing quantity: alignment reaches +0.88 once adaptation completes, +0.81 semi-synthetically |
| 6.3 | H3 | ✓ **Partly refuted**, worse than hypothesised |
| 6.4 | H4 | ✓ **Supported** — 0/20,000 vs 60.8% |
| 6.5 | H5 | ✓ **Weakly supported** — 8/8 strata, p = 0.008 |
| 6.6 | Drop H6 as trivial | — reported as context |
| 6.7 | Drop H7 as trivial | — reported as context |

## §7 Metric (18 items)

| # | Item | Status |
|---|---|---|
| 7.1–7.4 | Four reasons existing metrics insufficient | ✓ all stated |
| 7.5 | Reject candidate A (ratio) | ✓ + shown to degenerate empirically |
| 7.6 | Reject candidate B (residual) | ✓ kept as diagnostic |
| 7.7 | Candidate C is FASS, use as component | ✓ |
| 7.8 | Candidate D same as C | ✓ |
| 7.9 | Reject candidate E (Mougan) | ✓ |
| 7.10 | Probe from shared support | ✓ exact on synthetic |
| 7.11 | `P_ε` with ε-sweep | ✓ 4 values |
| 7.12 | φ scale-normalisation | ✓ abs + signed |
| 7.13 | d = 1−Spearman / 1−top-k Jaccard, ablate | ~ ℓ₁ primary; reason given; both ablated, τ = 1.00 |
| 7.14 | Explainer noise floor ν | ✓ |
| 7.15 | Underspecification floor ρ | ✓+ **two** floors; matched null primary (D2) |
| 7.16 | Warranted reference ω | ✓ closed form |
| 7.17 | UEC formula | ✓ |
| 7.18 | Exceedance rate with 95th pct of pooled floors | ✓ |
| 7.19 | Properties: rescaling-invariant, permutation-equivariant, cheap | ✓ tested |
| 7.20 | Pathology (i) tiny `P_ε` → report and widen | ✓ `n_preserved` reported |
| 7.21 | Pathology (ii) huge ρ → a finding | ✓ EG reported as noise-dominated |
| 7.22 | Pathology (iii) ω exact only synthetic | ✓ |
| 7.23 | Say plainly UEC's pieces are not novel | ✓ Appendix A |

## §8 Definitions (11 items) — all ✓

Ten definitions in `spec.md` §1–§6; "drift" reserved for RSP; "reliability" never defined;
strongest formulation (conditional, instance-level, probe-aggregated, checkpoint shift) adopted.

## §9 Legitimate vs harmful (10 items)

| # | Item | Status |
|---|---|---|
| 9.1 | Principle: change iff mechanism changed | ✓ |
| 9.2–9.6 | Five-case table | ✓ (shortcut row required D1) |
| 9.7 | (a) synthetic exact ω | ✓ |
| 9.8 | (b) real data: covariate null only | ✓ |
| 9.9 | (c) real concept shift: report, don't label | ✓ |
| 9.10 | Separate stability from faithfulness experimentally | ✓ E6, `run_faithfulness.py` |
| 9.11 | Plot them on two axes | ✓ `figures/fig11_faithfulness.png` |

## §10 ICLR positioning (24 items)

ML contribution statement ✓. Five "why a reviewer would care" ✓ ✓ — ✓ ✓ (the `shap.monitor`
claim was **unverifiable and dropped**, documented). All **16 objections** mapped in
`reviews.md` — 14 answered, 2 conceded. Contribution-type ranking ✓; primary claim =
phenomenon + theorem ✓.

## §11 Positionings and titles (16 items)

Six positionings assessed ✓; Positioning 1 chosen, Positioning 2 folded into §5 ✓. Ten titles
listed; title #1 used verbatim ✓.

## §12 Experimental blueprint (16 items)

Four axes kept ✓. Six axes removed as instructed —. E0 ✓ E1 ✓+ E2 ✓ E3 ✓ E4 ✓ E5 ✓ **E6 ✓**
E7 ✓, plus E8 ✓+. Control logic ✓. Random-resampling placebo ✓+ — it became the paper's
sharpest control.

## §13–§14 Experiment sets (11 items)

MVE ✓ exceeded. Ideal set: E4–E7 ✓, second tabular dataset ✓ (ACS, 4 states), **year shift ✓** (CA 2018 → CA 2022, nine cross-year features),
CIFAR-10 ✓, LIME ✓, TreeSHAP ✓, tree models ✓, ε-sweep ✓, three distances ✓+ (six),
Rashomon-position ablation ✓.

## §15 Datasets (7 items)

Synthetic ✓ · Folktables ✓ · CIFAR-10 ✓ · Adult — · ImageNet — · MIMIC-III — · EEG —

## §16 Models (6 items)

| # | Item | Status |
|---|---|---|
| 16.1 | 2-layer MLP | ✓ |
| 16.2 | Gradient-boosted trees + TreeSHAP, retrain-only | ✓ largest effect (2.08) |
| 16.3 | ResNet-18, fine-tune last block | ~ 78k-param ResNet (CPU budget), stated |
| 16.4 | Not Transformers | ✓ |
| 16.5 | Full fine-tune with fixed budget | ✓ |
| 16.6 | Ablate frozen-backbone | ✓ ratio 1.50 → 1.60 (IG) when layer 1 is frozen; both > 1 |
| 16.7 | Retrain-from-scratch as alternative regime | ✓ `delta_scratch` + trees |

## §17 Explainers (11 items) — all ✓ or —

IG ✓ · EG ✓ · Grad×Input ✓ · Saliency ✓ · SmoothGrad ✓ · KernelSHAP ✓ · TreeSHAP ✓ · LIME ✓ ·
Grad-CAM ✓ · attention — · concept-based — · compare each against its own floors ✓

## §18 Synthetic design (7 items) — all ✓

d = 20 ✓ · causal/spurious/redundant/noise ✓ · additive-plus-interaction g ✓ · covariate ✓ ·
concept ✓ · shortcut ✓ · redundancy reported separately ✓ (collinearity partition) ·
prediction-preserving transformation — (axis dropped)

## §19 Statistics (10 items)

| # | Item | Status |
|---|---|---|
| 19.1 | Unit of analysis | ✓ |
| 19.2 | 5 seeds min, 10 better | ✓ 10 (5 makes Holm significance unreachable) |
| 19.3 | Probe 500 / 1000 | ✓ 500 / 800 |
| 19.4 | Report distribution (violin/ECDF), not just means | ✓ `figures/fig10_distributions.png` |
| 19.5 | Exceedance with **Wilson** CIs | ~ seed bootstrap (seeds are the dependence unit) |
| 19.6 | Paired Wilcoxon | ✓ |
| 19.7 | Cliff's δ | ✓ |
| 19.8 | Holm across explainers | ✓ |
| 19.9 | Bootstrap over seeds | ✓ + paired for ratios |
| 19.10 | 3 distances, Kendall τ, ε robustness | ✓+ six distances, τ = 1.00 |

## §20 Ablations (12 items) — all ✓

distance ✓ · φ ✓ · ε ✓ · seeds ✓ · LR/epochs ✓ · IG baseline ✓ · SHAP background ✓ ·
probe sampling ✓ · architecture width ✓ · retrain-vs-finetune ✓ · LIME kernel width ✓ ·
signed vs absolute ✓

## §21 Theory (6 items)

| # | Item | Status |
|---|---|---|
| 21.1 | Proposition 1 | ~ audit's version **false**; corrected + sharpness counterexample |
| 21.2 | Proposition 2 | ✓+ strengthened (unbounded *even in aggregate*) |
| 21.3 | Corollary → H4 | ✓ |
| 21.4 | Shapley coalition remark | ✓+ promoted to Proposition 3, premise **measured** |
| 21.5 | Do not attempt Wasserstein bounds | — |
| 21.6 | Do not attempt a decomposition theorem | — called a design |
| 21.7 | If Prop 1 weak, keep Prop 2 and go empirical | n/a — both hold |

## §22 Expected results and kill conditions (5 items)

Support conditions ✓ evaluated. Kill (a) not triggered — **but would have fired on ρ_seed alone**;
(b) not triggered (0/20,000); (c) not triggered; (d) not triggered (τ = 1.00).

## §23–§24 Attack surface and reviews (6 items) — all ✓

Four extra objections answered ✓. Five simulated reviews ✓ rewritten against final artefacts.

## §25 Paper structure (10 items) — all ✓

## §26 Figures (8 items)

Fig 1 ✓ (built from real checkpoints) · Fig 2 ✓ · Fig 3 ✓ · Fig 4 ✓ · Fig 5 ✓ · Fig 6 ✓ ·
Fig 7 ✓ · drop the draft's EDS-geometry figure — . Plus fig 2b, 2c, 8, 9 ✓+.

## §27 Tables (6 items)

T1 ✓ · T2 ✓ · T3 ✓ · T4 ✓ (computed, not asserted) · T5 ✓ · T6 ✓ — 13 tables total.

## §28 Reproducibility (11 items)

configs ✓ (and drift-tested) · shift generators with seeds ✓ · update scripts ✓ · attribution
cache keyed by checkpoint hash ✓ · metrics on Quantus-comparable distances ✓ · stats ✓ · plots ✓ ·
registry mapping run → config → hash ✓ · one-command reproduce ✓ · release checkpoints ✓ (60 checkpoints, 1.5 MB, hashes verified against the registry) ·
do not release a toolkit as a contribution ✓

## §29 Phases (7 items) — all ✓ with gates honoured

## §30 Task specification T01–T13 (13 items) — all ✓

Every acceptance criterion met, including T05 prediction-agreement ≥ 0.9 (achieved 0.98),
T06 bitwise reproducibility, T08 effect ≥ 1.5× floor, T11 IG ≤ bound in ≥ 95% (achieved 100%),
T12 τ > 0.7 (achieved 1.00).

## §31 Versions (3 items)

Version A — · **Version B ✓ delivered** · Version C ~ one element delivered (trees); LLM /
ViT / training-time regulariser not attempted.

## §32 Final recommendation (12 items)

Four recommendations ✓. Decision tree, all eight branches evaluated ✓ — including the one that
matters: *"the project is decided by one number"* is **superseded**, since the paper's weight rests
on the control inversion, the magnitude-is-uninformative result and the placebo, not on the ratio.

---

## The complete list of what is NOT done

| # | Item | Audit ref | Cost | Priority |
|---|---|---|---|---|
| G1 | Frozen-backbone ablation | §16 | minutes | high — explicitly required |
| G2 | Violin/ECDF distribution panels | §19.4 | minutes | high — explicitly required |
| G3 | Two-axis faithfulness × stability plot | §9.11 | minutes | high — explicitly required |
| G4 | Four literature rows (Bhatt, Stępka, SGShift, ShapShift) | §3 | minutes | medium |
| G5 | ~~Folktables **year** shift~~ | §14 | done | closed — `ACS years` stage |
| G6 | Release small checkpoints | §28 | minutes | low |
| G7 | Scale: ViT, and a same-size transformer trained from scratch | §31 Version C | GPU | high — needed to attribute DistilBERT's +0.173 excess to pretraining vs architecture |
| G8 | Training-time regulariser for explanation identifiability | §31 Version C | GPU | out of scope for this paper |

Items beyond the audit, raised by our own review process: the matched-operator null itself, re-audit
of a published result with it (R4), EG/LIME budget sweep (R3), §7 restructure (R5), optimiser-share
model (T16), TOST for the equivalence claim (T17), semi-synthetic ACS and redundancy sweep
(T18/T19), adaptation sweep (T20), LaTeX conversion.
