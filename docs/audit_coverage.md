# Coverage checklist against `explanation_drift_audit.md`

Every section of the audit, what was done, and where the evidence is. Status codes:

- **DONE** — implemented as specified
- **DONE+** — implemented and extended beyond the spec
- **CHANGED** — deliberately done differently; the reason is given and it is a science decision
- **GAP** — specified but not done
- **DROPPED** — the audit itself says to drop it

---

## §1 Executive verdict — reframe, don't pursue as written

| Audit instruction | Status | Evidence |
|---|---|---|
| Reframe: narrow question, redesign metric, drop the branding | **DONE** | "Explanation Drift" and "EDS" appear nowhere. Title was audit title #1 until the final pass; it is now *The Control Decides the Answer*, because the matched null — which the audit did not anticipate — turned out to be the contribution that survives every scoping |
| Drop "EDS" as a metric | **DONE** | `docs/terminology_map.md` §"Claims we may not make" |
| Scope 5×5×4×3 is unachievable | **DONE** | 2 tabular model classes, 1 vision sanity check, 4 shift families |

## §2 The most important discovery

| Item | Status | Evidence |
|---|---|---|
| §2a Category error: frozen-model instance-level drift is vacuous | **DONE+** | Paper §2.1; verified **bitwise**, not asserted: `tests/test_harness.py::test_frozen_model_has_no_instance_level_change` |
| §2b Give the legitimacy question a reference | **DONE, CHANGED** | ω exists, but referenced to the **Bayes-optimal predictor per environment**, not the causal mechanism (departure D1). The audit's own §9 table is self-contradictory otherwise: under shortcut removal the causal mechanism is unchanged, so a causal ω would label correct adaptation "unwarranted" |

## §3 Novelty threat assessment (18 rows)

| Item | Status | Evidence |
|---|---|---|
| Delta table ≥ 20 rows | **DONE+** | `docs/lit_matrix.csv`, **25 rows**, 16 columns, zero empty fields |
| Verify the closest papers | **DONE+** | All 8 web-verified. **3 citation errors in the audit corrected**: ROS author list was the OpenXAI set; Attribution Impossibility is arXiv 2605 (May) not April; Delta-Audit has a second author |
| — | **DONE+** | Found **2 papers the audit missed**: *Rethinking Robustness* (2512.06665), *MCal* (2603.04831) |
| Terminology map | **DONE** | `docs/terminology_map.md` |

## §4–§5 Gap, contribution, claims, research question

| Item | Status | Evidence |
|---|---|---|
| One-sentence gap and contribution | **DONE** | Paper §1 |
| Conservative claim | **DONE** | §7.2, supported |
| Strong claim (path-integrated provably bounded) | **CHANGED** | The audit's version is **false**. Only the *aggregate* is bounded (departure D3, with a sharpness counterexample). Claimed in the corrected form |
| Three overclaims to avoid | **DONE** | None made; all three explicitly disclaimed in Appendix A |
| Primary RQ adopted | **DONE** | Paper §1 |
| Secondary RQ (which classes inherit stability) | **DONE** | §5, §7.6 |

## §6 Hypotheses

| ID | Status | Result |
|---|---|---|
| H1 unwarranted change > floor | **DONE** | **Supported.** 7/7 explainers, Holm p = 0.014, Cliff's δ ≤ +0.98 |
| H2 concept-shift change aligns with ω | **DONE** | **Conditional, and the condition is identified.** The audit's gate was ρ>0.5 (escalate if <0.3). On synthetic light updates we get +0.12 and −0.10, which *should* have triggered escalation and instead became a headline. §7.16 resolves it: alignment is governed by adaptation completeness Δ/ω, reaching **+0.88** at completeness 0.99 and **+0.81** semi-synthetically. The audit's gate passes where the model finishes adapting, and fails where it does not — which is the deployment regime |
| H3 not predicted by accuracy/ECE/agreement | **DONE** | **Partly refuted**, and worse than the hypothesis: agreement correlates *positively* (§7.7) |
| H4 method-class asymmetry | **DONE** | **Supported.** 0/20,000 IG violations; 60.8% Grad×Input |
| H5 Rashomon-position updates | **DONE** | **Weakly supported.** 8/8 strata, sign test p = 0.008; confounded marginally (§7.8b) |
| H6, H7 dropped as trivial | **DONE** | Reported as context only |

## §7 Metric

| Item | Status | Evidence |
|---|---|---|
| Reject candidates A–E | **DONE** | None used; ROS reimplemented only to show it degenerates (§7.5) |
| UEC = mean Δ − ω − max(ν, ρ) | **DONE** | `src/uec/metrics/uec.py` |
| Exceedance rate | **DONE** | Reported throughout |
| ν noise floor | **DONE** | Zero for deterministic explainers, verified bitwise |
| ρ underspecification floor | **DONE+, CHANGED** | Two floors. `ρ_seed` as specified, **plus `ρ_null` matched-operator null as primary** (departure D2) — the audit's control confounds operator with distribution |
| Pathology (ii): if ρ huge, report as finding | **DONE** | Expected Gradients is noise-dominated and reported as such (§7.1) |
| "Do not call it a new metric" | **DONE** | Stated in Appendix A |

## §8 Definitions

All ten definitions implemented in `docs/spec.md` §1–§6. "Drift" reserved for RSP's temporal object and not used for ours. "Reliability" never defined or claimed. **DONE.**

## §9 Legitimate adaptation vs harmful drift

| Item | Status | Evidence |
|---|---|---|
| Five-case legitimacy table | **DONE, CHANGED** | Paper §4; the shortcut row required the Bayes reference to come out right |
| Operationalisation (a) synthetic exact ω | **DONE** | Closed form, `tests/test_synthetic.py` |
| (b) real data: only covariate ω = 0 | **DONE** | §7.8; nothing on real data labelled warranted |
| (c) real concept shift: report, don't label | **DONE** | §7.8 |
| **Separate stability from faithfulness experimentally** | **DONE** | `run_faithfulness.py`, paper §7.7b. The effect survives on points where the explainer is faithful to *both* checkpoints (ratio 1.567 vs 1.563), and corr(Δ, faithfulness) ∈ [−0.08, +0.03] |

## §10 ICLR positioning and objection table

All 16 objections mapped in `paper/reviews.md`. Three conceded rather than argued away. **DONE.**

## §11 Alternative positionings

Positioning 1 chosen as recommended; Positioning 2 (method-class theory) folded in as §5 rather than held as a fallback. Title #1 used until the final pass, then replaced (see §1). **DONE.**

## §12 Experimental blueprint

| Exp | Purpose | Status |
|---|---|---|
| E0 | noise floor ν | **DONE** |
| E1 | underspecification floor ρ | **DONE+** (two floors) |
| E2 | unwarranted change, covariate | **DONE** |
| E3 | warranted change, concept | **DONE** |
| E4 | shortcut removal | **DONE** |
| E5 | Rashomon position (LR/epoch sweep) | **DONE** (600-run grid) |
| **E6** | **faithfulness × stability** | **DONE** — 20/20 cells; also yields a second result: fidelity to the mechanism falls under shortcut shift but is flat under covariate shift |
| E7 | theory check | **DONE** |
| E8 | frozen-model sanity panel *(added)* | **DONE+** |
| — | placebo / random-resampling pseudo-shift | **DONE+** — the `none` family, and it is the paper's sharpest control |

## §13–§14 Minimum and ideal experiment sets

Minimum set exceeded. Ideal set: Folktables **DONE** (4 states, 10 seeds), CIFAR-10 **DONE**, LIME and TreeSHAP **DONE**, ε-sweep **DONE** (4 values), 3 distances **DONE+** (6), Rashomon-position ablation **DONE**. Folktables *year* shift **DONE** (2018 → 2022).

## §15 Datasets

Synthetic **DONE** (required). Folktables **DONE** (recommended). CIFAR-10 **DONE** (optional). Adult, ImageNet, MIMIC-III, EEG correctly **DROPPED**.

## §16 Models

| Audit | Status |
|---|---|
| 2-layer MLP | **DONE** |
| Gradient-boosted trees, retrain-only | **DONE** — `run_trees.py`, exact TreeSHAP, and it gives the largest effect (2.08) |
| ResNet-18, fine-tune last block | **CHANGED** — a 78k-parameter ResNet on CPU, and the update replays old data (see §7.9). ResNet-18 was not affordable on this budget; stated as a limitation |
| Smooth activation | **CHANGED, added** — Proposition 1 assumes C¹ and a ReLU net is not one; ReLU makes IG completeness err at O(1/n) |

## §17 Explainers

IG, Expected Gradients, Gradient×Input, Saliency, SmoothGrad, KernelSHAP, LIME, TreeSHAP, Grad-CAM — all **DONE**. Attention and concept-based methods correctly **DROPPED**. Cross-type comparison done only via each explainer's own floors, as instructed. **DONE.**

## §18 Synthetic design

Causal / shortcut / redundant / noise blocks, covariate / concept / shortcut families, redundancy as the collinearity case — all **DONE**. Ground-truth attributions verified against quadrature to 1e-9 and against empirical label rates. Prediction-preserving input transformation correctly **DROPPED**.

## §19 Statistics

| Item | Status |
|---|---|
| Probe 500 (synthetic) / 800–1000 (tabular) | **DONE** |
| 5 seeds min, 10 better | **DONE** — 10 (5 would make Holm-corrected significance *unreachable*) |
| Paired Wilcoxon, Cliff's δ, Holm | **DONE** |
| Seed bootstrap for CIs; paired bootstrap for ratios | **DONE** |
| 3 distances, Kendall τ | **DONE+** — 6 distances, τ = 1.00 |
| Report distribution not just means | **PARTIAL** — CIs and exceedance reported; violin/ECDF panels not drawn |
| Exceedance with **Wilson** CIs | **CHANGED** — seed bootstrap used instead, since seeds are the unit of dependence and Wilson assumes independent Bernoulli trials |

## §20 Ablations

| Required | Status |
|---|---|
| distance | **DONE** (τ = 1.00) |
| normalisation φ | **DONE** |
| ε | **DONE** (4 values) |
| number of seeds | **DONE** (5 vs 10 in sweep vs main) |
| fine-tuning LR / epochs | **DONE** (600-run grid) |
| **IG baseline** | **DONE** — zeros 1.50 vs target-mean 2.34; our choice is conservative |
| **SHAP background set** | **DONE** — 1.46–1.47 across 10/25/50 |
| **probe sampling (shared vs target-only)** | **DONE** — source-only 1.23, shared 1.50, target-only 1.81; our choice is the middle |
| Optional: architecture width | **DONE** — 1.45–1.57 across 32/64/128 |
| Optional: retrain vs finetune | **DONE** (`delta_scratch`, §7.4) |
| Optional: LIME kernel width | **DONE** — 1.33–1.69 |
| Optional: signed vs absolute | **DONE** |

## §21 Theory

| Item | Status |
|---|---|
| Proposition 1 | **CHANGED** — the audit's claim that IG inherits output stability is false; only the aggregate does. Sharpness counterexample supplied |
| Proposition 2 | **DONE** — and strengthened: gradients are unbounded *even in aggregate*, which is the actual asymmetry |
| Corollary → H4 | **DONE** |
| Shapley remark | **DONE+** — promoted to Proposition 3 with the coalition premise **measured** |
| "Do not attempt" list | **DONE** — no Wasserstein bounds, no decomposition theorem; the decomposition is called a design |
| Numeric verification | **DONE+** | 11 tests; a proof failing its check is treated as a wrong proof |

## §22 Expected results and kill conditions

| Kill condition | Triggered? |
|---|---|
| (a) seed floor ≥ shift-induced change for all explainers | **YES on `ρ_seed`, and we say so in the main text.** Headline Δ/ρ_seed median **0.31** — the effect is *inverted* against the audit's control. §7.3 confronts this directly: the placebo, whose answer is fixed in advance, is what disqualifies ρ_seed (0.98 correct vs 1.90 invented). This is the paper's largest reviewer exposure |
| (b) IG violates its bound | **No** — 0/20,000 |
| (c) all explainers behave identically | **No** — EG noise-dominated, LIME partly |
| (d) results flip across distances | **No** — τ = 1.00 |

## §23–§24 Attack surface and reviews

Attack surface **DONE** (`paper/reviews.md`). Five simulated reviews **DONE**, rewritten against the final artefact set.

## §25–§27 Structure, figures, tables

Paper structure follows §25 **DONE**. Figures 1–7 **DONE+** (16 produced, each writing its own source data and each rebuildable by `make_figures.py` — three had lost their builders and were restored). Tables T1–T6 **DONE+** (25 produced).

## §28 Reproducibility

`shifts/` generators **DONE**, update scripts **DONE**, attribution cache keyed by checkpoint hash **DONE**, metrics **DONE**, stats **DONE**, plots **DONE**, `registry.csv` **DONE**, one-command reproduce **DONE**, "do not release a toolkit as a contribution" **DONE**. `configs/*.yaml` **DONE** — and they are *tested* against the runners' argparse defaults (`tests/test_configs.py`), so a config that drifts from what the code does fails CI rather than misleading a reader. Running any runner with no arguments now reproduces the paper.

## §29–§30 Phases and task spec

Phases 0–6 all executed. T01–T13 all discharged; T06's "deterministic explainers reproduce bitwise" is an actual test, and T05's "prediction agreement ≥ 0.9" is an actual gate that the design was tuned to meet.

## §31 Versions

**Version B (Strong ICLR) delivered, plus one Version-C element**: the tree/TreeSHAP arm goes beyond B and answers "is this a gradient artefact?".

## §32 Final recommendation and decision tree

| Decision point | Outcome |
|---|---|
| Kill the name and the metric | Done on day one |
| Phase 0 before compute | Done; gap survived |
| MVE before anything else | Done; gate passed |
| "The project is decided by one number" | **Superseded, and we lose on the audit's version of it.** Against ρ_seed the number is 0.31 (audit: reject below 1.3). Against ρ_null it is 1.4–2.1. The paper argues the denominator, not the numerator, and rests that argument on the placebo |
| Target ICLR with Version B only | Version B delivered |

---

## Beyond the audit

Five pieces of work the audit did not ask for. The first is the paper's contribution; the rest exist
because reviewing our own claims turned up holes the audit could not have foreseen.

| Work | Why it exists | Where |
|---|---|---|
| **Matched-operator null `ρ_null`** | The audit specified only `ρ_seed`. That control fails a no-shift placebo (reports 1.90 where the answer is 1.00), so the audit's own decisive number was measured against a broken reference | §7.2, §7.3 |
| **Re-audit of a published result** | The audit says to differentiate from Delta-Audit; applying our control to their 45 settings is stronger than a table comparison | §7.7 |
| **Optimiser-share model (T16)** | The 0.48–0.98 share range was a documented correlation with no mechanism. Capacity turns out not to predict it (p = 0.94 over 630×); update strength does | §7.12b |
| **Semi-synthetic ACS + redundancy sweep (T18, T19)** | The audit's ω is exact only on a generator that supplies its own covariates. Real covariates with a known mechanism test whether the closed form is a Gaussian artefact (it is not: exact to 1.1e-14) | §7.15 |
| **Adaptation sweep (T20)** | §7.15 produced a contradiction the audit never anticipated. Completeness Δ/ω resolves it and converts H2 from an unexplained reversal into a conditional claim | §7.16 |

Two of these were prompted by errors we caught in our own design: the first semi-synthetic tilt was
calibrated too weakly (domain AUC 0.759 against the synthetic 0.902) and would have produced a false
null, and an early version drew the null without replacement and the treatment with it, handicapping
the treatment. Both are recorded in the runners' comments so the corrections are auditable.

## Summary of genuine gaps

Closed in this pass: **E6 faithfulness × stability**, the five missing ablations, and `configs/`
(with a drift test). Test count 89 → 101; 14 figures, 25 tables, 30 literature rows.

Remaining and stated as limitations, not hidden:

1. ResNet-18 replaced by a 78k-parameter ResNet — a compute-budget deviation.
2. Scale: DistilBERT is covered, but nothing here speaks to ViTs, and the +0.173 excess over the
   from-scratch curve cannot be attributed to pretraining rather than architecture without a
   same-size transformer trained from scratch (audit §31 Version C).
3. Completeness (§7.16) is characterised on two shift families in one generator plus one
   semi-synthetic setting. Whether it governs tracking elsewhere is untested.

Closed since the previous pass: Folktables **year** shift (2018 → 2022, `ACS years` stage),
and the **violin/ECDF distribution panels** (audit §19.4) — the builder existed but was
not wired into `make_figures.py`, so Fig. 10 had been rendering a weaker median/IQR
summary instead.
