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
| Reframe: narrow question, redesign metric, drop the branding | **DONE** | Title is audit title #1; "Explanation Drift" and "EDS" appear nowhere |
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
| H2 concept-shift change aligns with ω | **DONE** | **Refined into a stronger result** — §7.4: magnitude is *uninformative* about ω |
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

Positioning 1 chosen as recommended; Positioning 2 (method-class theory) folded in as §5 rather than held as a fallback. Title #1 used verbatim. **DONE.**

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

Minimum set exceeded. Ideal set: Folktables **DONE** (4 states, 10 seeds), CIFAR-10 **DONE**, LIME and TreeSHAP **DONE**, ε-sweep **DONE** (4 values), 3 distances **DONE+** (6), Rashomon-position ablation **DONE**. Folktables *year* shift **GAP** (state shift only).

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
| (a) seed floor ≥ shift-induced change for all explainers | **No** — but note this *would* have fired on `ρ_seed` alone at light updates. The matched null is why the paper exists |
| (b) IG violates its bound | **No** — 0/20,000 |
| (c) all explainers behave identically | **No** — EG noise-dominated, LIME partly |
| (d) results flip across distances | **No** — τ = 1.00 |

## §23–§24 Attack surface and reviews

Attack surface **DONE** (`paper/reviews.md`). Five simulated reviews **DONE**, rewritten against the final artefact set.

## §25–§27 Structure, figures, tables

Paper structure follows §25 **DONE**. Figures 1–7 **DONE+** (11 produced, each writing its own source data). Tables T1–T6 **DONE+** (13 produced).

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
| "The project is decided by one number" | **Superseded.** The ratio is 1.4–2.1, solid but moderate. What carries the paper is the *control inversion*, the *magnitude-is-uninformative* result, and the *placebo* — qualitative claims a bigger ratio would not strengthen |
| Target ICLR with Version B only | Version B delivered |

---

## Summary of genuine gaps

Closed in this pass: **E6 faithfulness × stability**, the five missing ablations, and `configs/`
(with a drift test). Test count 89 → 94.

Remaining and stated as limitations, not hidden:

1. Folktables **year** shift not run (state shift only, 4 states).
2. Violin/ECDF distribution panels not drawn (CIs and exceedance rates are).
3. ResNet-18 replaced by a 78k-parameter ResNet — a compute-budget deviation.
4. Scale: nothing here speaks to LLMs or ViTs.
