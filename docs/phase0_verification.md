# Phase 0 — Novelty Resolution and Gate Verdict

**Date:** 29 Aug 2026. **Method:** direct web verification of every paper the audit treats as a
critical or high threat, plus adversarial searches designed to *find* a paper that already owns the
legitimacy question. Verification matters here because the audit cites work published after this
executor's knowledge cutoff (Jan 2026); those citations could not be taken on trust.

**Gate question:** does any existing paper provide a reference for *how much an explanation should
change* under a given shift — i.e. a criterion separating warranted from unwarranted change?

**Verdict: NO. The gap survives. Proceed with Positioning 1.**

---

## 1. Verification of the audit's closest works

| Audit row | Claimed identity | Verified? | Correction | Legitimacy reference? |
|---|---|---|---|---|
| 1 | Mougan et al., *Explanation Shift*, TMLR 2025 | **Yes** — arXiv:2303.08081, TMLR 01/2025, dblp `journals/tmlr/MouganBKTS25` | Audit says "arXiv 2022/23"; actual arXiv posting is Mar 2023. Final TMLR author list drops David Masip. | **None** — explanation shift is a *detector* of data shift, never asks if change is legitimate |
| 2 | Agarwal et al., RIS/RRS/ROS, arXiv 2203.06877 | **Yes** — ICLR 2022 workshop; authors Agarwal, Johnson, Pawelczyk, Krishna, Saxena, Zitnik, Lakkaraju | Audit's author list was wrong (it listed the OpenXAI author set). Corrected. | **None** — ROS normalises by output change; a ratio, not a reference |
| 3 | FASS, Apr 2026 | **Yes** — arXiv:2604.02532, 2 Apr 2026, Subramaniakuppusamy & Gajjar (GWU) | Confirmed verbatim: prediction-invariance filtering, SSIM + rank correlation + top-k Jaccard, IG/GradientSHAP/Grad-CAM/LIME, 4 architectures, ImageNet-1K/COCO/CIFAR-10, "up to 99% of pairs involve changed predictions" | **None** — residual after filtering is called "instability" by fiat |
| 4 | Delta-Audit, Aug 2025 | **Yes** — arXiv:2508.19589, 27 Aug 2025, Hemmat & Fatemi | Audit omits the second author. Metrics confirmed: L1/Top-k/entropy, rank-overlap@10, JSD, Delta Conservation Error, Behaviour–Attribution Coupling. Scope is 45 settings over 5 classical families × 3 small UCI datasets. | **None** — "risky reliance redistribution" is a heuristic label, not a reference |
| 5 | RSP, Jan 2026 | **Yes** — arXiv:2601.11625, Dhayalkar (ASU), under ACL ARR review | Confirmed: *explanation drift* = epoch-to-epoch change in normalised token attributions on a fixed probe set, measured by 1 − Spearman. This is the term-owner. | **None** — drift is a training-time signal for checkpoint selection |
| 6 | Chest X-ray fine-tuning drift, Apr 2026 | **Yes** — arXiv:2604.08513, 9 Apr 2026, Elangovan et al. | Confirmed: "semantic drift", reference-free metrics, LayerCAM vs Grad-CAM++ stability rankings reverse under converged performance. | **None** — explicitly *reference-free* |
| 7 | EvoXplain, Dec 2025 | **Yes** — arXiv:2512.22240, Bensmail (Hertfordshire) | Sole author. Scope smaller than the audit implies: Breast Cancer + COMPAS, LogReg + RF. Now has a commercial arm and a UK provisional patent. | **None** — makes multiplicity visible, does not adjudicate it |
| 8 | Hypothesis Class Determines Explanation, Mar 2026 | **Yes** — arXiv:2603.15821, 16 Mar 2026, Thackshanaramana B | Confirmed: 24 datasets, "Explanation Lottery", "Agreement Gap", proposes an Explanation Reliability Score R(x). | **None** — R(x) predicts *stability across architectures*, not legitimacy |
| 17 | The Attribution Impossibility | **Yes** — arXiv:**2605.21492** (audit said Apr 2026; actual **May 2026**), Caraker | Formally verified in Lean 4 (305 theorems, 16 axioms). Adds a **Rashomon Characterization theorem** with a Z-test: rankings unreliable when Z_jk < 1.96. 68% of 77 datasets show attribution instability. | N/A — impossibility result, not a reference |
| 18 | TTA / continual-learning explanation stability | **Confirmed open** | No paper measures attribution stability during TTA or CL. The literature splits into explanation-drift-over-epochs and TTA-stability-by-accuracy/entropy. | — |

## 2. Papers the audit missed (found by adversarial search)

| Paper | Why it surfaced | Threat | Action |
|---|---|---|---|
| **Rethinking Robustness: A New Approach to Evaluating Feature Attribution Methods** (arXiv:2512.06665, Dec 2025; Kiourti, Singh, Duraipandian, Zhou, Li) | Closest hit for "how much *should* it change". Critiques existing robustness metrics for ignoring the model's output difference; proposes a new definition of "similar inputs", a new robustness metric, and GAN-generated test cases. | **MEDIUM** — same critique, different object: **input perturbations only, single model**. No model updates, no shift typology, no mechanism reference. | Cite as the nearest neighbour of our critique. Delta: we change the *object* (model updates) and supply a *reference*, not a better ratio. |
| **Missingness Bias Calibration in Feature Attribution Explanations** (MCal, arXiv:2603.04831, Mar 2026) | Directly relevant to departure D4. Reframes missingness bias as an artifact of a model's *output space* on ablated, off-manifold inputs, and corrects it with a linear head on a frozen base model. | **LOW as prior art, HIGH as support** | Cite as independent evidence that models are uncontrolled on coalition inputs — precisely the premise of our D4 remark. Our measurement (coalition deviation *between two checkpoints*) is not in MCal. |
| **Comparing Explanations is Not Enough, Explain the Change** (arXiv:2602.02304, Feb 2026) | Delta-attribution applied to LLM pre/post fine-tuning token attributions. | **LOW–MEDIUM** | Cite; confirms the "explain the change" framing is active but still reference-free. |
| **Practical Attribution Guidance for Rashomon Sets** (arXiv:2407.18482); **Allocation Multiplicity** (arXiv:2503.16621) | Rashomon-set attribution background. | **LOW** | Cite as background for the ρ floor. |

## 3. Three findings that strengthen the plan

**(a) The matched-operator null is itself novel methodology.** A targeted search for matched-control
designs in attribution drift returned nothing: *"Nothing surfaced pairing shifted/fine-tuned models
against seed-matched controls to isolate the shift effect"*, and *"seed floor" is not an established
term*. Departure D2 is therefore not merely an engineering nicety — it is a methodological
contribution no prior work has made, and it is what makes the "is this just Rashomon?" objection
answerable.

**(b) The seed floor's magnitude is genuinely contested, which makes measuring it worthwhile.**
The literature holds both positions: Laberge et al. and DenseNet121 seed studies find LIME/SHAP
explanations vary substantially across seeds, while other work holds that seed variation produces
*"different weights but often similar underlying reasoning."* The size of ρ is an open empirical
question whose answer is method- and correlation-structure dependent. Our design measures it under
matched conditions across explainer classes, resolving the tension rather than assuming a side.

**(c) The Attribution Impossibility gives us a free, principled diagnostic.** Its Rashomon
Characterization theorem (rankings unreliable when Z_jk < 1.96) applies exactly to our redundant
feature block `R` (noisy copies of `C`). We can therefore *partition* probe features into
collinearity-unreliable and collinearity-reliable sets a priori, and show that unwarranted change
persists on the reliable partition. That pre-empts the strongest deflation of our result — "this is
just collinearity" — using the deflating paper's own test.

## 4. Constraint inherited from Mougan's theory

Shapley efficiency implies **prediction shift ⟹ explanation shift** (if two instances' predictions
differ, their explanation vectors must differ in at least one component); the converse does not
hold. Consequently, measuring explanation change *without* conditioning on prediction preservation
partly measures a mathematical necessity rather than a model property. This is an independent
argument for conditioning on `P_ε`, and the formal reason FASS's filtering step is right. We adopt
it and cite the implication.

## 5. Gate decision

All six of the audit's closest papers, plus the two it missed, have `legitimacy_reference = none`.
Every one measures *magnitude of change* and labels the residual by fiat. The audit's Phase-0
failure condition ("one of them already has a legitimacy reference → pivot to Positioning 2") is
**not** triggered.

**Proceed with Positioning 1: warranted vs. unwarranted explanation change under model updates,**
with the theory (Positioning 2) folded in as the method-class result rather than held as a fallback.

**Residual risks carried forward:**
1. The empirical gate (ratio ≥ 1.5 over the matched null) is still unmet and is the real risk.
2. The audit's claim of a `shap.monitor` feature request (§10, motivation 3) could not be verified —
   **it is dropped from the paper** rather than asserted.
3. FASS and the X-ray paper make our vision experiment strictly a sanity check; the paper must say
   so explicitly and never claim vision benchmark novelty.
