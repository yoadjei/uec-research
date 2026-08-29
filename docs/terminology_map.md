# Terminology Map

The phrase "explanation drift" is polysemous as of Aug 2026. Three communities use it, or a
near-neighbour, for three different objects. Using it in a title invites a reviewer to assume we are
redoing whichever one they know. This file fixes our vocabulary and records what we may not claim.

## Who owns what

| Term | Owner | Object | Direction |
|---|---|---|---|
| **Explanation shift** | Mougan et al., TMLR 2025 (`skshift` package) | Divergence between `P(E_f(X_source))` and `P(E_f(X_target))` for a **frozen** model. Population-level. | Data changes → explanation *distribution* changes |
| **Explanation drift** | Dhayalkar (RSP), arXiv:2601.11625 | Epoch-to-epoch change in normalised attributions on a fixed probe set, **within one training run**. | Training time → attributions move |
| **Semantic drift** | Elangovan et al., arXiv:2604.08513 | Change in CAM attribution structure between transfer-learned and fully fine-tuned checkpoints. | Fine-tuning → evidence moves |
| **Drift explanation** | Hinder, Vaquet & Hammer | Explaining **why the data drifted**. | Opposite direction entirely |
| **Δ-Attribution** | Hemmat & Fatemi, arXiv:2508.19589 | `φ_B(x) − φ_A(x)` between two model versions. | Model A/B → attribution difference |
| **Mechanistic multiplicity** | Bensmail (EvoXplain) | Explanation variation across retraining runs. | Seeds → attributions differ |
| **Explanation Lottery** | Thackshanaramana B, arXiv:2603.15821 | Attribution disagreement across hypothesis classes at fixed predictions. | Model class → attributions differ |

## Our vocabulary

We do **not** use "explanation drift", "explanation shift", or "semantic drift" for our object.

| Our term | Definition | Notes |
|---|---|---|
| **Explanation change** `Δ_E(x)` | `d(φ(E_{f_t}(x)), φ(E_{f_{t+1}}(x)))` at a fixed probe input across one model update | Neutral word; makes no claim about legitimacy |
| **Warranted change** | Change consistent with the change in the environment's Bayes-optimal predictor `h_e` | The reference `ω` |
| **Unwarranted change** | Change on the shared support exceeding `ω` and both floors | The phenomenon |
| **Explainer noise floor** `ν_E` | Change between two runs of a stochastic explainer on the *same* checkpoint | Zero for deterministic explainers |
| **Matched-operator null floor** `ρ_null` | Change induced by the *same update operator* applied to fresh **source** data | Our primary control (departure D2) |
| **Seed floor** `ρ_seed` | Change between independent from-scratch retrains on source | The EvoXplain/Rashomon baseline, reported for comparability |
| **Unwarranted Explanation Change** `UEC` | `mean Δ − ω − max(ν, ρ_null)` | A decomposition, **not** a new stability metric |
| **Drift** | Reserved for a *temporal sequence* `{Δ_E^{(t)}}` over checkpoints | We do not study this; RSP owns it and we cite them for it |

## Claims we may not make

1. "We propose a new explanation stability metric." — Every distance we use already exists
   (Alvarez-Melis & Jaakkola 2018; Yeh et al. 2019; Agarwal et al. 2022; Quantus). The contribution
   is the *reference* and the *decomposition*.
2. "Explanations are unreliable under distribution shift." — "Reliable" is a human-trust construct
   we do not measure. We speak only of warranted/unwarranted change and stability.
3. "Our score detects distribution shift better than X." — That is Mougan's contribution, and their
   frozen-model setting is not ours.
4. "First to condition on prediction preservation." — FASS did it first, in vision, for input
   perturbations. We adopt their conditioning and say so.
5. "First to measure attribution change across a model update." — Delta-Audit and the X-ray paper
   did. We add the reference, the floors, and the theory.
6. Anything about a `shap.monitor` deployment-monitoring feature request — unverified, dropped.

## Title consequence

The audit's ten candidate titles all avoid "Explanation Drift" correctly. Our working title is
**"Stable Predictions, Shifting Evidence: Warranted and Unwarranted Explanation Change Under Model
Updates"**, with the subtitle carrying the two words that separate us from every row above:
*warranted* and *reference*.
