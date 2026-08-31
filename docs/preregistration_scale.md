# Pre-registration: does the effect hold at transformer scale?

Written **31 Aug 2026, before running the update-strength sweep**, and after seeing a single
DistilBERT result that we cannot yet interpret. The point of writing it now is that the
interpretation is fixed before the data arrives; a paper that accuses others of missing a control
cannot itself choose its reading after the fact.

---

## 1. What has been measured

DistilBERT-base (66,955,010 parameters), IMDB → Rotten Tomatoes, additive update, 3 seeds,
250-example probe, normalised-ℓ₁ distance at ε = 0.05.

| explainer | ratio Δ/ρ_null [95% CI] | per-seed | Cliff's δ |
|---|---|---|---|
| Integrated Gradients | **0.985 [0.946, 1.038]** | 0.974, 0.947, 1.038 | −0.11 |
| Gradient × Input | 1.035 [1.015, 1.052] | 1.015, 1.052, 1.041 | +0.56 |

Diagnostics, all healthy: prediction agreement **0.908**, preserved fraction at ε = 0.05
**0.607** (≈121 of 200 probe points per seed), matched-null and treatment step counts asserted
equal.

**This is a clean null, not a failed run.** The interval is tight and the conditioning set is
large. Taken at face value it says the effect does not appear at 66M parameters.

## 2. Why it cannot yet be reported as a scale result

The run used a **heavy** update: the full 5,000-example source replayed alongside 2,000 new
examples at learning rate 1×10⁻⁵. Our own tabular sweep (600 runs, `results/sweep_regime.parquet`)
shows the ratio collapsing toward 1 as the update grows, *independently of whether anything
shifted*:

| update | tabular ratio (IG) | agreement |
|---|---|---|
| 1 epoch | 1.56 | 0.962 |
| 2 epochs | 1.39 | 0.956 |
| 5 epochs | 1.27 | 0.943 |
| 20 epochs | 1.13 | 0.919 |
| 60 epochs | 1.05 | 0.886 |
| **DistilBERT, heavy** | **0.99** | **0.908** |

At agreement 0.908 our *20-parameter-per-layer MLPs* already read ≈ 1.05–1.13. The DistilBERT
number is therefore exactly what the tabular curve predicts for an update of that size, and the
observation is consistent with **both** of the following:

- **(A) Scale hypothesis** — the phenomenon does not occur in a 66M-parameter transformer.
- **(B) Update-size hypothesis** — the update was large enough to swamp the shift signal, as it
  does in every model class we have measured.

Reporting (A) without excluding (B) would be the same error the paper attributes to prior work:
drawing a conclusion from a measurement whose control was not matched.

## 3. The disambiguating experiment

`scale_probe.py --task text-sweep`: update learning rate ∈ {1×10⁻⁶, 3×10⁻⁶, 1×10⁻⁵}, 3 seeds,
source model trained once per seed and reused. Everything else is held fixed.

**The comparison is made at matched prediction agreement, not matched learning rate.** Agreement is
the operative measure of how far an update moved a model, and it is the only quantity comparable
across an MLP and a transformer. The tabular curve is read at the same agreement level as each
DistilBERT point.

## 4. Decision rule, fixed in advance

Let `r*` be the DistilBERT ratio for Integrated Gradients at the lightest update whose preserved
fraction is at least 0.15, and let `r_tab` be the tabular ratio at the *same* prediction agreement.

| Outcome | Condition | What we record |
|---|---|---|
| **S1 — effect holds at scale** | `r*` ≥ 1.25 and its CI excludes 1 | The phenomenon transfers. The flat first result was an artefact of update size, and that itself becomes evidence for the paper's central claim that update strength, not shift, drives most attribution movement. |
| **S2 — effect does not transfer** | `r*` ≤ 1.10 across *all three* update strengths, each with preserved fraction ≥ 0.15 | The phenomenon is bounded to smaller models. §9 gains a limitation stating the claim holds for MLPs, trees and small CNNs up to ~1M parameters and **fails** at 66M, and the abstract is amended. |
| **S3 — inconclusive** | anything else, including every update strength having preserved fraction < 0.15 | We report the curve, claim neither, and say the experiment did not resolve it. |
| **S4 — attenuated but present** | `1.10 < r*` < 1.25 with CI excluding 1, *and* `r* < r_tab` at matched agreement | The effect exists at scale but is weaker than in small models. Reported as attenuation, with both numbers given. |

Three seeds cannot support a strong claim in any branch. Whatever we record will say so.

## 5. Commitments

1. **No outcome will be attributed to a bug without evidence.** If S2 holds, we do not go looking
   for a reason to discard it.
2. **No tuning toward S1.** The sweep is over update strength only. We will not adjust the shift
   pair, probe, attribution method or ε to move the result.
3. **S2 is reported in the abstract**, not only in the limitations. A scale bound discovered by our
   own control is a finding, and burying it would be the same failure the paper documents in
   others.
4. **The already-measured heavy-update point is reported in every branch**, because it is what a
   practitioner using a standard fine-tuning recipe would actually see.

## 6. Why this document exists

The paper's contribution is that a missing control changes conclusions. That argument only holds if
we apply the same discipline to ourselves — including when the honest answer is that our headline
effect disappears on the largest model we tested.
