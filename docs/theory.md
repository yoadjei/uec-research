# Theory: which explainers inherit stability from predictions

Throughout, `f, f' : R^d → R` are the explained scalar outputs (logits) of two checkpoints,
`δ := f − f'`, the attribution baseline `b` is fixed and shared, `u := x − b`, and the straight path
is `γ(α) := b + αu` for `α ∈ [0,1]`. All explainers below are linear in the explained function, so
`E_f − E_{f'} = E_δ`; we use this repeatedly without comment.

Integrated Gradients:

    IG_j^f(x) = u_j ∫₀¹ ∂_j f(γ(α)) dα

The question is: if two checkpoints agree on their outputs, must their attributions agree?

---

## Proposition 1 (path-integrated attributions: the aggregate is inherited, the allocation is not)

**(i) Aggregate identity and bound.** For any `f, f'` differentiable on `γ`,

    Σ_j ΔIG_j(x) = δ(x) − δ(b)

hence if `|δ(x)| ≤ ε` and `|δ(b)| ≤ ε` then `|Σ_j ΔIG_j(x)| ≤ 2ε`.

*Proof.* By the fundamental theorem of calculus along `γ`,
`Σ_j IG_j^f(x) = ∫₀¹ ⟨∇f(γ(α)), u⟩ dα = ∫₀¹ (d/dα) f(γ(α)) dα = f(x) − f(b)` (completeness).
Applying this to `δ` and using linearity gives `Σ_j ΔIG_j(x) = δ(x) − δ(b)`. The bound is the
triangle inequality. ∎

Two things are worth noting. The identity requires output agreement **only at the two endpoints**
`x` and `b`, not along the whole path — a weaker premise than the audit assumed. And it is an
*equality*, not merely a bound, which makes it an exact empirical check on our implementation rather
than a loose inequality.

**(ii) Componentwise bound requires gradient control.**

    |ΔIG_j(x)| ≤ |u_j| · sup_{z∈γ} |∂_j δ(z)|,      ‖ΔIG(x)‖₁ ≤ ‖u‖₁ · sup_{z∈γ} ‖∇δ(z)‖_∞

*Proof.* `ΔIG_j(x) = u_j ∫₀¹ ∂_j δ(γ(α)) dα`; bound the integrand by its supremum on `γ` and sum. ∎

**(iii) Sharpness: output preservation alone does not bound individual attributions.**
For every `ε > 0` and every `M > 0` there exist `f, f'` with `sup_{z ∈ R^d} |f(z) − f'(z)| ≤ ε` and
`‖ΔIG(x)‖_∞ ≥ M`.

*Proof.* Take `d ≥ 2`, `b = 0`, `x = 1` (the all-ones vector), and

    δ(z) = ε · sin(ω(z₁ − z₂))

Then `|δ| ≤ ε` everywhere, so the premise holds globally, not merely on the path. On the path
`γ(α) = α·1` we have `γ₁ − γ₂ = 0`, so `∂₁δ(γ(α)) = εω cos(0) = εω` and `∂₂δ = −εω` for all `α`.
Therefore `ΔIG₁(x) = u₁ · εω = εω` and `ΔIG₂(x) = −εω`. Choosing `ω ≥ M/ε` gives the claim.
Consistently with (i), the aggregate is `εω − εω = 0 = δ(x) − δ(b)`. ∎

The construction is not pathological in kind: it is a difference that oscillates *transverse* to the
integration path while vanishing *on* it. Two checkpoints that agree on the data manifold but curve
differently off it — which is what fine-tuning produces — realise exactly this geometry. This is the
mechanism Dombrowski et al. (2019) identified for input-space fragility, transposed to model space.

---

## Proposition 2 (local-gradient attributions inherit nothing, not even the aggregate)

For every `ε > 0` and `M > 0` there exist `f, f'` with `sup_z |δ(z)| ≤ ε` and

    |Σ_j Δ(G×I)_j(x)| ≥ M,        ‖Δ∇f(x)‖_∞ ≥ M

where `(G×I)_j(x) = x_j ∂_j f(x)`.

*Proof.* Take `δ(z) = ε sin(ω z₁)` and any `x` with `x₁ ≠ 0` and `cos(ωx₁) = 1` (choose `ω = 2πk/x₁`).
Then `sup|δ| = ε`, while `∂₁δ(x) = εω` and `∂_jδ(x) = 0` for `j ≠ 1`. Hence
`Σ_j Δ(G×I)_j(x) = x₁ εω`, and `‖Δ∇f(x)‖_∞ = εω`. Take `ω ≥ M/(ε|x₁|)`. ∎

**This is the asymmetry.** Under bounded output change, the aggregate attribution mass of IG is
pinned to within `2ε` by Proposition 1(i); for gradient×input and saliency it is unbounded. The
distinction is not smoothness or dimension — it is *completeness*. Path integration averages the
gradient over `[0,1]`, and averaging annihilates precisely the high-frequency transverse components
that a local gradient reads off at a single point.

**Corollary (what is and is not inherited).** Let an update satisfy `|δ| ≤ ε` on the probe and at
the baseline. Then:

| Quantity | Path-integrated (IG, EG) | Local gradient (saliency, G×I) |
|---|---|---|
| aggregate attribution mass | **bounded by 2ε** (P1i, an equality) | **unbounded** (P2) |
| individual attributions | unbounded (P1iii) | unbounded (P2) |
| allocation across features (rank order) | **unbounded** | **unbounded** |

The third row is the practically important one and it is negative for both classes. Completeness
buys aggregate stability and nothing else. Since practitioners read the *allocation* — which feature
ranks first — no attribution method in either class carries a stability guarantee for the thing that
is actually consumed.

---

## Proposition 3 (Shapley-type attributions, under coalition-level preservation)

Let `v_f, v_{f'} : 2^{[d]} → R` be coalition value functions and `φ^f` the Shapley values. If
`|v_f(S) − v_{f'}(S)| ≤ ε_coal` for **every** `S ⊆ [d]`, then `‖φ^f − φ^{f'}‖_∞ ≤ 2ε_coal`.

*Proof.* With `w_{|S|} = |S|!(d−|S|−1)!/d!` and `Σ_{S ⊆ [d]\{j}} w_{|S|} = 1`,

    φ_j^f − φ_j^{f'} = Σ_{S ⊆ [d]\{j}} w_{|S|} [ (v_f(S∪{j}) − v_{f'}(S∪{j})) − (v_f(S) − v_{f'}(S)) ]

Each bracket is bounded in absolute value by `2ε_coal`, and the weights are a probability
distribution. ∎

### Remark (the premise is the whole problem)

`ε_coal` is a supremum over **masked** inputs. For KernelSHAP with a background distribution,
`v(S) = E_{z∼bg}[ f(x_S ; z_{−S}) ]` — composites that lie off the data manifold. Prediction
preservation on the data distribution, `|f(x) − f'(x)| ≤ ε` for `x ∼ P`, places **no** constraint on
`ε_coal`: fine-tuning optimises a loss over `P` and leaves the model unconstrained on ablated
inputs. MCal (arXiv:2603.04831) independently establishes that models behave anomalously on ablated,
out-of-distribution inputs, and reframes that anomaly as a property of the output space — exactly the
quantity `ε_coal` measures.

The prediction is therefore `ε_coal ≫ ε_data`, and it is measurable: for each probe point we record
the maximum coalition-value deviation over the coalitions KernelSHAP actually samples, alongside
`ε_data = |f_t(x) − f_{t+1}(x)|`. If the prediction holds, Proposition 3 explains observed SHAP
instability *from theory* rather than asserting it, and the practical consequence is sharp: a
practitioner cannot infer SHAP stability from prediction stability, because the two are measured on
different input sets.

---

## Testable predictions (H4)

Define per probe point, with the checkpoints actually trained:

    slack_IG(x)  = |Σ_j ΔIG_j(x)|      / (|δ(x)| + |δ(b)|)
    slack_GI(x)  = |Σ_j Δ(G×I)_j(x)|   / (|δ(x)| + |δ(b)|)
    coal_ratio(x) = ε_coal(x) / ε_data(x)

Predictions, in falsifiable form, with the measured outcome (10 seeds × 4 shift families ×
500 probe points = 20,000 points; `results/synthetic_theory.parquet`):

| # | Prediction | Outcome |
|---|---|---|
| 1 | `slack_IG(x) ≤ 1` for **every** probe point, up to quadrature error | **Held. 0 violations / 20,000 points.** Max slack 1.00023 against a propagated quadrature tolerance of 0.00034; the identity residual is 0 to 1e-8. The bound is also *tight* — median slack 0.87–0.99 — so it is not vacuous. |
| 2 | `slack_GI(x) > 1` for a substantial fraction | **Held. 60.8% of points**, and 60.5% restricted to prediction-preserved points; median 1.03–1.15, 90th percentile 1.47–1.62. |
| 3 | `coal_ratio(x) ≫ 1` | **Held in direction, weaker in size.** Median `ε_coal/ε_data` is 1.5–2.0, not orders of magnitude. The honest statement is that coalition-level deviation *systematically exceeds* data-level deviation, so Proposition 3's usable bound is roughly `3ε_data` rather than `2ε_data`, and it degrades with off-manifold divergence rather than being controlled by prediction preservation. |
| 4 | The *allocation* is unprotected for both classes | **Held**, and it is the practically important one — see §7 of the paper. |

A single clean violation of prediction 1 would have falsified either Proposition 1(i) or the
implementation; the check was written to be able to fail and did not. Prediction 3 was stated as
`≫` and came back as `≈1.7×`; that overstatement is corrected here rather than in the reader's
head. Prediction 2 is a claim about typicality, not about the theorem, which is an existence
statement — had gradients never exceeded the bound, Proposition 2 would still stand and only H4's
empirical half would fail.

---

## What we do not attempt

- Wasserstein bounds on population explanation drift in input space: too loose to constrain anything.
- A decomposition *theorem*. The warranted/unwarranted split is a **design** grounded in a
  generative reference, not a theorem, and the paper says so. Claiming otherwise is the fastest way
  to lose a theory-minded reviewer.
- Any claim that IG is "stable". It is not; only its total is pinned.

## Numeric verification

Every proposition above is checked in `tests/test_theory_numeric.py` against functions whose truth
is computable in closed form:

| Test | Checks |
|---|---|
| `test_ig_aggregate_identity` | P1(i) as an equality, to 1e-8 |
| `test_ig_components_unbounded` | P1(iii): component gap grows linearly in ω while `sup\|δ\| ≤ ε` |
| `test_gradient_aggregate_unbounded` | P2: aggregate G×I gap grows linearly in ω |
| `test_shapley_coalition_bound` | P3 on an exact 6-feature enumeration |

A proof that fails its numeric check is a wrong proof, not a flaky test.
