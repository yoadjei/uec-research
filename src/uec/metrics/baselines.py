"""Prior-work metrics, reimplemented so the comparison table is computed rather than asserted.

Each is applied to the same checkpoints, probe and attributions as UEC. The point of the exercise
is the shortcut case: there the model *should* change its explanation, and these metrics have no
way to say so.
"""

import numpy as np

from .normalise import l1_abs


def relative_output_stability(A_t, A_u, logit_t, logit_u, eps: float = 1e-12):
    """ROS (Agarwal et al. 2022), transposed from an input neighbourhood to a model update:
    ||dE|| / ||d f||. Undefined as the denominator vanishes, which is exactly the
    prediction-preserving regime, so we report the divergence rather than hiding it."""
    num = np.linalg.norm(l1_abs(A_t) - l1_abs(A_u), axis=-1)
    den = np.abs(np.asarray(logit_t) - np.asarray(logit_u))
    return num / np.maximum(den, eps), den


def fass_filtered_distance(delta, mask):
    """FASS: filter to prediction-preserved pairs, then report the raw distance. No floor, no
    reference -- whatever remains is called instability."""
    return float(delta[mask].mean()) if mask.any() else np.nan


def delta_audit_coupling(A_t, A_u, p_t, p_u):
    """Delta-Audit's behaviour-attribution coupling and its 'spurious' residual.

    BAC correlates attribution movement with prediction movement across the probe; the fraction of
    movement not explained by behaviour change is what Delta-Audit flags as risky redistribution.
    """
    move = np.abs(l1_abs(A_t) - l1_abs(A_u)).sum(-1)
    behave = np.abs(np.asarray(p_t) - np.asarray(p_u))
    if move.std() < 1e-12 or behave.std() < 1e-12:
        return np.nan, float(move.mean())
    bac = float(np.corrcoef(move, behave)[0, 1])
    resid = move - np.polyval(np.polyfit(behave, move, 1), behave)
    return bac, float(np.abs(resid).mean())


def jensen_shannon(A_t, A_u, eps: float = 1e-12):
    """Delta-Audit's redistribution term: JSD between normalised attribution masses."""
    p, q = l1_abs(A_t) + eps, l1_abs(A_u) + eps
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * np.log(a / b)).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare_all(A_t, A_u, logit_t, logit_u, p_t, p_u, delta, omega, mask, floor):
    """One row of the differentiation table."""
    ros, den = relative_output_stability(A_t, A_u, logit_t, logit_u)
    bac, spurious = delta_audit_coupling(A_t[mask], A_u[mask], p_t[mask], p_u[mask])
    return {
        "ros_mean": float(np.mean(ros[mask])) if mask.any() else np.nan,
        "ros_max": float(np.max(ros[mask])) if mask.any() else np.nan,
        "ros_denom_min": float(np.min(den[mask])) if mask.any() else np.nan,
        "fass_distance": fass_filtered_distance(delta, mask),
        "delta_audit_jsd": float(jensen_shannon(A_t[mask], A_u[mask]).mean()) if mask.any() else np.nan,
        "delta_audit_bac": bac,
        "delta_audit_spurious": spurious,
        "omega": float(omega[mask].mean()) if mask.any() else np.nan,
        "uec": float(delta[mask].mean() - omega[mask].mean() - floor) if mask.any() else np.nan,
    }
