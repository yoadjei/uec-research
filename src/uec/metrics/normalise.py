import numpy as np


def _safe_div(a, s):
    return np.divide(a, s, out=np.zeros_like(a, dtype=float), where=s > 0)


def l1_abs(a: np.ndarray) -> np.ndarray:
    """Primary phi: attribution mass, invariant to positive rescaling."""
    a = np.abs(np.asarray(a, dtype=float))
    return _safe_div(a, a.sum(-1, keepdims=True))


def l1_signed(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return _safe_div(a, np.abs(a).sum(-1, keepdims=True))


NORMALISERS = {"abs": l1_abs, "signed": l1_signed}
