from dataclasses import dataclass
from typing import Callable

from . import gradient as g
from . import perturbation as p


@dataclass(frozen=True)
class Explainer:
    name: str
    fn: Callable
    stochastic: bool
    needs_background: bool = False
    family: str = "gradient"  # gradient | path | surrogate | shapley

    def __call__(self, model, X, **kw):
        if not self.needs_background:
            kw.pop("background", None)
        return self.fn(model, X, **kw)


EXPLAINERS = {
    e.name: e
    for e in [
        Explainer("saliency", g.saliency, False, family="gradient"),
        Explainer("gradient_x_input", g.gradient_x_input, False, family="gradient"),
        Explainer("integrated_gradients", g.integrated_gradients, False, family="path"),
        Explainer("expected_gradients", g.expected_gradients, True, True, family="path"),
        Explainer("smoothgrad", g.smoothgrad, True, family="gradient"),
        Explainer("kernel_shap", p.kernel_shap, True, True, family="shapley"),
        Explainer("lime", p.lime_tabular, True, True, family="surrogate"),
    ]
}

# Gradient explainers run on the full probe; KernelSHAP and LIME are quadratic in cost and run on
# a prefix of it. The prefix is nested so every explainer is compared on a common subset.
CHEAP = ["saliency", "gradient_x_input", "integrated_gradients", "expected_gradients", "smoothgrad"]
EXPENSIVE = ["kernel_shap", "lime"]
MVE = ["integrated_gradients", "gradient_x_input", "kernel_shap"]
