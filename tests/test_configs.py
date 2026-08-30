"""The configs record what was actually run. If they drift from the runners they are worse than
useless, so every value is asserted against the runner's own argparse default or the explainer
function's signature.
"""

import importlib.util
import inspect
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _load_runner(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _defaults(module):
    """Re-run the runner's argparse setup without executing main()."""
    import argparse

    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def fake_parse(self, *a, **kw):
        captured.update({k: v.default for k, v in self._option_string_actions.items()
                         if k.startswith("--")})
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = fake_parse
    try:
        module.main()
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    return {k.lstrip("-").replace("-", "_"): v for k, v in captured.items()}


@pytest.mark.parametrize("name", ["synthetic", "folktables", "vision", "trees"])
def test_config_matches_runner_defaults(name):
    cfg = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    runner = ROOT / cfg.pop("runner")
    assert runner.exists(), runner
    defaults = _defaults(_load_runner(runner))

    mismatches = []
    for key, value in cfg.items():
        if key not in defaults:
            continue
        got = defaults[key]
        if isinstance(value, list) and isinstance(got, list):
            # name lists are sets; numeric lists (epoch grids, block counts) are ordered
            ok = (sorted(value) == sorted(got) if all(isinstance(v, str) for v in value)
                  else list(value) == list(got))
        elif isinstance(value, (int, float)) and isinstance(got, (int, float)):
            ok = float(value) == float(got)
        else:
            ok = str(value) == str(got)
        if not ok:
            mismatches.append(f"{key}: config={value!r} runner={got!r}")
    assert not mismatches, f"{name}.yaml drifted from {runner.name}: " + "; ".join(mismatches)


def test_explainer_config_matches_function_signatures():
    from uec.explain import gradient, perturbation

    cfg = yaml.safe_load((CONFIGS / "explainers.yaml").read_text())
    checks = [
        (gradient.integrated_gradients, "n_steps", cfg["integrated_gradients"]["n_steps"]),
        (gradient.expected_gradients, "n_samples", cfg["expected_gradients"]["n_samples"]),
        (gradient.smoothgrad, "sigma", cfg["smoothgrad"]["sigma"]),
        (gradient.smoothgrad, "n_samples", cfg["smoothgrad"]["n_samples"]),
        (perturbation.kernel_shap, "nsamples", cfg["kernel_shap"]["nsamples"]),
        (perturbation.lime_tabular, "num_samples", cfg["lime"]["num_samples"]),
        (perturbation.lime_tabular, "kernel_width", cfg["lime"]["kernel_width"]),
    ]
    for fn, param, expected in checks:
        got = inspect.signature(fn).parameters[param].default
        assert got == expected, f"{fn.__name__}.{param}: config={expected!r} code={got!r}"
