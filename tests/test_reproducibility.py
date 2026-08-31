"""Reproducibility guards.

Two failure modes these catch, both of which happened during development:

1. A builder silently overwriting committed tables or figures with nothing when `results/` is
   empty, which is what a reader gets on a fresh clone.
2. A number in the paper drifting away from the data it came from.

The second check is skipped when the result files are absent, so a fresh clone still passes the
suite; it becomes active as soon as anything has been generated.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
TABLES = REPO / "paper" / "tables"
FIGURES = REPO / "figures"


def _run(cmd, timeout):
    """pytest replaces stdin, and on Windows subprocess cannot duplicate that handle
    (WinError 6). Handing it an explicit DEVNULL avoids it."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL, cwd=str(REPO))


def test_committed_deliverables_are_present():
    """A fresh clone must already contain the paper's artefacts."""
    assert len(list(FIGURES.glob("*.png"))) >= 13, "figures are part of the deliverable"
    assert len(list(TABLES.glob("*.csv"))) >= 15, "tables are part of the deliverable"
    assert (TABLES / "tables.md").stat().st_size > 5000, "tables.md looks truncated or emptied"


def test_every_declared_dependency_is_importable():
    import importlib.util
    import re
    import tomllib

    cfg = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    alias = {"scikit-learn": "sklearn", "pyyaml": "yaml"}
    missing = []
    for spec in cfg["project"]["dependencies"]:
        name = re.split(r"[><=!\[]", spec)[0].strip()
        mod = alias.get(name, name.replace("-", "_"))
        if importlib.util.find_spec(mod) is None:
            missing.append(name)
    assert not missing, f"declared but not importable: {missing}"


def test_reproduce_entry_point_lists_its_plan():
    r = _run([sys.executable, str(REPO / "experiments" / "reproduce.py"), "--list"], 300)
    assert r.returncode == 0, r.stderr
    for stage in ("synthetic", "trees", "re-audit", "vision", "ACS states"):
        assert stage in r.stdout, f"stage '{stage}' missing from the plan"


@pytest.mark.skipif(not (RESULTS / "synthetic_metrics.parquet").exists(),
                    reason="results absent; nothing to verify against")
def test_paper_numbers_match_the_data():
    r = _run([sys.executable, str(REPO / "experiments" / "verify_paper.py")], 900)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr
    assert r.returncode == 0, f"paper numbers drifted from the data:\n{r.stdout}\n{r.stderr}"
    assert "0 mismatched" in tail, tail
