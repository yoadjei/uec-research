"""Static checks on the Kaggle script.

It cannot be exercised here -- it needs a GPU, `transformers` and a live session -- so a broken
edit is only discovered by the person running it, minutes into a job. Twice now an edit removed a
module-level definition that a function still referenced, and both times a plain `ast.parse` said
the file was fine. These tests resolve every name the module actually uses.
"""

import ast
import builtins
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "kaggle" / "scale_probe.py"
REPORTER = Path(__file__).resolve().parents[1] / "kaggle" / "run_and_report.py"


def _module_names(tree):
    """Every name bound at module level: imports, assignments, defs, classes."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _local_names(fn):
    """Names bound inside a function: args, assignments, imports, comprehensions, nested defs."""
    names = set()
    for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs:
        names.add(a.arg)
    for a in (fn.args.vararg, fn.args.kwarg):
        if a:
            names.add(a.arg)
    for node in ast.walk(fn):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            for a in getattr(node, "args", ast.arguments(
                    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])).args:
                names.add(a.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            names.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


@pytest.mark.parametrize("path", [SCRIPT, REPORTER], ids=lambda p: p.name)
def test_every_referenced_name_resolves(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module = _module_names(tree)
    allowed = module | set(dir(builtins)) | {"__file__", "__name__"}

    # a nested function may legitimately read names bound in any enclosing function, so the
    # visible scope for a node is the union of its own bindings and those of its ancestors
    unresolved = {}

    def walk(fn, enclosing):
        scope = enclosing | _local_names(fn)
        inner = {n for n in ast.walk(fn)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn}
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in allowed and node.id not in scope:
                    unresolved.setdefault(fn.name, set()).add(node.id)
        for child in inner:
            walk(child, scope)

    for fn in tree.body:
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            walk(fn, set())

    assert not unresolved, f"{path.name} references undefined names: {unresolved}"


def test_declared_tasks_all_have_a_runner_and_dependencies():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    module = _module_names(tree)
    src = SCRIPT.read_text(encoding="utf-8")

    for task in ("width", "vision", "text"):
        assert f"run_{task}" in module, f"no runner for task '{task}'"
        assert f'"{task}"' in src, f"task '{task}' not wired into the CLI"
    assert "text-sweep" in src, "the disambiguating sweep must stay reachable from the CLI"
    assert "REQUIRES" in module and "check_deps" in module, "dependency preflight was removed"
