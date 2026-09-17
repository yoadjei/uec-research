"""Build paper/supplementary.zip: anonymised, complete, and runnable from a clean extract.

    python experiments/build_supplementary.py

The archive submitted on 2026-09-05 was assembled by hand and carried three defects that only a
scripted build can keep out:

  1. It leaked the author's username through six `.pyc` caches, which embed the absolute build
     path, and two logs carrying it in library warnings. Under double-blind review that is a
     de-anonymising leak, and `paper/iclr2027/check_submission.py` cannot see inside a zip.
  2. It shipped no `src/`, so every experiment script failed at `from uec... import` and nothing in
     the archive ran.
  3. It placed the tables at the archive root, where `verify_paper.py` (which looks in
     `paper/tables`) could not find them. Because that script returns None for a missing folder and
     its callers guard with `if x is not None`, the table-backed claims were not run and not
     counted as skipped: it reported 66 of 81 while looking like a clean pass.

So this script refuses to write an archive that leaks an identity or is missing a module, and the
layout is chosen so the archive root is importable: `src/uec/paths.py` sets `ROOT = parents[2]`, so
with `src/` at the top level `results/`, `figures/` and `paper/tables/` all resolve.

Nothing identifying is hardcoded here. The strings to redact are derived at run time from the git
identity, the OS user and the home directory, so this file is safe to keep in a public repository.
"""

import getpass
import hashlib
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / ".supp_stage"
OUT = REPO / "paper" / "supplementary.zip"

# Matched against any path component.
SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".ipynb_checkpoints", "uec.egg-info"}
# Matched at the repo root ONLY. /data is the CIFAR download and /runs is scratch, but
# src/uec/data is the generator and must ship. Anchoring these is not optional: an earlier
# version matched "data" anywhere and silently dropped the generator.
SKIP_TOP = {"data", "runs", "tasks", "docs", "paper", "kaggle_output", ".supp_stage"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pyd"}

TREES = [
    ("src", "src"),
    ("experiments", "experiments"),
    ("tests", "tests"),
    ("kaggle", "kaggle"),
    ("configs", "configs"),
    ("results", "results"),
    ("figures", "figures"),
    ("checkpoints", "checkpoints"),
    ("paper/tables", "paper/tables"),
]
FILES = [("pyproject.toml", "pyproject.toml")]

README = """# Supplementary material

Code, artefacts and checkpoints for *The Control Decides the Answer: A Matched-Operator Null for
Explanation Change Under Model Updates*.

## Layout

    src/uec/          the library: generator, explainers, metrics, training harness, theory bounds
    experiments/      one script per stage; `reproduce.py` runs all fourteen in dependency order
    tests/            101 tests covering the generator, the operators and each proposition
    kaggle/           the width sweep and the GPU text arm
    configs/          hyperparameters for each dataset arm
    results/          every produced artefact (parquet, npz, csv, logs)
    figures/          each figure with its source data beside it
    checkpoints/      the headline checkpoints named in results/registry.csv, with a manifest
    paper/tables/     the numbered tables the paper cites

## Reproducing

    pip install -e .
    python experiments/verify_paper.py     # re-derives 81 headline numbers from the artefacts
    python -m pytest -q                    # 101 tests
    python experiments/reproduce.py --list # the fourteen stages and their runtimes

`verify_paper.py` and the test suite run against the committed artefacts and need no training.
`reproduce.py` regenerates them; the synthetic, tree, ACS and CIFAR arms are CPU-only, and the
transformer arm needs one GPU. Its measured output is committed, so everything else reproduces
without it.

## Notes

Absolute paths in the committed logs are replaced with `<HOME>` for anonymity; nothing else in them
is altered. `data/` is not shipped: CIFAR-10 and ACS Income download on first use.
"""


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def redaction_rules() -> list[tuple[str, str]]:
    """Derived, never hardcoded, so this file carries no identity of its own."""
    rules: list[tuple[str, str]] = []
    for email in {_git("config", "user.email"), *_git("log", "--format=%ae").split()}:
        if email:
            rules.append((email, "anonymous@example.com"))
    for name in {_git("config", "user.name"), *_git("log", "--format=%an").splitlines()}:
        if name and len(name) > 2:
            rules.append((name, "Anonymous Author"))

    home = Path.home()
    rules += [(str(home), "<HOME>"), (str(home).replace("\\", "/"), "<HOME>")]

    user = getpass.getuser() or os.environ.get("USERNAME", "")
    if user and len(user) > 2:
        rules.append((user, "anon"))

    # Longest first, so a home path is replaced before the bare username inside it.
    return sorted(set(rules), key=lambda r: -len(r[0]))


def redact(b: bytes, rules) -> tuple[bytes, int]:
    n = 0
    for a, c in rules:
        for enc in ("utf-8", "utf-16-le"):
            try:
                pa, pc = a.encode(enc), c.encode(enc)
            except UnicodeEncodeError:
                continue
            if pa in b:
                n += b.count(pa)
                b = b.replace(pa, pc)
        for variant in (a.lower(), a.upper(), a.capitalize()):
            v = variant.encode()
            if variant != a and v in b:
                n += b.count(v)
                b = b.replace(v, c.encode())
    return b, n


def want(rel: Path) -> bool:
    if any(p in SKIP_DIRS or p.endswith(".egg-info") for p in rel.parts):
        return False
    if rel.parts and rel.parts[0] in SKIP_TOP and rel.parts[0] != "paper":
        return False
    return rel.suffix.lower() not in SKIP_SUFFIXES


def main() -> None:
    rules = redaction_rules()
    print(f"redaction rules derived: {len(rules)}")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    copied = touched = 0
    for src_rel, dst_rel in TREES:
        src = REPO / src_rel
        if not src.exists():
            print(f"  MISSING  {src_rel}")
            continue
        for f in sorted(src.rglob("*")):
            if not f.is_file() or not want(f.relative_to(REPO)):
                continue
            dst = STAGE / dst_rel / f.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            data, n = redact(f.read_bytes(), rules)
            if n:
                touched += 1
                print(f"  redacted {n:5d}x  {dst_rel}/{f.relative_to(src)}")
            dst.write_bytes(data)
            copied += 1

    for src_rel, dst_rel in FILES:
        data, n = redact((REPO / src_rel).read_bytes(), rules)
        (STAGE / dst_rel).write_bytes(data)
        copied += 1
        touched += bool(n)

    (STAGE / "README.md").write_text(README, encoding="utf-8")
    copied += 1
    print(f"\nstaged {copied} files, redacted {touched}")

    needles = [a.lower().encode() for a, _ in rules if len(a) > 3]
    leaks = [str(f.relative_to(STAGE)) for f in STAGE.rglob("*")
             if f.is_file() and any(x in f.read_bytes().lower() for x in needles)]
    if leaks:
        shutil.rmtree(STAGE)
        raise SystemExit(f"ANONYMITY LEAK, refusing to build: {leaks}")
    print("anonymity sweep: clean")

    missing = [str(f.relative_to(REPO / "src")) for f in (REPO / "src").rglob("*.py")
               if "__pycache__" not in f.parts
               and not (STAGE / "src" / f.relative_to(REPO / "src")).exists()]
    missing += [r for r in ("experiments/verify_paper.py", "tests/conftest.py",
                            "kaggle/scale_probe.py", "paper/tables/T2_uec.csv",
                            "results/synthetic_metrics.parquet")
                if not (STAGE / r).exists()]
    if missing:
        shutil.rmtree(STAGE)
        raise SystemExit(f"INCOMPLETE, refusing to build. Missing: {missing}")
    print(f"completeness sweep: clean ({len(list((STAGE / 'src').rglob('*.py')))} modules)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file():
                z.write(f, Path("supplementary") / f.relative_to(STAGE))
    shutil.rmtree(STAGE)

    with zipfile.ZipFile(OUT) as z:
        n = len(z.namelist())
    print(f"\nwrote {OUT.relative_to(REPO)}\n  {n} entries, {OUT.stat().st_size / 1e6:.1f} MB, "
          f"md5 {hashlib.md5(OUT.read_bytes()).hexdigest()}")
    print("\nverify with:  python experiments/check_supplementary.py")


if __name__ == "__main__":
    main()
