"""Prove paper/supplementary.zip is anonymous, complete and runnable.

    python experiments/check_supplementary.py

Extracts the archive to a temporary directory outside the repository and runs the verification
script and the test suite there, with nothing installed. Building an archive is not evidence that
it works: the 2026-09-05 archive looked fine and could not import its own library, and a later
build passed an anonymity sweep while silently missing `src/uec/data`. Only a clean extract
catches that.

Exit code 1 if any check fails.
"""

import getpass
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ZIP = REPO / "paper" / "supplementary.zip"


def identity_needles() -> list[bytes]:
    out = []
    try:
        for args in (("config", "user.email"), ("config", "user.name")):
            v = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                               timeout=30).stdout.strip()
            if len(v) > 3:
                out.append(v.lower().encode())
    except Exception:
        pass
    user = getpass.getuser() or os.environ.get("USERNAME", "")
    if len(user) > 3:
        out.append(user.lower().encode())
    home = Path.home().name
    if len(home) > 3:
        out.append(home.lower().encode())
    return sorted(set(out))


def main() -> int:
    if not ZIP.exists():
        print(f"FAIL  {ZIP} does not exist; run experiments/build_supplementary.py")
        return 1

    failures = []
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        print(f"archive: {len(names)} entries, {ZIP.stat().st_size / 1e6:.1f} MB\n")

        caches = [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
        print(f"{'ok ' if not caches else 'FAIL'}  no byte caches" +
              (f" -- found {len(caches)}" if caches else ""))
        failures += caches[:1]

        needles = identity_needles()
        leaks = [n for n in names if not n.endswith("/")
                 and any(x in z.read(n).lower() for x in needles)]
        print(f"{'ok ' if not leaks else 'FAIL'}  no identifying strings" +
              (f" -- {leaks}" if leaks else f" (checked {len(needles)} patterns)"))
        failures += leaks[:1]

        for required in ("supplementary/src/uec/data/synthetic.py",
                         "supplementary/src/uec/paths.py",
                         "supplementary/tests/conftest.py",
                         "supplementary/paper/tables/T2_uec.csv",
                         "supplementary/README.md"):
            ok = required in names
            print(f"{'ok ' if ok else 'FAIL'}  {required}")
            if not ok:
                failures.append(required)

    tmp = Path(tempfile.mkdtemp(prefix="supp_check_"))
    try:
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(tmp)
        root = tmp / "supplementary"
        print(f"\nextracted to {root}, running with nothing installed\n")

        for label, cmd, want in [
            ("verify_paper", [sys.executable, "experiments/verify_paper.py"], "0 mismatched"),
            ("pytest", [sys.executable, "-m", "pytest", "-q", "--no-header", "tests"], "passed"),
        ]:
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=3600)
            tail = [ln for ln in (r.stdout + r.stderr).strip().splitlines() if ln.strip()]
            summary = tail[-1] if tail else "(no output)"
            ok = r.returncode == 0 and want in summary
            print(f"{'ok ' if ok else 'FAIL'}  {label}: {summary}")
            if not ok:
                failures.append(label)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} failure(s)")
        return 1
    print("supplementary is anonymous, complete and runnable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
