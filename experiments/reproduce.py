"""Run the whole CPU pipeline in dependency order and rebuild every table and figure.

    python experiments/reproduce.py            # everything (~4 h on 8 CPU cores)
    python experiments/reproduce.py --quick    # headline result only (~15 min)
    python experiments/reproduce.py --list     # show the plan and exit

Stages are skipped when their output already exists, so an interrupted run resumes. `--force`
reruns regardless. Nothing here needs a GPU; the transformer arm is `kaggle/scale_probe.py` and
its measured output is committed under results/ because it cannot be regenerated locally.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uec.paths import FIGURES, RESULTS  # noqa: E402

# (name, script, args, output that proves it ran, rough minutes, in --quick?)
STAGES = [
    ("synthetic", "run_synthetic.py", [], "synthetic_metrics.parquet", 15, True),
    ("regime sweep", "sweep_regime.py",
     ["--seeds", "5", "--magnitudes", "0.75", "1.0", "1.5", "2.0",
      "--update-epochs", "1", "2", "5", "20", "60",
      "--update-lrs", "2e-4", "5e-4", "2e-3"], "sweep_regime.parquet", 45, False),
    ("trees", "run_trees.py", [], "trees_metrics.parquet", 2, True),
    ("differentiation", "run_differentiation.py",
     ["--seeds", "5", "--update-epochs", "2", "20", "100", "400"],
     "differentiation.parquet", 25, False),
    ("faithfulness (E6)", "run_faithfulness.py",
     ["--seeds", "10", "--families", "covariate", "concept", "shortcut", "none"],
     "faithfulness.parquet", 8, False),
    ("extra ablations", "run_ablations.py", ["--seeds", "5"], "ablations_extra.parquet", 20, False),
    ("budget sweep", "run_budget_sweep.py", ["--seeds", "3"], "budget_sweep.parquet", 15, False),
    ("re-audit", "run_reaudit.py", [], "reaudit_verdicts.parquet", 30, False),
    ("width scaling", "scale_width", [], "scale_width.parquet", 5, False),
    ("ACS states", "run_folktables.py", [], "folktables_metrics.parquet", 60, False),
    ("ACS years", "run_folktables.py", ["--target-year", "2022"],
     "folktables_year_metrics.parquet", 25, False),
    ("vision", "run_vision.py", [], "vision_metrics.parquet", 40, False),
    ("semi-synthetic ACS", "run_semisynthetic.py", ["--seeds", "10"],
     "semisynthetic_metrics.parquet", 12, False),
    ("redundancy sweep", "run_redundancy.py", ["--seeds", "8"],
     "redundancy_sweep.parquet", 6, False),
]

BUILDERS = [("tables", "make_tables.py"), ("figures", "make_figures.py"),
            ("scale analysis", "analyse_scale.py")]


def run_width_stage():
    """The width sweep lives in the Kaggle script but needs no GPU, so it runs here."""
    sys.path.insert(0, str(REPO / "kaggle"))
    import scale_probe

    scale_probe.OUT = RESULTS
    scale_probe.run_width(seeds=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="headline result only")
    ap.add_argument("--force", action="store_true", help="rerun stages whose output exists")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    a = ap.parse_args()

    stages = [s for s in STAGES if s[5] or not a.quick]
    total = sum(s[4] for s in stages)

    if a.list:
        print(f"{'stage':22s} {'output':38s} {'min':>5}  status")
        for name, _, _, out, mins, _ in STAGES:
            have = "have" if (RESULTS / out).exists() else "-"
            mark = "quick" if _ else ""
            print(f"{name:22s} {out:38s} {mins:5d}  {have:5s} {mark}")
        print(f"\ntotal if nothing cached: ~{total} min "
              f"({'quick set' if a.quick else 'full set'})")
        return

    print(f"reproduce: {len(stages)} stage(s), ~{total} min if nothing is cached\n")
    RESULTS.mkdir(parents=True, exist_ok=True)
    failed = []

    for name, script, args, out, mins, _ in stages:
        if (RESULTS / out).exists() and not a.force:
            print(f"[skip] {name:22s} ({out} exists)")
            continue
        print(f"[run ] {name:22s} ~{mins} min", flush=True)
        t0 = time.time()
        if script == "scale_width":
            try:
                run_width_stage()
                rc = 0
            except Exception as e:  # noqa: BLE001 - report and continue to the next stage
                print(f"        {type(e).__name__}: {e}")
                rc = 1
        else:
            rc = subprocess.call([sys.executable, str(REPO / "experiments" / script), *args])
        ok = rc == 0 and (RESULTS / out).exists()
        print(f"        {'ok' if ok else 'FAILED'} in {(time.time() - t0) / 60:.1f} min")
        if not ok:
            failed.append(name)

    print()
    for name, script in BUILDERS:
        print(f"[run ] {name}", flush=True)
        subprocess.call([sys.executable, str(REPO / "experiments" / script)])

    print("\n" + "=" * 60)
    n_fig = len(list(FIGURES.glob("*.png")))
    n_tab = len(list((REPO / "paper" / "tables").glob("*.csv")))
    print(f"figures: {n_fig}   tables: {n_tab}")
    if failed:
        print(f"FAILED stages: {', '.join(failed)}")
        sys.exit(1)
    print("all stages completed")


if __name__ == "__main__":
    main()
