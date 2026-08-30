"""Run the text arm if needed, then print a compact summary that can be pasted back verbatim.

There is no file transfer between the Kaggle session and the analysis here, so the results have to
survive as text. This prints exactly the columns `experiments/analyse_scale.py` consumes, plus the
diagnostics needed to judge whether the run is trustworthy.

    !python uec-research/kaggle/run_and_report.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO / "results"
TEXT = OUT / "scale_text.parquet"


def code_version():
    try:
        return subprocess.run(["git", "-C", str(REPO), "log", "--oneline", "-1"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def report(df, name):
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    print(f"seeds: {sorted(df.seed.unique().tolist())}   rows: {len(df)}")
    for col in ("agree_treat", "agree_null"):
        if col in df:
            print(f"{col}: {df[col].mean():.4f}")

    cols = ["explainer", "distance", "eps", "delta", "rho_null", "ratio",
            "n_preserved", "preserved_frac"]
    cols = [c for c in cols if c in df]
    g = df.groupby(["explainer", "distance", "eps"], as_index=False)[
        [c for c in cols if c not in ("explainer", "distance", "eps")]
    ].mean()
    print("\n--- mean over seeds ---")
    print(g.round(4).to_string(index=False))

    print("\n--- per seed, l1, primary eps ---")
    q = df[df.distance == "l1"]
    if len(q):
        eps = 0.05 if (q.eps == 0.05).any() else float(q.eps.min())
        q = q[q.eps == eps]
        print(f"(eps = {eps})")
        print(q[["seed", "explainer", "delta", "rho_null", "ratio", "n_preserved"]]
              .sort_values(["explainer", "seed"]).round(4).to_string(index=False))


def main():
    print(f"code: {code_version()}")
    if TEXT.exists():
        done = pd.read_parquet(TEXT).seed.nunique()
        print(f"found existing scale_text.parquet with {done} seed(s)")
    else:
        print("no scale_text.parquet yet")

    need = not TEXT.exists() or pd.read_parquet(TEXT).seed.nunique() < 3
    if need:
        print("\nrunning the text arm (resumes any seeds already on disk)...\n", flush=True)
        rc = subprocess.call([sys.executable, str(REPO / "kaggle" / "scale_probe.py"),
                              "--task", "text", "--seeds", "3"])
        if rc != 0:
            print(f"\nrun failed with exit code {rc}")
            return

    if not TEXT.exists():
        print("\nno results were written -- paste the error above")
        return

    report(pd.read_parquet(TEXT), "DistilBERT  IMDB -> Rotten Tomatoes")
    print("\n" + "=" * 72)
    print("Paste everything from the first ==== line down.")


if __name__ == "__main__":
    main()
