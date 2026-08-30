# Scale probe — run on Kaggle

Closes the one audit item a laptop cannot reach: whether the effect survives on a model three
orders of magnitude larger than the MLPs. Everything else in the paper runs on CPU.

**Accelerator:** `GPU T4 ×2` or `GPU P100` — either works, the script uses one device. Prefer
**T4** if offered: the script uses mixed precision, and T4 has fp16 tensor cores while P100 does
not (roughly 2× on the transformer arm). **TPU v5e-8 will not work** — this is CUDA/CPU PyTorch,
not XLA.

## Run it

```python
!git clone -q https://github.com/yoadjei/uec-research.git
!pip install -q captum transformers datasets
!python uec-research/kaggle/scale_probe.py --task text --seeds 3
```

Add `--task vision` as a second run if you have session time left. The script checks its
dependencies and exits immediately with the exact pip command if any are missing, so a typo costs
seconds rather than a download.

Watch the per-seed line `probe gap |p0-pt|: ... preserved@0.05=...`. That number is the health
check: it is the fraction of probe points on which the two checkpoints still agree closely enough
for the comparison to mean anything. Below 0.15 the script tells you what to change.

Then download from `/kaggle/working/`:

| file | what it answers | T4 estimate |
|---|---|---|
| `scale_text.parquet` | **the headline** — DistilBERT, 66M params, token attributions | ~45 min |
| `scale_vision.parquet` | a real ResNet-18, replacing the 78k-param CPU stand-in | ~25 min |

**If time is short, run `--task text` alone.** It is the one that answers the reviewer objection;
the vision arm only upgrades a sanity check we already have.

The width sweep (`--task width`) is *already done on CPU* and is in the paper — it turned out to
cost minutes, not GPU-hours. It is left in the script only so the arm is reproducible.

## Safe to interrupt

Results are written after **every seed**, and re-running skips seeds already on disk. A session that
dies at seed 2 keeps seeds 0 and 1, and re-launching resumes. If you only get one seed, send it —
one seed is still informative, it just cannot carry a confidence interval.

## What each arm does

All arms use the design that is the point of the paper:

- **source** — train on data from the source distribution
- **matched null** — apply the *same* update operator with *more source data*
- **treatment** — apply the *same* update operator with *shifted data*

Only the distribution of the added data differs. Same learning rate, same epochs, same sample size,
and the script **asserts the step counts match** rather than trusting them.

Updates are **additive**: old data is replayed alongside new. Training on the new data alone is
domain replacement, not an update — on CIFAR it drove accuracy from 0.59 to 0.44 and prediction
agreement to 0.54, destroying the prediction-preserved probe the comparison depends on.

The text shift is IMDB → Rotten Tomatoes: same task (binary sentiment), different corpus, so label
semantics are preserved while the input distribution moves. The probe stays on held-out IMDB, where
both checkpoints are in-distribution by construction. Attribution runs in fp32 even though training
uses fp16 — gradients through 16 integration steps lose too much precision in half precision to
compare two checkpoints.

## Reading the output

`ratio` is `delta / rho_null` — shift-induced attribution change divided by what the same amount of
training produces on unshifted data. Above 1 means the shift moved explanations more than training
alone would have.

For reference, on CPU we measure **1.4–2.1** for MLPs (up to 1.07M parameters), gradient-boosted
trees, ACS Income, and a small ResNet.

**A ratio near 1.0 for DistilBERT is a real and publishable answer.** It would bound the claim to
small models rather than break the paper, and the limitations section is already written to accept
it. Do not tune anything to avoid that outcome.

## If something breaks

- **`preserved@0.05` below 0.15** → the update moved predictions too far and the conditioned
  comparison rests on too few points. The script warns and tells you to re-run with
  `--text-update-lr 2e-6`. Rows at `eps=0.20` are still written either way.
- `CUDA out of memory` → lower `--seeds` is not the fix; edit `batch=16` down to 8 in `run_text`.
- `ModuleNotFoundError` → the script now names the exact pip command and exits before
  downloading anything. Run it and re-launch.
- `datasets` download errors → Kaggle needs internet enabled in notebook settings.
- No GPU detected → the script warns and continues on CPU; stop it, the text arm would take hours.
