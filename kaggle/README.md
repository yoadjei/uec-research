# Scale probe — run on Kaggle (free H100)

Closes the one item the CPU work cannot reach: whether the effect survives at scale. Everything
else in the paper runs on a laptop; this does not.

## Run it

New Kaggle notebook → Settings → Accelerator: **GPU H100** (or P100/T4; slower but fine).
Paste into one cell:

```python
!git clone -q https://github.com/yoadjei/uec-research.git
!pip install -q transformers datasets
!python uec-research/kaggle/scale_probe.py --task all --seeds 3
```

Then download the three parquets from `/kaggle/working/`:

| file | what it answers |
|---|---|
| `scale_width.parquet` | does the ratio vanish as capacity grows? widths 32 → 1024 |
| `scale_vision.parquet` | a real ResNet-18, replacing the 78k-parameter CPU stand-in |
| `scale_text.parquet` | **the headline** — DistilBERT (66M params), token attributions |

Expected wall-clock on an H100: width ~10 min, vision ~25 min, text ~45 min. Well inside a
9-hour session. If time is short, `--task text` alone is the one that matters.

## What each arm does

All three use the same design as the CPU experiments, and the design is the point:

- **source** — train on data from the source distribution
- **matched null** — apply the *same* update operator with *more source data*
- **treatment** — apply the *same* update operator with *shifted data*

Only the distribution of the added data differs. Same learning rate, same epochs, same step count,
same sample size — the runner asserts the step counts match rather than trusting them.

Updates are **additive** (old data replayed alongside new). Training on the new data alone is domain
replacement, not an update: on CIFAR it drove accuracy from 0.59 to 0.44 and prediction agreement to
0.54, which destroys the prediction-preserved probe the whole comparison depends on.

Text shift is IMDB → Rotten Tomatoes: same task (binary sentiment), different corpus, so the label
semantics are preserved while the input distribution moves. The probe stays on held-out IMDB, where
both checkpoints are in-distribution by construction.

## Reading the output

`ratio` is `delta / rho_null` — shift-induced attribution change divided by what the same amount of
training produces on unshifted data. Above 1 means the shift moved explanations more than training
alone. On CPU we measure 1.4–2.1 for MLPs, trees and a small ResNet.

The question this run settles is whether that survives at 66M parameters, or whether it is an
artefact of small models. **A ratio near 1.0 for DistilBERT is a real and publishable answer** — it
would bound the claim rather than break the paper, and the limitations section is written to accept
it.
