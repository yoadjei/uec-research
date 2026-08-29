"""Vision sanity check on CIFAR-10. Not a benchmark: FASS and the chest X-ray study own vision
attribution stability, and this experiment claims only that the tabular pattern reappears.

Design note. A corrupted image is not a clean image, so a corruption shift has no shared support
and the instance-level comparison would be undefined. The update is therefore *additive*, which is
also what deployment actually looks like: the model is fine-tuned on extra data that is either
clean (null) or corrupted (treatment), and the probe stays on the clean distribution where both
checkpoints are in-distribution by construction. Only the distribution of the added data differs.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.data.vision_data import corrupt, load_cifar, normalise, plant_shortcut, subset  # noqa: E402
from uec.explain.vision import VISION_EXPLAINERS  # noqa: E402
from uec.metrics.distances import d_cosine, d_l1, d_spearman, topk_with  # noqa: E402
from uec.metrics.normalise import l1_abs  # noqa: E402
from uec.metrics.uec import change, preserved_mask, summarise  # noqa: E402
from uec.models.resnet import FixedClassLogit, SmallResNet, predict_logits, softmax  # noqa: E402
from uec.paths import RESULTS  # noqa: E402
from uec.rng import pin_threads, seed_everything  # noqa: E402

PATCH = 6
DISTANCES = {"spearman": d_spearman, "topk1pct": None, "cosine": d_cosine, "l1": d_l1}


def fit(model, X, y, lr, epochs, batch, seed, freeze_stem=False):
    seed_everything(seed)
    if freeze_stem:
        for p in model.stem.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.as_tensor(normalise(X), dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    model.train()
    steps = 0
    for _ in range(epochs):
        order = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            opt.zero_grad(set_to_none=True)
            lossf(model(Xt[idx]), yt[idx]).backward()
            opt.step()
            steps += 1
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return steps


def accuracy(model, X, y, batch=256):
    return float((predict_logits(model, normalise(X), batch).argmax(1) == y).mean())


def patch_mass(A, shape=(3, 32, 32), patch=PATCH):
    """Fraction of attribution mass inside the planted-shortcut corner."""
    A = np.abs(A).reshape(len(A), *shape)
    return A[:, :, :patch, :patch].sum(axis=(1, 2, 3)) / (A.sum(axis=(1, 2, 3)) + 1e-12)


def run(a):
    Xtr, ytr = load_cifar(train=True)
    Xte, yte = load_cifar(train=False)
    rows = []

    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        Xs, ys = subset(Xtr, ytr, a.n_source, rng)
        Xa, ya = subset(Xtr, ytr, a.n_update, np.random.default_rng([seed, 1]))
        Xb, yb = subset(Xtr, ytr, a.n_update, np.random.default_rng([seed, 2]))
        probe_X, probe_y = subset(Xte, yte, a.n_probe, np.random.default_rng([seed, 3]))

        for family in a.families:
            t0 = time.time()
            if family == "corruption":
                src_train, add_null, add_treat = Xs, Xa, corrupt(Xb, rng=rng)
                probe = probe_X
            elif family == "shortcut":
                src_train = plant_shortcut(Xs, ys, rng=rng)
                add_null = plant_shortcut(Xa, ya, rng=rng)
                add_treat = Xb  # tint removed: the model should stop using it
                probe = plant_shortcut(probe_X, probe_y, rng=rng)
            else:
                src_train, add_null, add_treat, probe = Xs, Xa, Xb, probe_X

            seed_everything(seed)
            f0 = SmallResNet(width=a.width, blocks=tuple(a.blocks))
            fit(f0, src_train, ys, a.lr, a.epochs, a.batch, seed)

            def update(add_X, add_y, s):
                child = copy.deepcopy(f0)
                n = fit(child, add_X, add_y, a.update_lr, a.update_epochs, a.batch, s,
                        freeze_stem=a.freeze_stem)
                return child, n

            f_null, n_null = update(add_null, ya, seed + 1)
            f_treat, n_treat = update(add_treat, yb, seed + 1)
            seed_everything(seed + 5000)
            f_seed = SmallResNet(width=a.width, blocks=tuple(a.blocks))
            fit(f_seed, src_train, ys, a.lr, a.epochs, a.batch, seed + 5000)
            assert n_null == n_treat

            models = {"source": f0, "null": f_null, "treatment": f_treat, "seed": f_seed}
            Xp = normalise(probe)
            p0 = softmax(predict_logits(f0, Xp))
            target_class = p0.argmax(1)
            pt = softmax(predict_logits(f_treat, Xp))
            gap = np.abs(p0[np.arange(len(Xp)), target_class]
                         - pt[np.arange(len(Xp)), target_class])

            diag = {
                "acc_source": accuracy(f0, probe, probe_y),
                "acc_treat": accuracy(f_treat, probe, probe_y),
                "agree_treat": float((p0.argmax(1) == pt.argmax(1)).mean()),
                "n_steps": n_treat,
            }

            wrapped = {k: FixedClassLogit(m, target_class).eval() for k, m in models.items()}
            k_top = max(8, int(0.01 * Xp[0].size))
            dists = dict(DISTANCES, topk1pct=topk_with(k_top))

            for name, (fn, stochastic) in VISION_EXPLAINERS.items():
                if name not in a.explainers:
                    continue
                A = {k: fn(w, Xp) for k, w in wrapped.items()}
                A0b = fn(wrapped["source"], Xp, run=1) if stochastic else A["source"]

                pm = {k: patch_mass(v).mean() for k, v in A.items()} if family == "shortcut" else {}

                for dname, dist in dists.items():
                    raw = {k: change(A["source"], A[k], l1_abs, dist)
                           for k in ("treatment", "null", "seed")}
                    nu = change(A["source"], A0b, l1_abs, dist)
                    for eps in (0.02, 0.05, 0.10):
                        m = preserved_mask(np.zeros(len(gap)), gap, eps)
                        if m.sum() < 10:
                            continue
                        s = summarise(
                            raw["treatment"][m], np.zeros(int(m.sum())),
                            nu[m] if stochastic else np.zeros(0),
                            raw["null"][m], raw["seed"][m], n_probe=len(Xp),
                            seed=seed, family=family, explainer=name, distance=dname,
                            phi="abs", features="all", eps=eps,
                            patch_mass_source=pm.get("source", np.nan),
                            patch_mass_treat=pm.get("treatment", np.nan),
                            patch_mass_null=pm.get("null", np.nan),
                            **diag,
                        )
                        rows.append(s.as_dict())

            print(f"  s{seed} {family:11s} acc={diag['acc_source']:.3f}->{diag['acc_treat']:.3f} "
                  f"agree={diag['agree_treat']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--families", nargs="+", default=["corruption", "shortcut"])
    ap.add_argument("--explainers", nargs="+",
                    default=["integrated_gradients", "gradient_x_input", "grad_cam", "saliency"])
    ap.add_argument("--n-source", type=int, default=8000)
    ap.add_argument("--n-update", type=int, default=4000)
    ap.add_argument("--n-probe", type=int, default=200)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--blocks", type=int, nargs="+", default=[1, 1, 1])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--update-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--update-lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--freeze-stem", action="store_true")
    a = ap.parse_args()

    pin_threads(8)
    df = run(a)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "vision_metrics.parquet")
    print(f"\nwrote {len(df)} rows")


if __name__ == "__main__":
    main()
