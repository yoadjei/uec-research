"""Scale probe for Kaggle's free GPUs (T4 x2 or P100). Closes the one audit item a CPU cannot reach.

Three experiments, all reusing the matched-operator design from the CPU work:

  width   Synthetic MLPs at widths 32 -> 1024. Does the ratio vanish with capacity? The CPU
          ablation stopped at 128 and showed a mild decline, which is the one piece of evidence
          that cuts against the scale argument, so this is the cheapest way to settle it.
  vision  A real ResNet-18 on CIFAR-10, replacing the 78k-parameter stand-in the CPU budget forced.
  text    DistilBERT (66M params) fine-tuned on IMDB, updated additively with either more IMDB
          (matched null) or Rotten Tomatoes (treatment), with token attributions. This is the
          headline scale result -- three orders of magnitude above the MLPs.

Every arm uses an *additive* update: the model is fine-tuned on its old data plus new data that is
either same-distribution (null) or shifted (treatment), and the step counts are asserted equal
rather than assumed. Training on the new data alone is domain replacement, not an update, and it
destroys the prediction-preserved probe the comparison needs.

Sized for a T4. Results are written after every seed, so a session that dies mid-run keeps what it
had, and re-running skips seeds already on disk.

    !git clone -q https://github.com/yoadjei/uec-research.git
    !pip install -q captum transformers datasets
    !python uec-research/kaggle/scale_probe.py --task text
"""

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from uec.metrics.distances import d_l1, d_spearman  # noqa: E402
from uec.metrics.normalise import l1_abs  # noqa: E402
from uec.metrics.uec import preserved_mask, summarise  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO / "results"


def amp_ctx():
    return torch.autocast("cuda", dtype=torch.float16) if AMP else contextlib.nullcontext()


def save_incremental(rows, name):
    """Write after every seed. A Kaggle session that dies at seed 2 should keep seeds 0 and 1."""
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / name)
    return df


def done_seeds(name):
    p = OUT / name
    if not p.exists():
        return set()
    try:
        return set(pd.read_parquet(p).seed.unique().tolist())
    except Exception:
        return set()


def row_change(A, B, distance=d_l1):
    """Distance between two ragged attribution lists, per example."""
    return np.array([
        float(distance(l1_abs(a[None, :]), l1_abs(b[None, :]))[0]) for a, b in zip(A, B)
    ])


def _report(df, title):
    print(f"\n=== {title} ===")
    if df.empty or "distance" not in df:
        print("no rows: every probe point failed the prediction-preservation filter. "
              "Increase --seeds or n_probe.")
        return
    q = df[(df.distance == "l1") & (df.eps == 0.05)]
    if q.empty:
        print("no rows at eps=0.05; showing all eps")
        q = df[df.distance == "l1"]
    cols = [c for c in ("rho_null", "delta", "ratio", "preserved_frac", "agree_treat") if c in q]
    print(q.groupby("explainer")[cols].mean().round(4).to_string())


# ---------------------------------------------------------------------------- width

def run_width(seeds=5, widths=(32, 64, 128, 256, 512, 1024), n_probe=300, n_steps=32):
    """Attributions stay on the validated CPU float64 path; only training moves to the GPU.
    The IG step count is reduced from 64 to 32 for this arm -- both checkpoints use the same
    quadrature, so the ratio is unaffected, and completeness still holds to ~4e-4."""
    from uec.data.support import shared_support_probe
    from uec.data.synthetic import make_pair
    from uec.explain.cache import attribute
    from uec.metrics.uec import change
    from uec.models.mlp import probabilities
    from uec.train.harness import TrainConfig, UpdateConfig, build_checkpoints

    name = "scale_width.parquet"
    rows = []
    seen = done_seeds(name)
    if seen:
        rows = pd.read_parquet(OUT / name).to_dict("records")
        print(f"  resuming, seeds already done: {sorted(seen)}", flush=True)

    for seed in range(seeds):
        if seed in seen:
            continue
        src, tgt = make_pair("covariate", magnitude=1.5)
        probe = shared_support_probe(src, tgt, n_probe, np.random.default_rng(500 + seed))
        for w in widths:
            t0 = time.time()
            ck = build_checkpoints(
                src, tgt, seed, 8000, 4000,
                TrainConfig(epochs=60, hidden=(w, w), device=DEV),
                UpdateConfig(lr=2e-4, epochs=2, device=DEV),
                regimes=("matched_null", "treatment"),
            )
            f0, f1, fn = ck["source"][0], ck["treatment"][0], ck["matched_null"][0]
            m = preserved_mask(probabilities(f0, probe), probabilities(f1, probe), 0.05)
            if m.sum() < 15:
                continue
            for ex in ("integrated_gradients", "gradient_x_input"):
                kw = {"n_steps": n_steps} if ex == "integrated_gradients" else {}
                A0 = attribute(f0, probe, ex, use_cache=False, **kw)
                delta = change(A0, attribute(f1, probe, ex, use_cache=False, **kw), l1_abs, d_l1)[m]
                rho = change(A0, attribute(fn, probe, ex, use_cache=False, **kw), l1_abs, d_l1)[m]
                rows.append({
                    "seed": seed, "width": w, "explainer": ex,
                    "n_params": 20 * w + w * w + w,
                    "delta": float(delta.mean()), "rho_null": float(rho.mean()),
                    "ratio": float(delta.mean() / rho.mean()),
                    "n_preserved": int(m.sum()), "seconds": time.time() - t0,
                })
            print(f"  width {w:5d} seed {seed} ratio={rows[-1]['ratio']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        save_incremental(rows, name)

    df = pd.DataFrame(rows)
    print("\n" + df.groupby(["width", "explainer"])[["delta", "rho_null", "ratio"]]
          .mean().round(4).to_string())
    return df


# --------------------------------------------------------------------------- vision

def run_vision(seeds=3, n_source=20000, n_add=10000, n_probe=400, epochs=8,
               lr=1e-3, update_lr=1e-4, batch=128, ig_steps=16):
    import torchvision
    from captum.attr import IntegratedGradients, Saliency
    from torchvision import transforms

    name = "scale_vision.parquet"
    seen = done_seeds(name)
    rows = pd.read_parquet(OUT / name).to_dict("records") if seen else []
    if seen:
        print(f"  resuming, seeds already done: {sorted(seen)}", flush=True)

    root = str(OUT / "cifar")
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True,
                                      transform=transforms.ToTensor())
    te = torchvision.datasets.CIFAR10(root, train=False, download=True,
                                      transform=transforms.ToTensor())
    Xtr = (tr.data.astype(np.float32) / 255).transpose(0, 3, 1, 2)
    ytr = np.array(tr.targets)
    Xte = (te.data.astype(np.float32) / 255).transpose(0, 3, 1, 2)
    yte = np.array(te.targets)
    mean = Xtr.mean((0, 2, 3), keepdims=True)
    std = Xtr.std((0, 2, 3), keepdims=True)

    def norm(X):
        return (X - mean) / std

    def corrupt(X, rng):
        out = np.clip(X + rng.normal(0, 0.08, X.shape).astype(np.float32), 0, 1)
        m = out.mean((2, 3), keepdims=True)
        return np.clip((out - m) * 0.5 + m, 0, 1)

    def resnet18():
        m = torchvision.models.resnet18(weights=None, num_classes=10)
        m.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # CIFAR stem
        m.maxpool = torch.nn.Identity()
        return m.to(DEV)

    def fit(model, X, y, n_epochs, learning_rate, seed):
        torch.manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        lossf = torch.nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda") if AMP else None
        Xt = torch.tensor(norm(X))
        yt = torch.tensor(y)
        g = torch.Generator().manual_seed(seed)
        model.train()
        steps = 0
        for _ in range(n_epochs):
            order = torch.randperm(len(yt), generator=g)
            for i in range(0, len(order), batch):
                b = order[i:i + batch]
                xb, yb = Xt[b].to(DEV, non_blocking=True), yt[b].to(DEV, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with amp_ctx():
                    loss = lossf(model(xb), yb)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                steps += 1
        model.eval()
        return steps

    @torch.no_grad()
    def logits(model, X):
        out = []
        Xt = torch.tensor(norm(X))
        for i in range(0, len(Xt), 256):
            out.append(model(Xt[i:i + 256].to(DEV)).float().cpu().numpy())
        return np.concatenate(out)

    for seed in range(seeds):
        todo = [x for x in lrs if (seed, float(x)) not in done]
        if not todo:
            continue
        t0 = time.time()
        rng = np.random.default_rng(seed)
        i_src = rng.choice(len(Xtr), n_source, replace=False)
        i_a = np.random.default_rng([seed, 1]).choice(len(Xtr), n_add, replace=False)
        i_b = np.random.default_rng([seed, 2]).choice(len(Xtr), n_add, replace=False)
        i_p = np.random.default_rng([seed, 3]).choice(len(Xte), n_probe, replace=False)
        Xs, ys = Xtr[i_src], ytr[i_src]
        probe = Xte[i_p]

        torch.manual_seed(seed)
        f0 = resnet18()
        fit(f0, Xs, ys, epochs, lr, seed)
        sd0 = {k: v.detach().clone() for k, v in f0.state_dict().items()}

        def updated(add_X, add_y, s):
            m = resnet18()
            m.load_state_dict(sd0)
            n = fit(m, np.concatenate([Xs, add_X]), np.concatenate([ys, add_y]), 1, update_lr, s)
            return m, n

        f_null, n_null = updated(Xtr[i_a], ytr[i_a], seed + 1)
        f_treat, n_treat = updated(corrupt(Xtr[i_b], rng), ytr[i_b], seed + 1)
        assert n_null == n_treat, "matched null must take the same number of steps"

        z0, zt = logits(f0, probe), logits(f_treat, probe)
        target = z0.argmax(1)

        def soft(z):
            e = np.exp(z - z.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)

        gap = np.abs(soft(z0)[np.arange(len(z0)), target] - soft(zt)[np.arange(len(zt)), target])
        Xp = torch.tensor(norm(probe))
        tgt = torch.tensor(target).to(DEV)

        def attr(model, method):
            out = []
            for i in range(0, len(Xp), 32):
                x = Xp[i:i + 32].to(DEV)
                t = tgt[i:i + 32]
                if method == "ig":
                    a = IntegratedGradients(model).attribute(
                        x, baselines=torch.zeros_like(x), target=t, n_steps=ig_steps)
                else:
                    a = Saliency(model).attribute(x, target=t, abs=False) * x
                out.append(a.float().detach().cpu().numpy().reshape(len(x), -1))
            return np.concatenate(out)

        for method, ename in (("ig", "integrated_gradients"), ("gxi", "gradient_x_input")):
            A0, An, At = (attr(m, method) for m in (f0, f_null, f_treat))
            for dist, dname in ((d_l1, "l1"), (d_spearman, "spearman")):
                delta = dist(l1_abs(A0), l1_abs(At))
                rho = dist(l1_abs(A0), l1_abs(An))
                for eps in (0.02, 0.05, 0.10):
                    m = preserved_mask(np.zeros(len(gap)), gap, eps)
                    if m.sum() < 15:
                        continue
                    rows.append(summarise(
                        delta[m], np.zeros(int(m.sum())), np.zeros(0), rho[m], rho[m],
                        n_probe=len(probe), seed=seed, family="cifar_corruption",
                        model="resnet18", explainer=ename, distance=dname, phi="abs",
                        features="all", eps=eps,
                        acc_source=float((z0.argmax(1) == yte[i_p]).mean()),
                        acc_treat=float((zt.argmax(1) == yte[i_p]).mean()),
                        agree_treat=float((z0.argmax(1) == zt.argmax(1)).mean()),
                    ).as_dict())
        save_incremental(rows, name)
        print(f"  vision seed {seed} acc={(z0.argmax(1) == yte[i_p]).mean():.3f} "
              f"agree={(z0.argmax(1) == zt.argmax(1)).mean():.3f} ({time.time() - t0:.0f}s)",
              flush=True)

    df = pd.DataFrame(rows)
    _report(df, "ResNet-18  CIFAR-10 corruption")
    return df


# ----------------------------------------------------------------------------- text

def run_text(seeds=3, n_source=5000, n_add=2000, n_probe=250, lr=2e-5, update_lr=5e-6,
             batch=16, max_len=160, ig_steps=16, n_replay=None, update_lrs=None,
             out_name="scale_text.parquet"):
    """DistilBERT, IMDB -> Rotten Tomatoes, with the matched-operator null.

    `update_lrs` sweeps update strength while reusing one source model per seed. That matters: the
    tabular results show the ratio falls toward 1 as the update grows, so a single heavy update
    cannot distinguish "no effect at this scale" from "an update too large to see it through".
    """
    import logging

    import datasets as _ds
    import transformers
    from captum.attr import LayerIntegratedGradients
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    transformers.logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    _ds.logging.set_verbosity_error()

    def load_either(*names):
        """datasets>=4 rejects bare ids like 'imdb'; older versions do not know namespaced ones."""
        last: Exception = RuntimeError(f"no candidate loaded: {names}")
        for n in names:
            try:
                return load_dataset(n)
            except Exception as e:  # noqa: BLE001 - try the next candidate
                last = e
        raise last

    model_name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(model_name)
    imdb = load_either("stanfordnlp/imdb", "imdb")
    rt = load_either("cornell-movie-review-data/rotten_tomatoes", "rotten_tomatoes")

    lrs = [float(x) for x in (update_lrs or [update_lr])]
    prev = pd.read_parquet(OUT / out_name) if (OUT / out_name).exists() else None
    rows = prev.to_dict("records") if prev is not None else []
    done = ({(int(a), float(b)) for a, b in zip(prev.seed, prev.update_lr)}
            if prev is not None and "update_lr" in prev else set())
    if done:
        print(f"  resuming, {len(done)} (seed, lr) cells already on disk", flush=True)

    def encode(texts):
        return tok(list(texts), truncation=True, padding="max_length",
                   max_length=max_len, return_tensors="pt")

    def take(ds, split, n, rng):
        idx = rng.choice(len(ds[split]), n, replace=False)
        sub = ds[split].select(idx.tolist())
        return list(sub["text"]), np.array(sub["label"])

    def fresh():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2).to(DEV)

    def fit(model, enc, labels, n_epochs, learning_rate, seed):
        torch.manual_seed(seed)
        y = torch.tensor(labels)
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scaler = torch.amp.GradScaler("cuda") if AMP else None
        g = torch.Generator().manual_seed(seed)
        model.train()
        steps = 0
        for _ in range(n_epochs):
            order = torch.randperm(len(y), generator=g)
            for i in range(0, len(order), batch):
                b = order[i:i + batch]
                opt.zero_grad(set_to_none=True)
                with amp_ctx():
                    out = model(input_ids=enc["input_ids"][b].to(DEV),
                                attention_mask=enc["attention_mask"][b].to(DEV),
                                labels=y[b].to(DEV))
                if scaler:
                    scaler.scale(out.loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    out.loss.backward()
                    opt.step()
                steps += 1
        model.eval()
        return steps

    @torch.no_grad()
    def probs(model, enc):
        out = []
        for i in range(0, enc["input_ids"].shape[0], 32):
            with amp_ctx():
                lg = model(input_ids=enc["input_ids"][i:i + 32].to(DEV),
                           attention_mask=enc["attention_mask"][i:i + 32].to(DEV)).logits
            out.append(torch.softmax(lg.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def token_attr(model, enc, target, method):
        """Layer attributions on the embedding output, summed over the hidden dimension and
        trimmed to each example's real tokens. Runs in fp32: fp16 gradients through the
        integration steps lose too much precision to compare two checkpoints."""
        emb = model.distilbert.embeddings

        def fwd(input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask).logits

        out = []
        step = 4 if method == "ig" else 16
        for i in range(0, enc["input_ids"].shape[0], step):
            ids = enc["input_ids"][i:i + step].to(DEV)
            am = enc["attention_mask"][i:i + step].to(DEV)
            tgt = torch.tensor(target[i:i + step]).to(DEV)
            if method == "ig":
                base = torch.full_like(ids, tok.pad_token_id)
                base[:, 0] = ids[:, 0]
                a = LayerIntegratedGradients(fwd, emb).attribute(
                    ids, baselines=base, target=tgt,
                    additional_forward_args=(am,), n_steps=ig_steps)
            else:
                store = {}
                h = emb.register_forward_hook(lambda m, i_, o: store.setdefault("v", o))
                lg = fwd(ids, am)
                store["v"].retain_grad()
                lg.gather(1, tgt.view(-1, 1)).sum().backward()
                a = store["v"] * store["v"].grad
                h.remove()
            a = a.sum(-1).float().detach().cpu().numpy()
            for j, row in enumerate(a):
                out.append(row[: int(am[j].sum().item())])
        return out

    for seed in range(seeds):
        todo = [x for x in lrs if (seed, x) not in done]
        if not todo:
            continue
        t0 = time.time()
        rng = np.random.default_rng(seed)
        src_txt, src_y = take(imdb, "train", n_source, rng)
        null_txt, null_y = take(imdb, "train", n_add, np.random.default_rng([seed, 1]))
        treat_txt, treat_y = take(rt, "train", n_add, np.random.default_rng([seed, 2]))
        probe_txt, _ = take(imdb, "test", n_probe, np.random.default_rng([seed, 3]))
        probe_enc = encode(probe_txt)

        torch.manual_seed(seed)
        f0 = fresh()
        fit(f0, encode(src_txt), src_y, 1, lr, seed)
        sd0 = {k: v.detach().clone() for k, v in f0.state_dict().items()}
        p0 = probs(f0, probe_enc)
        target = (p0 >= 0.5).astype(int)

        # Replay a slice, not the whole source. Replaying everything makes the update as large as
        # the original training, and the tabular sweep says that drives the ratio to 1 regardless
        # of whether a shift happened. The slice is identical in both arms.
        k = n_replay if n_replay is not None else n_add
        replay_txt, replay_y = src_txt[:k], src_y[:k]
        enc_null = encode(replay_txt + null_txt)
        enc_treat = encode(replay_txt + treat_txt)
        y_null = np.concatenate([replay_y, null_y])
        y_treat = np.concatenate([replay_y, treat_y])

        A0 = {m: token_attr(f0, probe_enc, target, m) for m in ("ig", "gxi")}

        for cur_lr in todo:
            def updated(enc, y, s, _lr=cur_lr):
                m = fresh()
                m.load_state_dict(sd0)
                return m, fit(m, enc, y, 1, _lr, s)

            f_null, n_null = updated(enc_null, y_null, seed + 1)
            f_treat, n_treat = updated(enc_treat, y_treat, seed + 1)
            assert n_null == n_treat, "matched null must take the same number of steps"

            pn, pt = probs(f_null, probe_enc), probs(f_treat, probe_enc)
            gap = np.abs(p0 - pt)
            pres = float(np.mean(gap <= 0.05))
            agree = float(((p0 >= .5) == (pt >= .5)).mean())
            print(f"    lr={cur_lr:g}  agree={agree:.3f}  preserved@0.05={pres:.2f} "
                  f"@0.10={np.mean(gap <= .10):.2f}  median|dp|={np.median(gap):.4f}", flush=True)
            if pres < 0.15:
                print("    WARNING: too few preserved points; try a smaller --text-update-lr",
                      flush=True)

            for method, ename in (("ig", "integrated_gradients"),
                                  ("gxi", "gradient_x_input")):
                An = token_attr(f_null, probe_enc, target, method)
                At = token_attr(f_treat, probe_enc, target, method)
                for dist, dname in ((d_l1, "l1"), (d_spearman, "spearman")):
                    delta = row_change(A0[method], At, dist)
                    rho = row_change(A0[method], An, dist)
                    for eps in (0.02, 0.05, 0.10, 0.20):
                        m = preserved_mask(p0, pt, eps)
                        if m.sum() < 10:
                            continue
                        rows.append(summarise(
                            delta[m], np.zeros(int(m.sum())), np.zeros(0), rho[m], rho[m],
                            n_probe=len(p0), seed=seed, family="imdb_to_rt",
                            model="distilbert", explainer=ename, distance=dname,
                            phi="abs", features="all", eps=eps,
                            update_lr=cur_lr, n_replay=k, n_steps=n_treat,
                            agree_treat=agree,
                            agree_null=float(((p0 >= .5) == (pn >= .5)).mean()),
                        ).as_dict())
            save_incremental(rows, out_name)
        print(f"  text seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    _report(df, "DistilBERT  IMDB -> Rotten Tomatoes")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",
                    choices=["width", "vision", "text", "text-sweep", "all"],
                    default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--width-seeds", type=int, default=5)
    ap.add_argument("--text-update-lr", type=float, default=5e-6,
                    help="lower this if preserved@0.05 comes out below 0.15")
    ap.add_argument("--text-n-source", type=int, default=5000)
    ap.add_argument("--text-n-add", type=int, default=2000)
    ap.add_argument("--text-update-lrs", type=float, nargs="+",
                    default=[1e-6, 3e-6, 1e-5],
                    help="update-strength sweep for --task text-sweep")
    a = ap.parse_args()

    gpu = torch.cuda.get_device_name(0) if DEV == "cuda" else "NONE"
    print(f"device={DEV} ({gpu})  amp={AMP}  torch={torch.__version__}  out={OUT}", flush=True)
    if DEV == "cpu":
        print("WARNING: no GPU detected. Enable an accelerator in Kaggle settings "
              "(T4 x2 or P100); on CPU the text arm will take many hours.", flush=True)

    tasks = ["width", "vision", "text"] if a.task == "all" else [a.task]
    tasks = ["text" if t == "text-sweep" else t for t in tasks]
    check_deps(tasks)
    OUT.mkdir(parents=True, exist_ok=True)

    if a.task in ("width", "all"):
        run_width(seeds=a.width_seeds)
    if a.task in ("vision", "all"):
        run_vision(seeds=a.seeds)
    if a.task == "text-sweep":
        run_text(seeds=a.seeds, update_lrs=a.text_update_lrs,
                 n_source=a.text_n_source, n_add=a.text_n_add,
                 out_name="scale_text_sweep.parquet")
    elif a.task in ("text", "all"):
        run_text(seeds=a.seeds, update_lr=a.text_update_lr,
                 n_source=a.text_n_source, n_add=a.text_n_add)


if __name__ == "__main__":
    main()
