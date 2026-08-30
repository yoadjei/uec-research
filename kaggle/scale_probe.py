"""Scale probe for the H100. Closes the one audit item that a CPU cannot reach.

Three experiments, all reusing the same matched-operator design as the CPU work:

  text    DistilBERT (66M params) fine-tuned on IMDB, updated additively with either more IMDB
          (matched null) or Rotten Tomatoes (treatment). Token attributions via layer IG and
          layer Grad x Input. This is the headline scale result -- three orders of magnitude
          above the MLPs.
  vision  A real ResNet-18 on CIFAR-10, replacing the 78k-parameter stand-in the CPU budget forced.
  width   Synthetic MLPs at widths 32 -> 1024, answering whether the ratio vanishes with capacity.

Every arm uses an *additive* update: the model is fine-tuned on its old data plus new data that is
either same-distribution (null) or shifted (treatment). Training on the new data alone is domain
replacement, not an update, and it destroys the prediction-preserved probe the comparison needs.

Run on Kaggle:

    !git clone -q https://github.com/yoadjei/uec-research.git
    !pip install -q transformers datasets
    !python uec-research/kaggle/scale_probe.py --task all

Outputs one parquet per task in the working directory.
"""

import argparse
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
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO / "results"


def row_change(A, B, distance=d_l1):
    """Distance between two ragged attribution lists, per example."""
    return np.array([
        float(distance(l1_abs(a[None, :]), l1_abs(b[None, :]))[0]) for a, b in zip(A, B)
    ])


# ----------------------------------------------------------------------------- text

def run_text(seeds=3, n_source=6000, n_add=3000, n_probe=250, epochs=1,
             lr=2e-5, update_lr=1e-5, batch=32, max_len=192):
    from datasets import load_dataset
    from captum.attr import LayerIntegratedGradients
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

    name = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    imdb = load_dataset("imdb")
    rt = load_dataset("rotten_tomatoes")

    def encode(texts):
        return tok(list(texts), truncation=True, padding="max_length",
                   max_length=max_len, return_tensors="pt")

    def take(ds, split, n, rng):
        idx = rng.choice(len(ds[split]), n, replace=False)
        return [ds[split][int(i)]["text"] for i in idx], np.array(
            [ds[split][int(i)]["label"] for i in idx])

    def fit(model, texts, labels, n_epochs, learning_rate, seed):
        torch.manual_seed(seed)
        enc = encode(texts)
        y = torch.tensor(labels)
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()
        steps = 0
        g = torch.Generator().manual_seed(seed)
        for _ in range(n_epochs):
            order = torch.randperm(len(y), generator=g)
            for i in range(0, len(order), batch):
                b = order[i:i + batch]
                opt.zero_grad(set_to_none=True)
                out = model(input_ids=enc["input_ids"][b].to(DEV),
                            attention_mask=enc["attention_mask"][b].to(DEV),
                            labels=y[b].to(DEV))
                out.loss.backward()
                opt.step()
                steps += 1
        model.eval()
        return steps

    @torch.no_grad()
    def probs(model, enc):
        out = []
        for i in range(0, enc["input_ids"].shape[0], batch):
            logits = model(input_ids=enc["input_ids"][i:i + batch].to(DEV),
                           attention_mask=enc["attention_mask"][i:i + batch].to(DEV)).logits
            out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def token_attr(model, enc, target, method="ig", n_steps=24):
        """Layer attributions on the embedding output, summed over the hidden dimension and
        trimmed to each example's real tokens."""
        emb = model.distilbert.embeddings

        def fwd(input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask).logits

        out = []
        for i in range(0, enc["input_ids"].shape[0], 8):
            ids = enc["input_ids"][i:i + 8].to(DEV)
            am = enc["attention_mask"][i:i + 8].to(DEV)
            tgt = torch.tensor(target[i:i + 8]).to(DEV)
            if method == "ig":
                lig = LayerIntegratedGradients(fwd, emb)
                base = torch.full_like(ids, tok.pad_token_id)
                base[:, 0] = tok.cls_token_id
                a = lig.attribute(ids, baselines=base, target=tgt,
                                  additional_forward_args=(am,), n_steps=n_steps)
            else:
                acts, grads = {}, {}
                h = emb.register_forward_hook(lambda m, i_, o: acts.__setitem__("v", o))
                e = emb(ids)
                e.retain_grad()
                logits = fwd(ids, am)
                logits.gather(1, tgt.view(-1, 1)).sum().backward()
                a = (acts["v"] * acts["v"].grad) if acts["v"].grad is not None else e * e.grad
                h.remove()
            a = a.sum(-1).detach().cpu().numpy()
            for j, row in enumerate(a):
                k = int(am[j].sum().item())
                out.append(row[:k])
        return out

    rows = []
    for seed in range(seeds):
        t0 = time.time()
        rng = np.random.default_rng(seed)
        src_txt, src_y = take(imdb, "train", n_source, rng)
        add_null_txt, add_null_y = take(imdb, "train", n_add, np.random.default_rng([seed, 1]))
        add_treat_txt, add_treat_y = take(rt, "train", n_add, np.random.default_rng([seed, 2]))
        probe_txt, probe_y = take(imdb, "test", n_probe, np.random.default_rng([seed, 3]))
        probe_enc = encode(probe_txt)

        torch.manual_seed(seed)
        f0 = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(DEV)
        fit(f0, src_txt, src_y, epochs, lr, seed)
        sd0 = {k: v.detach().clone() for k, v in f0.state_dict().items()}

        def updated(add_txt, add_y, s):
            m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(DEV)
            m.load_state_dict(sd0)
            n = fit(m, list(src_txt) + list(add_txt), np.concatenate([src_y, add_y]),
                    1, update_lr, s)
            return m, n

        f_null, n_null = updated(add_null_txt, add_null_y, seed + 1)
        f_treat, n_treat = updated(add_treat_txt, add_treat_y, seed + 1)
        assert n_null == n_treat

        p0, pn, pt = (probs(m, probe_enc) for m in (f0, f_null, f_treat))
        target = (p0 >= 0.5).astype(int)

        for method in ("ig", "gxi"):
            A0 = token_attr(f0, probe_enc, target, method)
            An = token_attr(f_null, probe_enc, target, method)
            At = token_attr(f_treat, probe_enc, target, method)
            for dist, dname in ((d_l1, "l1"), (d_spearman, "spearman")):
                delta = row_change(A0, At, dist)
                rho = row_change(A0, An, dist)
                for eps in (0.02, 0.05, 0.10):
                    m = preserved_mask(p0, pt, eps)
                    if m.sum() < 15:
                        continue
                    s = summarise(delta[m], np.zeros(int(m.sum())), np.zeros(0),
                                  rho[m], rho[m], n_probe=len(p0),
                                  seed=seed, family="imdb_to_rt", model="distilbert",
                                  explainer={"ig": "integrated_gradients",
                                             "gxi": "gradient_x_input"}[method],
                                  distance=dname, phi="abs", features="all", eps=eps,
                                  agree_treat=float(((p0 >= .5) == (pt >= .5)).mean()),
                                  agree_null=float(((p0 >= .5) == (pn >= .5)).mean()),
                                  n_steps=n_treat)
                    rows.append(s.as_dict())
        print(f"  text seed {seed} agree={((p0 >= .5) == (pt >= .5)).mean():.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "scale_text.parquet")
    _report(df, "DistilBERT  IMDB -> Rotten Tomatoes")
    return df


# --------------------------------------------------------------------------- vision

def run_vision(seeds=3, n_source=20000, n_add=10000, n_probe=500, epochs=8,
               lr=1e-3, update_lr=1e-4, batch=128):
    import torchvision
    from captum.attr import IntegratedGradients, Saliency
    from torchvision import transforms

    root = "/kaggle/working/cifar" if OUT.name == "working" else str(REPO / "data" / "cifar")
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
    norm = lambda X: (X - mean) / std

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
        Xt = torch.tensor(norm(X))
        yt = torch.tensor(y)
        g = torch.Generator().manual_seed(seed)
        model.train()
        steps = 0
        for _ in range(n_epochs):
            order = torch.randperm(len(yt), generator=g)
            for i in range(0, len(order), batch):
                b = order[i:i + batch]
                opt.zero_grad(set_to_none=True)
                lossf(model(Xt[b].to(DEV)), yt[b].to(DEV)).backward()
                opt.step()
                steps += 1
        model.eval()
        return steps

    @torch.no_grad()
    def logits(model, X):
        out = []
        Xt = torch.tensor(norm(X))
        for i in range(0, len(Xt), 256):
            out.append(model(Xt[i:i + 256].to(DEV)).cpu().numpy())
        return np.concatenate(out)

    rows = []
    for seed in range(seeds):
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
            n = fit(m, np.concatenate([Xs, add_X]), np.concatenate([ys, add_y]),
                    1, update_lr, s)
            return m, n

        f_null, n_null = updated(Xtr[i_a], ytr[i_a], seed + 1)
        f_treat, n_treat = updated(corrupt(Xtr[i_b], rng), ytr[i_b], seed + 1)
        assert n_null == n_treat

        z0, zt = logits(f0, probe), logits(f_treat, probe)
        target = z0.argmax(1)
        soft = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(
            z - z.max(1, keepdims=True)).sum(1, keepdims=True)
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
                        x, baselines=torch.zeros_like(x), target=t, n_steps=16)
                else:
                    a = Saliency(model).attribute(x, target=t, abs=False) * x
                out.append(a.detach().cpu().numpy().reshape(len(x), -1))
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
                    s = summarise(delta[m], np.zeros(int(m.sum())), np.zeros(0),
                                  rho[m], rho[m], n_probe=len(probe),
                                  seed=seed, family="cifar_corruption", model="resnet18",
                                  explainer=ename, distance=dname, phi="abs",
                                  features="all", eps=eps,
                                  acc_source=float((z0.argmax(1) == yte[i_p]).mean()),
                                  acc_treat=float((zt.argmax(1) == yte[i_p]).mean()),
                                  agree_treat=float((z0.argmax(1) == zt.argmax(1)).mean()))
                    rows.append(s.as_dict())
        print(f"  vision seed {seed} acc={(z0.argmax(1) == yte[i_p]).mean():.3f} "
              f"agree={(z0.argmax(1) == zt.argmax(1)).mean():.3f} ({time.time() - t0:.0f}s)",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "scale_vision.parquet")
    _report(df, "ResNet-18  CIFAR-10 corruption")
    return df


# ---------------------------------------------------------------------------- width

def run_width(seeds=5, widths=(32, 64, 128, 256, 512, 1024)):
    """Does the ratio vanish with capacity? The CPU ablation stopped at 128 and showed a mild
    decline, which is the one piece of evidence that cuts against the scale argument."""
    from uec.data.support import shared_support_probe
    from uec.data.synthetic import make_pair
    from uec.explain.cache import attribute
    from uec.metrics.uec import change
    from uec.models.mlp import probabilities
    from uec.train.harness import TrainConfig, UpdateConfig, build_checkpoints

    rows = []
    for seed in range(seeds):
        src, tgt = make_pair("covariate", magnitude=1.5)
        probe = shared_support_probe(src, tgt, 500, np.random.default_rng(500 + seed))
        for w in widths:
            t0 = time.time()
            ck = build_checkpoints(src, tgt, seed, 8000, 4000,
                                   TrainConfig(epochs=60, hidden=(w, w)),
                                   UpdateConfig(lr=2e-4, epochs=2),
                                   regimes=("matched_null", "treatment"))
            f0, f1, fn = ck["source"][0], ck["treatment"][0], ck["matched_null"][0]
            m = preserved_mask(probabilities(f0, probe), probabilities(f1, probe), 0.05)
            for name in ("integrated_gradients", "gradient_x_input"):
                A0 = attribute(f0, probe, name, use_cache=False)
                delta = change(A0, attribute(f1, probe, name, use_cache=False), l1_abs, d_l1)[m]
                rho = change(A0, attribute(fn, probe, name, use_cache=False), l1_abs, d_l1)[m]
                rows.append({"seed": seed, "width": w, "explainer": name,
                             "n_params": 20 * w + w * w + w,
                             "delta": delta.mean(), "rho_null": rho.mean(),
                             "ratio": delta.mean() / rho.mean(),
                             "n_preserved": int(m.sum()), "seconds": time.time() - t0})
            print(f"  width {w:5d} seed {seed} ratio={rows[-1]['ratio']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "scale_width.parquet")
    print("\n" + df.groupby(["width", "explainer"])[["delta", "rho_null", "ratio"]]
          .mean().round(4).to_string())
    return df


def _report(df, title):
    q = df[(df.distance == "l1") & (df.eps == 0.05)]
    print(f"\n=== {title} ===")
    print(q.groupby("explainer")[["rho_null", "delta", "ratio", "preserved_frac",
                                  "agree_treat"]].mean().round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["text", "vision", "width", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    print(f"device={DEV}  torch={torch.__version__}  out={OUT}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    if a.task in ("width", "all"):
        run_width(seeds=max(a.seeds, 5))
    if a.task in ("vision", "all"):
        run_vision(seeds=a.seeds)
    if a.task in ("text", "all"):
        run_text(seeds=a.seeds)


if __name__ == "__main__":
    main()
