import csv
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REGISTRY = RESULTS / "registry.csv"

_REGISTRY_FIELDS = [
    "run_id", "dataset", "shift", "regime", "seed", "config_hash",
    "ckpt_hash", "n_train", "n_steps", "lr", "epochs", "acc", "path",
]


def _canonical(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def config_hash(cfg, n: int = 12) -> str:
    blob = json.dumps(_canonical(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


def checkpoint_hash(state_dict, n: int = 12) -> str:
    """Identifies a checkpoint by its weights, so a stale cache can never be served."""
    h = hashlib.sha1()
    for key in sorted(state_dict):
        t = state_dict[key]
        t = t.detach().cpu() if isinstance(t, torch.Tensor) else torch.as_tensor(t)
        h.update(key.encode())
        h.update(np.ascontiguousarray(t.numpy()).tobytes())
    return h.hexdigest()[:n]


def run_dir(dataset: str, shift: str, regime: str, seed: int) -> Path:
    p = RUNS / dataset / shift / regime / f"seed{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_registry(record: dict) -> None:
    """`regime` must never be the bare string "null": pandas reads it back as NaN, which would
    silently drop every matched-null row for anyone reproducing from the registry."""
    assert record.get("regime") != "null", "rename the regime; 'null' round-trips as NaN"

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    new = not REGISTRY.exists()
    with REGISTRY.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_REGISTRY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(record)
