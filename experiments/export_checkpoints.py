"""Export the headline checkpoints named in results/registry.csv.

The registry records a weight hash for every run; without the weights that hash is unverifiable.
These are 2-layer MLPs -- a few tens of kilobytes each -- so the whole headline set fits in the
repository and a reader can recompute any attribution we report.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uec.data.synthetic import make_pair  # noqa: E402
from uec.paths import ROOT, checkpoint_hash  # noqa: E402
from uec.rng import pin_threads  # noqa: E402
from uec.train.harness import TrainConfig, UpdateConfig, build_checkpoints  # noqa: E402

OUT = ROOT / "checkpoints"
HEADER = "file,shift,regime,seed,ckpt_hash,n_train,n_steps"


def main(seeds: int = 3, families=("none", "covariate", "concept", "shortcut")):
    pin_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, ucfg = TrainConfig(epochs=60), UpdateConfig(lr=2e-4, epochs=2)
    manifest = []

    for seed in range(seeds):
        for family in families:
            src, tgt = make_pair(family, magnitude=1.5)
            ck = build_checkpoints(src, tgt, seed, 8000, 4000, cfg, ucfg)
            for regime, (model, meta) in ck.items():
                h = checkpoint_hash(model.state_dict())
                name = f"synthetic_{family}_{regime}_s{seed}_{h}.pt"
                torch.save(model.state_dict(), OUT / name)
                manifest.append(
                    f"{name},{family},{regime},{seed},{h},{meta['n_train']},{meta['n_steps']}"
                )
        print(f"  seed {seed} exported", flush=True)

    (OUT / "MANIFEST.csv").write_text("\n".join([HEADER, *manifest]) + "\n", encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.glob("*.pt"))
    print(f"\nwrote {len(manifest)} checkpoints, {total / 1024:.0f} KB total")


if __name__ == "__main__":
    main()
