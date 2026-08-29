import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def pin_threads(n: int = 1) -> None:
    """CPU reductions are only bitwise-reproducible at a fixed thread count."""
    torch.set_num_threads(n)
