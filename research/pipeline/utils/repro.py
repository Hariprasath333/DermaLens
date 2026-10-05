"""Reproducibility helpers: seeding and manifest hashing."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch deterministically.

    PyTorch is imported lazily so this module is safe to import in
    environments without torch (e.g. data-only utilities).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def manifest_hash(obj: Any) -> str:
    """SHA1 of a stable JSON-serialization of `obj`.

    Used as a fingerprint for experiment configs: same config -> same
    hash -> we can safely skip re-running.
    """
    serialized = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()
