"""Consistent log config used by every stage."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None
                  ) -> None:
    """Idempotent setup: removes existing handlers and reinstalls."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(name)s  %(levelname)-7s  %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(ch)
    root.setLevel(level)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(fh)
