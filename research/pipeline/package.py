"""
Package the pipeline into a single transportable zip.

Output: lesioniq-pipeline-vX.Y.zip in this directory.

The zip contains everything needed to run the pipeline on a fresh
machine: code, bootstrap scripts, requirements, docs. It does NOT
include:
  - any dataset images
  - any model checkpoints
  - the local .venv/
  - __pycache__/ or transient runtime artifacts
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

VERSION = "1.2.0"
OUT_NAME = f"lesioniq-pipeline-v{VERSION}.zip"

# Inclusion list (relative to HERE)
INCLUDE_FILES = [
    "README.md",
    "pipeline.yaml",
    "requirements.txt",
    "run.py",
    "lesioniq.bat",
    "lesioniq.sh",
    "package.py",
]
INCLUDE_DIRS = [
    "docs",
    "stages",
    "models",
    "utils",
]

# Exclusion patterns
EXCLUDE_FRAGMENTS = (
    "__pycache__",
    ".pyc",
    ".pyo",
    ".venv/",
    "runs/",
    "splits/",
    "preprocessed/",
    "datasets/",
)


def _include(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    for frag in EXCLUDE_FRAGMENTS:
        if frag in rel:
            return False
    return True


def main() -> int:
    out = HERE / OUT_NAME
    if out.exists():
        out.unlink()
    print(f"[package] writing {out}")
    n_files = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        # standalone files
        for fname in INCLUDE_FILES:
            p = HERE / fname
            if not p.exists():
                print(f"[package] WARNING missing {fname}, skipped")
                continue
            z.write(p, arcname=f"lesioniq-pipeline/{fname}")
            n_files += 1

        # directories
        for d in INCLUDE_DIRS:
            dpath = HERE / d
            if not dpath.exists():
                print(f"[package] WARNING missing dir {d}, skipped")
                continue
            for root, dirs, files in os.walk(dpath):
                # prune excluded subdirs in-place to skip descent
                dirs[:] = [dd for dd in dirs
                           if _include(str(Path(root) / dd))]
                for fname in files:
                    full = Path(root) / fname
                    rel = full.relative_to(HERE).as_posix()
                    if not _include(rel):
                        continue
                    z.write(full, arcname=f"lesioniq-pipeline/{rel}")
                    n_files += 1

    print(f"[package] OK: {n_files} files packaged into {out}")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"[package] size: {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
