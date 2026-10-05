"""
LesionIQ — Layer-0 CSV Builder
================================

Builds the 3 manifests the training/inference pipeline expects:

    layer0_train.csv   — 80% of ISIC 2019 training set, stratified
    layer0_val.csv     — 20% of ISIC 2019 training set, stratified
    layer0_test.csv    — ISIC 2019 public test set (8,238 images)

All 3 share the exact column schema consumed by
``backend/classifier/dataloader.py``:

    image_path        absolute file path to the image
    class_encoded     integer 0..7 (MEL, NV, BCC, AK, BKL, DF, VASC, SCC)
    age_approx        float age (NaN -> 0)
    sex_female / sex_male / sex_unknown            (one-hot)
    site_anterior torso / site_head/neck / ...      (one-hot, 9 sites
        including "unknown")

ISIC UNK-class rows are dropped (8-class task, UNK is the 9th unknown bin).

Run once after placing the dataset:

    $env:PYTHONPATH = "C:\\LesionIQ"
    python -m backend.data.build_layer0_csvs \\
        --train-images dataset/ISIC_2019_Training_Input/ISIC_2019_Training_Input \\
        --test-images  dataset/ISIC_2019_Test_Input

The script auto-finds the GroundTruth + Metadata CSVs under
``backend/data/`` (the standard ISIC filenames). Override with
``--train-gt``, ``--train-meta``, ``--test-gt``, ``--test-meta`` if needed.

Output: written to ``backend/data/layer0_{train,val,test}.csv`` by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# ── Schema (mirrors backend/classifier/dataloader.py) ──────────────
LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
SITE_COLS = [
    "site_anterior torso",
    "site_head/neck",
    "site_lateral torso",
    "site_lower extremity",
    "site_oral/genital",
    "site_palms/soles",
    "site_posterior torso",
    "site_upper extremity",
    "site_unknown",
]
SEX_COLS = ["sex_female", "sex_male", "sex_unknown"]
META_COLS = ["age_approx", *SEX_COLS, *SITE_COLS]
SITE_ALIASES = {s.replace("site_", "") for s in SITE_COLS}

REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_DIR   = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = DATA_DIR


# ── Helpers ────────────────────────────────────────────────────────

def _absolute(path_like: str | Path) -> Path:
    """Resolve a CLI path against the repo root for relative inputs."""
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _merge_and_encode(gt_csv: Path, meta_csv: Path, image_dir: Path,
                      drop_unk: bool = True) -> pd.DataFrame:
    """Read ISIC GroundTruth + Metadata, return a layer-0-ready DataFrame."""
    print(f"\n[READ] GroundTruth: {gt_csv}")
    gt = pd.read_csv(gt_csv)
    print(f"[READ] Metadata   : {meta_csv}")
    meta = pd.read_csv(meta_csv)

    df = gt.merge(meta, on="image", how="left")
    print(f"[OK]   Merged: {len(df)} rows, {len(df.columns)} cols")

    # Drop rows whose only label is UNK (8-class task)
    if drop_unk and "UNK" in df.columns:
        before = len(df)
        df = df[df["UNK"] != 1.0].copy()
        df = df.drop(columns=["UNK"])
        print(f"[FILT] Dropped {before - len(df)} UNK-class rows -> {len(df)} remain")

    # ── Encode class ──
    label_arr = df[LABEL_COLS].to_numpy(dtype=np.float32)
    if (label_arr.sum(axis=1) == 0).any():
        n_blank = int((label_arr.sum(axis=1) == 0).sum())
        raise RuntimeError(f"{n_blank} rows have no label (all-zero one-hot)")
    df["class_encoded"] = label_arr.argmax(axis=1).astype(np.int64)

    # ── image_path (absolute) ──
    df["image_path"] = df["image"].apply(lambda x: str((image_dir / f"{x}.jpg").resolve()))

    # ── age ──
    # Empty -> 0 (dataloader divides by 90; population-mean-ish fallback)
    df["age_approx"] = pd.to_numeric(df.get("age_approx", 0), errors="coerce").fillna(0)

    # ── sex one-hot ──
    sex_lower = df.get("sex", pd.Series(["unknown"] * len(df))).astype(str).str.lower().str.strip()
    df["sex_female"]  = (sex_lower == "female").astype(np.float32)
    df["sex_male"]    = (sex_lower == "male").astype(np.float32)
    df["sex_unknown"] = ((~sex_lower.isin(["female", "male"]))).astype(np.float32)

    # ── site one-hot (9 dims incl. "unknown") ──
    raw_site = df.get("anatom_site_general", pd.Series([""] * len(df))).astype(str).str.lower().str.strip()
    # ISIC uses "" or "NA" for unknown — fold them all into "unknown"
    site_normalized = raw_site.apply(lambda v: v if v in SITE_ALIASES else "unknown")
    for col in SITE_COLS:
        bare = col.replace("site_", "")
        df[col] = (site_normalized == bare).astype(np.float32)

    # ── Verify image files actually exist (sample check) ──
    missing = 0
    for sample_path in df["image_path"].sample(min(20, len(df)), random_state=0):
        if not Path(sample_path).exists():
            missing += 1
    if missing:
        raise FileNotFoundError(
            f"{missing}/20 sampled image_path entries do not exist. "
            f"Pass --train-images / --test-images pointing at the actual "
            f"directory holding the .jpg files."
        )

    keep = ["image_path", "class_encoded", *META_COLS]
    return df[keep].reset_index(drop=True)


def _stratified_split(df: pd.DataFrame, val_frac: float = 0.20,
                       random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified by class_encoded — same convention used elsewhere
    in the codebase (StratifiedShuffleSplit, random_state=42)."""
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=random_state
    )
    idx_train, idx_val = next(splitter.split(df, df["class_encoded"]))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_val].reset_index(drop=True)


def _print_class_distribution(df: pd.DataFrame, tag: str) -> None:
    counts = df["class_encoded"].value_counts().sort_index()
    total = len(df)
    print(f"\n[DIST] {tag} ({total} rows)")
    for i, name in enumerate(LABEL_COLS):
        n = int(counts.get(i, 0))
        pct = 100 * n / total if total else 0
        print(f"   {name:<5s} {n:>6}  ({pct:5.2f}%)")


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Build LesionIQ layer-0 CSVs.")
    ap.add_argument("--train-images", required=True,
                    help="Directory containing the ISIC 2019 training .jpg files.")
    ap.add_argument("--test-images",  required=True,
                    help="Directory containing the ISIC 2019 test .jpg files.")
    ap.add_argument("--train-gt",   default=str(DATA_DIR / "ISIC_2019_Training_GroundTruth.csv"))
    ap.add_argument("--train-meta", default=str(DATA_DIR / "ISIC_2019_Training_Metadata.csv"))
    ap.add_argument("--test-gt",    default=str(DATA_DIR / "ISIC_2019_Test_GroundTruth.csv"))
    ap.add_argument("--test-meta",  default=str(DATA_DIR / "ISIC_2019_Test_Metadata.csv"))
    ap.add_argument("--out-dir",    default=str(DEFAULT_OUT_DIR),
                    help="Where to write layer0_{train,val,test}.csv.")
    ap.add_argument("--val-frac",   type=float, default=0.20)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    out_dir = _absolute(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training set: merge + encode + stratified 80/20 split ──
    print("\n" + "=" * 62)
    print(" Building TRAIN + VAL manifests")
    print("=" * 62)
    train_df = _merge_and_encode(
        gt_csv=_absolute(args.train_gt),
        meta_csv=_absolute(args.train_meta),
        image_dir=_absolute(args.train_images),
    )

    train_split, val_split = _stratified_split(
        train_df, val_frac=args.val_frac, random_state=args.seed
    )
    _print_class_distribution(train_split, "TRAIN split")
    _print_class_distribution(val_split,   "VAL split")

    train_path = out_dir / "layer0_train.csv"
    val_path   = out_dir / "layer0_val.csv"
    train_split.to_csv(train_path, index=False)
    val_split.to_csv(val_path, index=False)
    print(f"\n[WRITE] {train_path}  ({len(train_split)} rows)")
    print(f"[WRITE] {val_path}    ({len(val_split)} rows)")

    # ── Test set: merge + encode, no split ──
    print("\n" + "=" * 62)
    print(" Building TEST manifest")
    print("=" * 62)
    test_df = _merge_and_encode(
        gt_csv=_absolute(args.test_gt),
        meta_csv=_absolute(args.test_meta),
        image_dir=_absolute(args.test_images),
    )
    _print_class_distribution(test_df, "TEST")

    test_path = out_dir / "layer0_test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"\n[WRITE] {test_path}   ({len(test_df)} rows)")

    print("\n" + "=" * 62)
    print(" Done. Update backend/config.yaml (or env vars) to point at:")
    print(f"   train_csv: {train_path}")
    print(f"   val_csv:   {val_path}")
    print(f"   test_csv:  {test_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
