"""
Stage 4 — Dataloader.

Replaces the hackathon dataloader, which had several bugs:
  - Single hard-coded 13-d metadata schema (no multi-dataset support)
  - Augmentation arg errors (e.g. shift_limit on OpticalDistortion is
    not a valid kwarg in modern Albumentations)
  - No per-row missing-metadata mask (downstream SchemaAligner needs it)
  - No clean separation of train vs eval transform pipelines
  - WeightedRandomSampler logic conflated with the dataset class

What this module provides:

  * CanonicalDataset:
      Reads a Stage-3 CSV. Each row already has canonical schema. Per
      __getitem__: returns (image_tensor, metadata_vector, metadata_mask,
      label, row_id). The metadata mask is essential for the
      SchemaAligner in Stage 6 to distinguish "feature absent" from
      "feature present and zero".

  * Augmentation factory:
      Two named pipelines: 'train_aggressive' and 'eval'. Both validated
      against current Albumentations API. No silent argument typos.

  * build_loaders():
      One call builds train / val_select / val_calibrate / test loaders
      from a Stage-3 split directory. Returns a dict of DataLoaders plus
      a small SplitInfo dataclass with sizes + class counts.

  * Class-balanced sampling:
      Optional. WeightedRandomSampler with sqrt-inverse-frequency
      weights (less aggressive than 1/freq, which over-amplifies the
      rarest classes and produces high-variance gradients).

Design priorities:
  1. Efficiency  — albumentations pipelines built once at module import,
                   not per __getitem__; pin_memory, persistent_workers,
                   prefetch_factor all tuned; image read via OpenCV
                   (faster than PIL for JPG)
  2. Quality     — types everywhere; pure factories; no globals
  3. Errorless   — graceful per-row failure (returns a dummy sample with
                   `valid=False`) so a single corrupt image doesn't kill
                   the epoch; documented at the public API
  4. Explainable — dataset __init__ verifies every column it needs and
                   tells the user which is missing
  5. Logic       — the canonical metadata vector has one fixed schema
                   regardless of source dataset; missing fields use a
                   distinct sentinel (mask bit = 0) rather than a magic
                   number
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stages.stage3_split import (  # noqa: E402
    CANONICAL_CLASSES, CANONICAL_SITES, CLASS_TO_IDX,
)


log = logging.getLogger("lesioniq.stage4")


# ─────────────────────────────────────────────────────────────────────
# Canonical metadata feature layout
# ─────────────────────────────────────────────────────────────────────
# Order: age (normalized), sex 3-way one-hot, site 9-way one-hot,
#        fitzpatrick 6-way one-hot, dataset_source (filled at the
#        SchemaAligner level later — not here).
# Total: 1 + 3 + 9 + 6 = 19 features
# ─────────────────────────────────────────────────────────────────────

META_FEATURE_NAMES: tuple[str, ...] = (
    "age_norm",
    "sex_female", "sex_male", "sex_unknown",
    *[f"site_{s}" for s in CANONICAL_SITES],
    "fitz_1", "fitz_2", "fitz_3", "fitz_4", "fitz_5", "fitz_6",
)
META_DIM: int = len(META_FEATURE_NAMES)
assert META_DIM == 19, "META_FEATURE_NAMES drift; downstream models depend on this"


def encode_row_metadata(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Encode one CSV row to (canonical_meta_vector, presence_mask).

    Both are float32 arrays of length META_DIM. The mask is 1.0 where the
    feature is present, 0.0 where it is missing (in which case the
    corresponding meta value is also 0.0; downstream SchemaAligner
    replaces it with a learned 'absent' embedding).
    """
    vec = np.zeros(META_DIM, dtype=np.float32)
    mask = np.zeros(META_DIM, dtype=np.float32)

    # --- age ---
    age = row.get("age")
    try:
        if age is not None and not pd.isna(age):
            a = float(age)
            if 0 < a < 130:
                vec[0] = a / 100.0
                mask[0] = 1.0
    except (TypeError, ValueError):
        pass

    # --- sex ---
    sex = row.get("sex")
    if isinstance(sex, str) and sex.strip().lower() in ("female", "male"):
        col = "sex_" + sex.strip().lower()
        idx = META_FEATURE_NAMES.index(col)
        vec[idx] = 1.0
        # all three sex bits become "present" for the SchemaAligner
        for s in ("sex_female", "sex_male", "sex_unknown"):
            mask[META_FEATURE_NAMES.index(s)] = 1.0
    elif sex is None or pd.isna(sex):
        # missing — mask stays 0 on all three sex columns
        pass
    else:
        # treat anything else as unknown but PRESENT
        vec[META_FEATURE_NAMES.index("sex_unknown")] = 1.0
        for s in ("sex_female", "sex_male", "sex_unknown"):
            mask[META_FEATURE_NAMES.index(s)] = 1.0

    # --- site ---
    site = row.get("site")
    if site is None or pd.isna(site):
        pass
    else:
        site_str = str(site).strip().lower()
        if site_str in CANONICAL_SITES:
            col = "site_" + site_str
            idx = META_FEATURE_NAMES.index(col)
            vec[idx] = 1.0
            for s in CANONICAL_SITES:
                mask[META_FEATURE_NAMES.index("site_" + s)] = 1.0
        else:
            # unknown site — keep all-zero vector but mark presence on site_unknown
            vec[META_FEATURE_NAMES.index("site_unknown")] = 1.0
            for s in CANONICAL_SITES:
                mask[META_FEATURE_NAMES.index("site_" + s)] = 1.0

    # --- fitzpatrick ---
    fitz = row.get("fitzpatrick")
    if fitz is not None and not pd.isna(fitz):
        try:
            f = int(fitz)
            if 1 <= f <= 6:
                col = f"fitz_{f}"
                idx = META_FEATURE_NAMES.index(col)
                vec[idx] = 1.0
                for fi in range(1, 7):
                    mask[META_FEATURE_NAMES.index(f"fitz_{fi}")] = 1.0
        except (TypeError, ValueError):
            pass

    return vec, mask


# ─────────────────────────────────────────────────────────────────────
# Augmentations — built once at module import, no per-call rebuild
# ─────────────────────────────────────────────────────────────────────

def _build_train_augs(img_size: int) -> A.Compose:
    """Aggressive training augmentations.

    Designed to bridge cross-clinic acquisition shift (different
    dermoscopes, white balance, JPEG compression). All args validated
    against current Albumentations API.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.85, 1.15),
            rotate=(-45, 45),
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(
            hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=25,
            p=0.6),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                       hue=0.05, p=0.4),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=6.0, p=1.0),
            A.GridDistortion(p=1.0),
        ], p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(p=0.2),
        A.ImageCompression(quality_range=(60, 100), p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.4,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _build_eval_augs(img_size: int) -> A.Compose:
    """Deterministic eval pipeline: resize + normalize + to tensor."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

REQUIRED_COLS = (
    "image_id", "src_image_path", "class_name", "class_idx", "lesion_id",
)


class CanonicalDataset(Dataset):
    """Reads a Stage-3 CSV.

    Returns dict per __getitem__:
        {
          'image': (3, H, W) float tensor (normalized)
          'meta':  (META_DIM,) float tensor
          'meta_mask': (META_DIM,) float tensor (1.0 = present)
          'label': long tensor (class index)
          'row_id': int (position in CSV; for tracebacks)
          'valid': bool (False if image read failed)
        }
    """

    def __init__(self, csv_path: str | Path, *,
                 img_size: int = 384,
                 train_mode: bool = False) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        missing = [c for c in REQUIRED_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Split CSV {self.csv_path} missing required columns: "
                f"{missing}. Required: {REQUIRED_COLS}"
            )

        self.img_size = img_size
        self.train_mode = train_mode
        self.augs = (_build_train_augs(img_size) if train_mode
                     else _build_eval_augs(img_size))

        log.info("CanonicalDataset[%s mode=%s]: %d rows, img_size=%d",
                 self.csv_path.name, "train" if train_mode else "eval",
                 len(self.df), img_size)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _dummy_sample(meta_dim: int, img_size: int) -> dict:
        return {
            "image": torch.zeros(3, img_size, img_size, dtype=torch.float32),
            "meta": torch.zeros(meta_dim, dtype=torch.float32),
            "meta_mask": torch.zeros(meta_dim, dtype=torch.float32),
            "label": torch.tensor(-1, dtype=torch.long),
            "row_id": -1,
            "valid": False,
        }

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        try:
            img = cv2.imread(row["src_image_path"], cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"cv2.imread returned None for {row['src_image_path']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transformed = self.augs(image=img)
            image = transformed["image"]
            meta, mask = encode_row_metadata(row)
            return {
                "image": image,
                "meta": torch.from_numpy(meta),
                "meta_mask": torch.from_numpy(mask),
                "label": torch.tensor(int(row["class_idx"]), dtype=torch.long),
                "row_id": idx,
                "valid": True,
            }
        except Exception as e:  # noqa: BLE001 — keep epoch alive
            log.warning("Row %d of %s failed: %s",
                        idx, self.csv_path.name, e)
            d = self._dummy_sample(META_DIM, self.img_size)
            d["row_id"] = idx
            return d


# ─────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────

def build_balanced_sampler(df: pd.DataFrame,
                            n_classes: int = len(CANONICAL_CLASSES),
                            ) -> WeightedRandomSampler:
    """Sqrt-inverse-frequency sampler.

    Less aggressive than 1/freq (which over-samples DF/VASC to the point
    of producing pathological gradient variance). sqrt softens the
    reweighting while still raising rare classes meaningfully.
    """
    counts = np.bincount(df["class_idx"].astype(int).values,
                         minlength=n_classes)
    counts = np.clip(counts, 1, None)  # avoid div-by-zero on empty classes
    class_weights = 1.0 / np.sqrt(counts.astype(np.float64))
    sample_weights = class_weights[df["class_idx"].astype(int).values]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(df),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────
# Loader factory
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SplitInfo:
    split_dir: Path
    sizes: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, dict[str, int]] = field(default_factory=dict)


def collate_safe(batch: list[dict]) -> Optional[dict]:
    """Collate that quietly drops invalid samples.

    If the entire batch is invalid (extreme edge case) returns None and
    the training loop should skip the step. Otherwise returns a normal
    batched dict.
    """
    valid = [b for b in batch if b["valid"]]
    if not valid:
        return None
    out = {
        "image": torch.stack([b["image"] for b in valid]),
        "meta": torch.stack([b["meta"] for b in valid]),
        "meta_mask": torch.stack([b["meta_mask"] for b in valid]),
        "label": torch.stack([b["label"] for b in valid]),
        "row_id": torch.tensor([b["row_id"] for b in valid], dtype=torch.long),
    }
    return out


def build_loaders(
    split_dir: str | Path,
    *,
    batch_size: int = 32,
    img_size: int = 384,
    num_workers: int = 4,
    use_balanced_sampler: bool = True,
    pin_memory: bool = True,
) -> tuple[dict[str, DataLoader], SplitInfo]:
    """Build train / val_select / val_calibrate / test loaders.

    `split_dir` must contain {train,val_select,val_calibrate,test}.csv
    produced by stage 3 (single mode).
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir does not exist: {split_dir}")

    loaders: dict[str, DataLoader] = {}
    info = SplitInfo(split_dir=split_dir)

    split_names_and_modes = [
        ("train", True),
        ("val_select", False),
        ("val_calibrate", False),
        ("test", False),
    ]

    for name, is_train in split_names_and_modes:
        csv = split_dir / f"{name}.csv"
        if not csv.exists():
            log.warning("Missing %s — skipping that loader.", csv)
            continue
        ds = CanonicalDataset(csv, img_size=img_size, train_mode=is_train)
        info.sizes[name] = len(ds)
        info.class_counts[name] = ds.df["class_name"].value_counts().to_dict()

        # Sampler only for train
        sampler = None
        shuffle = is_train
        if is_train and use_balanced_sampler:
            sampler = build_balanced_sampler(ds.df)
            shuffle = False  # mutually exclusive with sampler

        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=collate_safe,
            drop_last=is_train,  # stable BN for train; keep all eval samples
        )

    log.info("Built loaders: sizes=%s", info.sizes)
    return loaders, info


# ─────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Smoke-test the canonical dataloader against a split dir.")
    p.add_argument("--split-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(name)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")

    loaders, info = build_loaders(
        args.split_dir,
        batch_size=args.batch_size, img_size=args.img_size,
        num_workers=args.num_workers,
    )
    for name, dl in loaders.items():
        try:
            batch = next(iter(dl))
            if batch is None:
                print(f"[{name}] entire first batch was invalid — investigate")
                continue
            print(f"[{name}] image {tuple(batch['image'].shape)}  "
                  f"meta {tuple(batch['meta'].shape)}  "
                  f"mask {tuple(batch['meta_mask'].shape)}  "
                  f"label {tuple(batch['label'].shape)}")
        except Exception as e:
            print(f"[{name}] FAILED first batch: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
