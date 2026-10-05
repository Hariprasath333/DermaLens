"""
Stage 1 — Dataset selection and verification.

Does NOT download anything. Reports which curated datasets are present on
disk, prints precise download instructions for missing ones, and returns
a registry of available datasets for downstream stages.

Design priorities (in order):
  1. Efficiency  — pure metadata scan; no image I/O at this stage
  2. Quality     — typed dataclasses, structured logging, deterministic output
  3. Errorless   — every filesystem touch wrapped in try/except with context
  4. Explainable errors — error messages name the dataset, the path
     checked, and the corrective action
  5. Logic       — single source of truth for dataset metadata; verification
     decoupled from registration so a missing dataset never crashes the
     pipeline

Usage:
    from stages.stage1_datasets import verify_datasets, DATASET_REGISTRY
    available, missing = verify_datasets(data_root='/path/to/datasets')

CLI:
    python -m stages.stage1_datasets --data-root /path/to/datasets
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# Logger — module-scoped so downstream stages can inherit handlers
# ─────────────────────────────────────────────────────────────────────
log = logging.getLogger("lesioniq.stage1")


# ─────────────────────────────────────────────────────────────────────
# Dataset descriptor
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a dataset entry in the catalogue.

    Frozen so DATASET_REGISTRY is hashable and side-effect free.
    """

    key: str                          # canonical short name used by pipeline
    display_name: str                 # human-facing name
    tier: int                         # 1 = essential, 2 = high, 3 = external-validation-only
    n_images_expected: int            # for sanity check after download
    image_subdir: str                 # relative to <data_root>/<key>/
    metadata_files: tuple[str, ...]   # relative to <data_root>/<key>/
    download_instructions: str        # exact steps for the user
    license: str                      # license string for the user's reference
    use_for: str                      # one-line purpose in the thesis
    schema_fields: tuple[str, ...] = field(default=())  # metadata columns to expect

    # ---- methods ----
    def root(self, data_root: Path) -> Path:
        return data_root / self.key

    def image_dir(self, data_root: Path) -> Path:
        return self.root(data_root) / self.image_subdir

    def metadata_paths(self, data_root: Path) -> list[Path]:
        return [self.root(data_root) / m for m in self.metadata_files]


# ─────────────────────────────────────────────────────────────────────
# DATASET_REGISTRY — single source of truth
# ─────────────────────────────────────────────────────────────────────
# Notes:
#   - Paths are *suggested* layouts; if the user lays them out differently
#     they configure `pipeline.yaml` to override.
#   - n_images_expected is a sanity threshold, not a strict requirement
#     (challenges sometimes release patch versions).
# ─────────────────────────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "isic2019": DatasetSpec(
        key="isic2019",
        display_name="ISIC 2019 Challenge (train + public test)",
        tier=1,
        n_images_expected=33_569,  # 25,331 train + 8,238 test
        image_subdir="images",
        metadata_files=(
            "ISIC_2019_Training_GroundTruth.csv",
            "ISIC_2019_Training_Metadata.csv",
            "ISIC_2019_Test_GroundTruth.csv",
            "ISIC_2019_Test_Metadata.csv",
        ),
        schema_fields=("age_approx", "sex", "anatom_site_general"),
        license="CC BY-NC 4.0",
        use_for="Reference baseline; existing hackathon results",
        download_instructions=(
            "1. Go to https://challenge.isic-archive.com/data/#2019\n"
            "2. Accept the ISIC license agreement.\n"
            "3. Download the four files into <data_root>/isic2019/:\n"
            "   - ISIC_2019_Training_Input.zip      -> unpack into images/\n"
            "   - ISIC_2019_Training_GroundTruth.csv\n"
            "   - ISIC_2019_Training_Metadata.csv\n"
            "   - ISIC_2019_Test_Input.zip          -> unpack into images/\n"
            "   - ISIC_2019_Test_GroundTruth.csv\n"
            "   - ISIC_2019_Test_Metadata.csv"
        ),
    ),
    "ham10000": DatasetSpec(
        key="ham10000",
        display_name="HAM10000 (Harvard Dataverse, full release)",
        tier=1,
        n_images_expected=10_015,
        image_subdir="HAM10000_images",
        metadata_files=("HAM10000_metadata.csv",),
        schema_fields=("age", "sex", "localization", "dx_type", "lesion_id"),
        license="CC BY-NC 4.0",
        use_for="Multi-image lesion baseline; richer lesion_id structure",
        download_instructions=(
            "1. Go to https://doi.org/10.7910/DVN/DBW86T  (Harvard Dataverse)\n"
            "2. Accept the Dataverse terms.\n"
            "3. Download HAM10000_images_part_1.zip + part_2.zip + HAM10000_metadata.csv\n"
            "4. Unzip both image archives into <data_root>/ham10000/HAM10000_images/\n"
            "5. Place HAM10000_metadata.csv at <data_root>/ham10000/"
        ),
    ),
    "isic2020": DatasetSpec(
        key="isic2020",
        display_name="ISIC 2020 Challenge",
        tier=1,
        n_images_expected=33_126,
        image_subdir="train",
        metadata_files=("ISIC_2020_Training_GroundTruth.csv",),
        schema_fields=(
            "patient_id", "lesion_id", "age_approx", "sex",
            "anatom_site_general_challenge", "diagnosis", "benign_malignant",
        ),
        license="CC BY-NC 4.0",
        use_for="Largest single-source dermoscopy collection; temporal-shift study; lesion aggregation",
        download_instructions=(
            "1. Go to https://challenge.isic-archive.com/data/#2020\n"
            "2. Accept the license agreement.\n"
            "3. Download:\n"
            "   - ISIC_2020_Training_JPEG.zip (or the resized 256/384 version to save disk)\n"
            "   - ISIC_2020_Training_GroundTruth.csv\n"
            "4. Unzip images into <data_root>/isic2020/train/\n"
            "5. Place the CSV at <data_root>/isic2020/\n"
            "RECOMMENDED: use the 256x256 resized version unless you have 100+ GB free."
        ),
    ),
    "pad_ufes_20": DatasetSpec(
        key="pad_ufes_20",
        display_name="PAD-UFES-20 (Pacheco et al. 2020)",
        tier=1,
        n_images_expected=2_298,
        image_subdir="imgs",
        metadata_files=("metadata.csv",),
        schema_fields=(
            "patient_id", "lesion_id", "age", "gender", "region",
            "fitspatrick", "diameter_1", "diameter_2",
            "smoke", "drink", "background_father", "background_mother",
            "pesticide", "has_piped_water", "has_sewage_system",
            "itch", "grew", "hurt", "changed", "bleed", "elevation",
        ),
        license="CC BY 4.0",
        use_for="Richest metadata in catalogue; non-dermoscopic modality; Fitzpatrick skin tone",
        download_instructions=(
            "1. Go to https://data.mendeley.com/datasets/zr7vgbcyr2/1\n"
            "2. No special terms beyond CC BY 4.0 attribution.\n"
            "3. Download the dataset zip.\n"
            "4. Unzip such that <data_root>/pad_ufes_20/imgs/ contains the .png files\n"
            "5. Place metadata.csv at <data_root>/pad_ufes_20/"
        ),
    ),
    "fitzpatrick17k": DatasetSpec(
        key="fitzpatrick17k",
        display_name="Fitzpatrick17k (Groh et al. 2021)",
        tier=1,
        n_images_expected=16_577,
        image_subdir="images",
        metadata_files=("fitzpatrick17k.csv",),
        schema_fields=("md5hash", "fitzpatrick_scale", "fitzpatrick_centaur", "label", "nine_partition_label", "three_partition_label"),
        license="Open (research)",
        use_for="Fairness audit only — skin-tone stratified evaluation",
        download_instructions=(
            "1. Go to https://github.com/mattgroh/fitzpatrick17k\n"
            "2. Follow the README to download images (links are in the CSV).\n"
            "3. Place fitzpatrick17k.csv at <data_root>/fitzpatrick17k/\n"
            "4. Run the project's download script (NOT this pipeline) to fetch the images.\n"
            "NOTE: Some images may have rotted links. Expect 10-20%% missing.\n"
            "USED FOR EXTERNAL EVAL ONLY — never enters the training mix."
        ),
    ),
    "derm7pt": DatasetSpec(
        key="derm7pt",
        display_name="Derm7pt (Kawahara et al. 2018, optional)",
        tier=2,
        n_images_expected=2_045,
        image_subdir="images",
        metadata_files=("meta.csv",),
        schema_fields=(
            "lesion_id", "diagnosis",
            "pigment_network", "streaks", "pigmentation",
            "regression_structures", "dots_and_globules",
            "blue_whitish_veil", "vascular_structures",
        ),
        license="Research-only (registration required)",
        use_for="Multi-task: predict 7-point checklist as auxiliary head",
        download_instructions=(
            "1. Register at https://derm.cs.sfu.ca/\n"
            "2. Download the dataset archive.\n"
            "3. Place images at <data_root>/derm7pt/images/ and meta.csv at <data_root>/derm7pt/"
        ),
    ),
    "ph2": DatasetSpec(
        key="ph2",
        display_name="PH² (Mendonca et al. 2013, optional external test)",
        tier=3,
        n_images_expected=200,
        image_subdir="PH2_Dataset_images",
        metadata_files=("PH2_dataset.txt",),
        schema_fields=("clinical_diagnosis", "asymmetry", "pigment_network", "dots_globules", "streaks", "regression", "blue_whitish_veil"),
        license="Research-only",
        use_for="Small external test set; never enters training",
        download_instructions=(
            "1. Visit https://www.fc.up.pt/addi/ph2%%20database.html\n"
            "2. Request access and accept terms.\n"
            "3. Unzip into <data_root>/ph2/"
        ),
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DatasetStatus:
    spec: DatasetSpec
    present: bool
    image_count: int
    image_dir_exists: bool
    metadata_present: dict[str, bool]
    issues: list[str]

    def summary(self) -> str:
        if self.present:
            return (f"[OK]      {self.spec.key:<16s} {self.image_count:>6,} images, "
                    f"all metadata present")
        missing_meta = [name for name, ok in self.metadata_present.items() if not ok]
        msg = f"[MISSING] {self.spec.key:<16s} "
        if not self.image_dir_exists:
            msg += f"image_dir not found at {self.spec.image_subdir}/"
        elif self.image_count < self.spec.n_images_expected * 0.5:
            msg += f"only {self.image_count} images (expected ~{self.spec.n_images_expected})"
        if missing_meta:
            msg += f"  missing CSVs: {missing_meta}"
        return msg


def _count_images(image_dir: Path) -> int:
    """Cheap image-count without loading any pixels.

    Counts files ending in common image extensions, case-insensitive.
    Returns 0 on any error to keep the caller robust.
    """
    if not image_dir.exists() or not image_dir.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    try:
        # iterdir() is faster than rglob for a flat image directory
        # but ISIC 2020 nests images under train/ so we tolerate nesting
        # by checking both flat and one-level-deep.
        n_flat = sum(1 for p in image_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in exts)
        if n_flat > 0:
            return n_flat
        # fall back to one-level rglob
        return sum(1 for p in image_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in exts)
    except (OSError, PermissionError) as e:
        log.warning("Could not enumerate %s: %s", image_dir, e)
        return 0


def verify_dataset(spec: DatasetSpec, data_root: Path) -> DatasetStatus:
    """Verify a single dataset on disk. Never raises; reports issues."""
    issues: list[str] = []

    image_dir = spec.image_dir(data_root)
    image_dir_exists = image_dir.exists()
    image_count = _count_images(image_dir) if image_dir_exists else 0

    metadata_present: dict[str, bool] = {}
    for meta_path in spec.metadata_paths(data_root):
        ok = meta_path.exists() and meta_path.is_file()
        metadata_present[meta_path.name] = ok
        if not ok:
            issues.append(f"metadata file missing: {meta_path}")

    if not image_dir_exists:
        issues.append(f"image directory missing: {image_dir}")
    elif image_count == 0:
        issues.append(f"image directory empty: {image_dir}")
    elif image_count < spec.n_images_expected * 0.5:
        issues.append(
            f"image count low: {image_count} found, ~{spec.n_images_expected} expected "
            f"(more than 50% missing)"
        )

    present = (
        image_dir_exists
        and image_count >= spec.n_images_expected * 0.5
        and all(metadata_present.values())
    )

    return DatasetStatus(
        spec=spec,
        present=present,
        image_count=image_count,
        image_dir_exists=image_dir_exists,
        metadata_present=metadata_present,
        issues=issues,
    )


def verify_datasets(
    data_root: Path | str,
    only_tier: Optional[int] = None,
) -> tuple[list[DatasetStatus], list[DatasetStatus]]:
    """Verify every dataset in DATASET_REGISTRY against data_root.

    Returns (present_list, missing_list). Never raises on missing data.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        log.warning("data_root does not exist: %s (creating)", data_root)
        try:
            data_root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.error("Cannot create data_root %s: %s", data_root, e)
            return [], list(DATASET_REGISTRY.values())  # type: ignore[arg-type]

    statuses: list[DatasetStatus] = []
    for spec in DATASET_REGISTRY.values():
        if only_tier is not None and spec.tier != only_tier:
            continue
        statuses.append(verify_dataset(spec, data_root))

    present = [s for s in statuses if s.present]
    missing = [s for s in statuses if not s.present]
    return present, missing


def print_status_report(
    present: list[DatasetStatus],
    missing: list[DatasetStatus],
    data_root: Path,
    show_download_instructions: bool = True,
) -> None:
    """Pretty-print the catalogue status to stdout (not the logger)."""
    print()
    print("=" * 76)
    print(f" Dataset catalogue status (root: {data_root})")
    print("=" * 76)
    print(f" Present: {len(present):>2}    Missing: {len(missing):>2}")
    print()

    print("--- Present ---")
    if present:
        for s in present:
            print(f"  {s.summary()}")
    else:
        print("  (none yet)")

    print()
    print("--- Missing ---")
    if missing:
        for s in missing:
            print(f"  {s.summary()}")
            for issue in s.issues:
                print(f"           ! {issue}")
    else:
        print("  (all datasets present)")

    if missing and show_download_instructions:
        print()
        print("=" * 76)
        print(" Download instructions for missing datasets")
        print("=" * 76)
        for s in missing:
            print()
            print(f"### {s.spec.display_name}  [tier {s.spec.tier}]")
            print(f"    Use: {s.spec.use_for}")
            print(f"    License: {s.spec.license}")
            print()
            for line in s.spec.download_instructions.splitlines():
                print(f"    {line}")

    print()
    print("=" * 76)
    print(" Recommended training mix")
    print("=" * 76)
    tiers: dict[int, list[str]] = {}
    for s in [*present, *missing]:
        tiers.setdefault(s.spec.tier, []).append(s.spec.key)
    for tier in sorted(tiers):
        names = ", ".join(tiers[tier])
        label = {1: "ESSENTIAL", 2: "HIGH-VALUE", 3: "EXTERNAL-EVAL"}.get(tier, str(tier))
        print(f"  Tier {tier} [{label}]: {names}")
    print()


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="LesionIQ Research — Stage 1: Verify dataset catalogue."
    )
    parser.add_argument(
        "--data-root", required=True,
        help="Directory containing per-dataset subfolders.",
    )
    parser.add_argument(
        "--only-tier", type=int, default=None, choices=[1, 2, 3],
        help="Verify only the given tier.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress download instructions in the report.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root = Path(args.data_root).expanduser().resolve()
    log.info("Verifying datasets under %s", data_root)

    present, missing = verify_datasets(data_root, only_tier=args.only_tier)
    print_status_report(present, missing, data_root,
                        show_download_instructions=not args.quiet)

    # Exit code: 0 if all Tier-1 essentials present, 1 if any missing
    tier1_missing = [s for s in missing if s.spec.tier == 1]
    if tier1_missing:
        log.warning("%d Tier-1 dataset(s) missing. Pipeline cannot proceed past "
                    "Stage 2 until these are downloaded.", len(tier1_missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
