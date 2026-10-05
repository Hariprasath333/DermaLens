"""
Stage 2 — Preprocessing pipeline.

Reads images from any verified dataset, applies the 4-step LesionIQ
preprocessing pipeline (DullRazor -> Shades-of-Gray -> CLAHE -> Border
removal), and writes pre-processed images + a per-dataset manifest CSV
to a single canonical output tree.

Design priorities:
  1. Efficiency  — multi-process worker pool; per-image cached output;
                   skip-if-exists; minimal memory footprint (one image
                   per worker at a time).
  2. Quality     — every preprocessing step is pure (no globals); output
                   manifest records every step's runtime + checksum.
  3. Errorless   — per-image try/except; corrupt files quarantined to
                   <out>/quarantine/ with a reason file; pipeline never
                   stops on a single failure.
  4. Explainable — every error logs (dataset, image_id, stage, original
                   exception). Final summary breaks down failures by
                   stage so the user knows where to look.
  5. Logic       — output layout mirrors the canonical dataset layout
                   (<out>/<dataset_key>/images/) so Stage 3+ can
                   discover preprocessed images without extra config.

Usage (CLI):
    python -m stages.stage2_preprocess \\
        --data-root <raw datasets root> \\
        --out-root  <preprocessed root> \\
        --datasets isic2019 ham10000 \\
        --workers 4 \\
        --resize 384

Library:
    from stages.stage2_preprocess import preprocess_dataset
    n_ok, n_fail = preprocess_dataset(spec, data_root, out_root,
                                       resize=384, workers=4)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Pipeline-internal imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stages.stage1_datasets import (  # noqa: E402
    DatasetSpec, DATASET_REGISTRY, verify_dataset,
)


log = logging.getLogger("lesioniq.stage2")


# ─────────────────────────────────────────────────────────────────────
# Preprocessing primitives
# ─────────────────────────────────────────────────────────────────────
# All inputs are uint8 BGR images (OpenCV's native format) to avoid
# unnecessary conversions. Each function is pure: input -> output, no
# side effects, no globals.
# ─────────────────────────────────────────────────────────────────────


def dullrazor(img_bgr: np.ndarray,
              kernel_length: int = 17,
              threshold: int = 10,
              inpaint_radius: int = 5) -> np.ndarray:
    """Hair removal via multi-directional morphological line filters.

    Detects hairs in 4 orientations (0/45/90/135 deg), takes the union
    of detected hair masks, then inpaints with Telea's method.

    Parameters are LesionIQ-tuned defaults (see project docs).
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("dullrazor: empty image")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    hair_mask = np.zeros_like(gray)
    for angle in (0, 45, 90, 135):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_length, 1))
        if angle != 0:
            M = cv2.getRotationMatrix2D(
                (kernel_length / 2, 0.5), angle, 1.0)
            kernel = cv2.warpAffine(
                kernel.astype(np.uint8), M, (kernel_length, kernel_length)
            ).astype(np.uint8)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, m = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
        hair_mask = cv2.bitwise_or(hair_mask, m)

    # Slight dilation, then inpaint
    hair_mask = cv2.dilate(
        hair_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    inpainted = cv2.inpaint(
        img_bgr, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return inpainted


def shades_of_gray(img_bgr: np.ndarray, power: int = 4) -> np.ndarray:
    """Color constancy via the Shades-of-Gray algorithm (Finlayson 2004).

    Lower power = more conservative correction. Power=4 chosen for
    dermoscopy to preserve melanoma-relevant color variation while
    correcting dermoscope white balance.
    """
    img_f = img_bgr.astype(np.float32)
    eps = 1e-6
    means = np.power(np.mean(np.power(img_f, power), axis=(0, 1)),
                     1.0 / power) + eps
    gray = np.mean(means)
    correction = gray / means
    out = img_f * correction
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe(img_bgr: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE on the L channel of LAB color space.

    Operating in LAB keeps chromaticity untouched (vital for skin lesions
    where color carries diagnostic signal). clip_limit=2.0 chosen
    post-DullRazor to avoid amplifying inpainted-region artifacts.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    L = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)


def remove_vignette_border(img_bgr: np.ndarray,
                            crop_frac: float = 0.06) -> np.ndarray:
    """Crop a fixed fraction from each edge to remove dermoscope vignette.

    Conservative 6% crop: removes the dark circular dermoscope lens
    artefact present in many ISIC images while keeping the entire lesion
    visible. For lesions photographed with NO vignette this is a small
    margin trim and has no clinical effect.
    """
    h, w = img_bgr.shape[:2]
    dh, dw = int(h * crop_frac), int(w * crop_frac)
    return img_bgr[dh:h - dh, dw:w - dw]


def resize_long_edge(img_bgr: np.ndarray, target: int = 384) -> np.ndarray:
    """Resize so the long edge equals `target`, preserving aspect ratio."""
    h, w = img_bgr.shape[:2]
    if max(h, w) == target:
        return img_bgr
    scale = target / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)


def center_pad_to_square(img_bgr: np.ndarray, target: int = 384,
                         pad_color: int = 0) -> np.ndarray:
    """Pad to a square `target x target` canvas, centered.

    Black padding is used because the dermoscope vignette is also black,
    so the model sees a consistent "off-lesion" background.
    """
    h, w = img_bgr.shape[:2]
    if h == target and w == target:
        return img_bgr
    top = (target - h) // 2
    bottom = target - h - top
    left = (target - w) // 2
    right = target - w - left
    return cv2.copyMakeBorder(
        img_bgr, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_color, pad_color, pad_color),
    )


# ─────────────────────────────────────────────────────────────────────
# Per-image worker (runs in a subprocess)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    image_id: str
    src_path: str
    dst_path: str
    ok: bool
    failed_stage: Optional[str]
    error: Optional[str]
    elapsed_ms: float
    output_sha1: Optional[str]


def _sha1_of_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _preprocess_one(args: tuple[str, str, str, int, bool]) -> PreprocessResult:
    """Preprocess one image; return a structured result.

    The function never raises — failures are returned as a result with
    ok=False and an `error` string. This lets the worker pool drain
    cleanly even if hundreds of images fail.
    """
    image_id, src_path, dst_path, resize_to, skip_if_exists = args
    t0 = time.perf_counter()
    stage = "init"

    try:
        # Skip-if-exists is the cheap dedup that lets the user re-run
        # this stage incrementally without redoing finished work.
        if skip_if_exists and Path(dst_path).exists():
            return PreprocessResult(
                image_id=image_id, src_path=src_path, dst_path=dst_path,
                ok=True, failed_stage=None, error=None,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                output_sha1=None,
            )

        stage = "read"
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imread returned None for {src_path}")

        stage = "dullrazor"
        img = dullrazor(img)

        stage = "shades_of_gray"
        img = shades_of_gray(img)

        stage = "clahe"
        img = apply_clahe(img)

        stage = "vignette_crop"
        img = remove_vignette_border(img)

        stage = "resize"
        img = resize_long_edge(img, target=resize_to)
        img = center_pad_to_square(img, target=resize_to)

        stage = "write"
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise IOError(f"cv2.imwrite failed for {dst_path}")

        stage = "checksum"
        with open(dst_path, "rb") as fh:
            sha1 = _sha1_of_bytes(fh.read())

        return PreprocessResult(
            image_id=image_id, src_path=src_path, dst_path=dst_path,
            ok=True, failed_stage=None, error=None,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            output_sha1=sha1,
        )

    except Exception as e:  # noqa: BLE001 — intentional broad catch
        # Quarantine on read failure or anything else; this image won't
        # appear in the manifest so downstream stages skip it.
        return PreprocessResult(
            image_id=image_id, src_path=src_path, dst_path=dst_path,
            ok=False, failed_stage=stage,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            output_sha1=None,
        )


# ─────────────────────────────────────────────────────────────────────
# Per-dataset orchestration
# ─────────────────────────────────────────────────────────────────────

def _iter_dataset_images(spec: DatasetSpec, data_root: Path,
                          path_override: Optional[Path]) -> list[tuple[str, Path]]:
    """Yield (image_id, abs_path) for every image in a dataset directory.

    Honors path_override to support the user's existing layouts (e.g.
    ISIC's nested dataset/ISIC_2019_Training_Input/ISIC_2019_Training_Input/).
    """
    image_dir = path_override if path_override else spec.image_dir(data_root)
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    out: list[tuple[str, Path]] = []
    for p in sorted(image_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        # image_id == stem without the _downsampled tag (PAD/HAM10000 quirk)
        image_id = p.stem
        if image_id.endswith("_downsampled"):
            image_id = image_id[:-len("_downsampled")]
        out.append((image_id, p))
    return out


def preprocess_dataset(
    spec: DatasetSpec,
    data_root: Path,
    out_root: Path,
    *,
    path_override: Optional[Path] = None,
    resize: int = 384,
    workers: int = 4,
    skip_if_exists: bool = True,
    quarantine_failures: bool = True,
) -> tuple[int, int]:
    """Preprocess every image in a dataset.

    Returns (n_ok, n_fail).
    """
    log.info("Preprocessing dataset: %s (target %dpx, %d workers)",
             spec.key, resize, workers)

    images = _iter_dataset_images(spec, data_root, path_override)
    log.info("Found %d source images in %s",
             len(images), path_override or spec.image_dir(data_root))

    out_dir = out_root / spec.key / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / spec.key / "preprocess_manifest.csv"
    quarantine_path = out_root / spec.key / "quarantine"
    if quarantine_failures:
        quarantine_path.mkdir(parents=True, exist_ok=True)

    # Build worker args
    args_list = [
        (
            image_id,
            str(src.resolve()),
            str((out_dir / f"{image_id}.jpg").resolve()),
            resize,
            skip_if_exists,
        )
        for image_id, src in images
    ]

    # Pool execution
    t0 = time.time()
    results: list[PreprocessResult] = []
    if workers <= 1:
        for a in args_list:
            results.append(_preprocess_one(a))
    else:
        # spawn for Windows compatibility; processes are stateless
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            # imap_unordered for memory efficiency + early progress signal
            for i, r in enumerate(pool.imap_unordered(
                    _preprocess_one, args_list, chunksize=16), 1):
                results.append(r)
                if i % 500 == 0 or i == len(args_list):
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0.0
                    log.info("  %d / %d  (%.1f imgs/sec)",
                             i, len(args_list), rate)

    n_ok = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_ok

    # Failure quarantine + log
    if quarantine_failures and n_fail > 0:
        with (quarantine_path / "failures.txt").open("w", encoding="utf-8") as fh:
            fh.write(f"# {n_fail} images failed during preprocessing\n")
            fh.write(f"# columns: image_id | failed_stage | error\n")
            for r in results:
                if not r.ok:
                    fh.write(f"{r.image_id}\t{r.failed_stage}\t"
                             f"{(r.error or '').splitlines()[0]}\n")

    # Failure-stage breakdown
    stage_counts: dict[str, int] = {}
    for r in results:
        if not r.ok and r.failed_stage:
            stage_counts[r.failed_stage] = stage_counts.get(r.failed_stage, 0) + 1
    if stage_counts:
        log.warning("Failure breakdown for %s: %s", spec.key, stage_counts)

    # Manifest CSV
    with manifest_path.open("w", encoding="utf-8") as fh:
        fh.write("image_id,dst_path,ok,failed_stage,elapsed_ms,output_sha1\n")
        for r in results:
            fh.write(
                f"{r.image_id},"
                f"{r.dst_path},"
                f"{int(r.ok)},"
                f"{r.failed_stage or ''},"
                f"{r.elapsed_ms:.2f},"
                f"{r.output_sha1 or ''}\n"
            )

    elapsed = time.time() - t0
    log.info("Done %s: ok=%d fail=%d  (%.1fs total, %.1f imgs/sec)",
             spec.key, n_ok, n_fail, elapsed,
             len(results) / elapsed if elapsed > 0 else 0.0)
    return n_ok, n_fail


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="LesionIQ Research — Stage 2: preprocess images.")
    parser.add_argument("--data-root", required=True,
                        help="Root containing per-dataset raw folders.")
    parser.add_argument("--out-root", required=True,
                        help="Where to write preprocessed outputs.")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Subset of datasets to process (defaults to all present).",
    )
    parser.add_argument("--resize", type=int, default=384)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
        help="Process workers. Default = cpu_count() - 1.",
    )
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Re-preprocess images even if output exists.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    selected = args.datasets or list(DATASET_REGISTRY.keys())
    total_ok = total_fail = 0
    for key in selected:
        if key not in DATASET_REGISTRY:
            log.error("Unknown dataset key: %s (skipping)", key)
            continue
        spec = DATASET_REGISTRY[key]
        status = verify_dataset(spec, data_root)
        if not status.present:
            log.warning("Skipping %s — verification failed: %s",
                        key, status.summary())
            continue
        try:
            n_ok, n_fail = preprocess_dataset(
                spec, data_root, out_root,
                resize=args.resize,
                workers=args.workers,
                skip_if_exists=not args.no_skip_existing,
            )
            total_ok += n_ok
            total_fail += n_fail
        except Exception as e:
            log.error("Stage 2 failed for %s: %s", key, e)
            log.debug(traceback.format_exc())
            continue

    log.info("Stage 2 complete. Total ok=%d fail=%d", total_ok, total_fail)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
