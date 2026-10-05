"""
Post-processing pipeline for StyleGAN2-ADA synthetic images:
  1. Deduplicate DF and VASC synthetic folders (pHash hamming < 8)
  2. Apply unsharp masking to all VASC synthetics
  3. Re-run quality gate on sharpened VASC
  4. Generate final dataset manifest CSV
"""
import os
import sys
import csv
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

try:
    import imagehash
except ImportError:
    print("ERROR: pip install imagehash"); sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("postprocess")

SYNTH_ROOT   = Path(r"path/to/synthetic")
REAL_ROOT    = Path(r"path/to/output")
REPORT_DIR   = Path(r"path/to/quality_reports")
MANIFEST_CSV = Path(r"path/to/dataset_manifest.csv")

PHASH_THRESHOLD = 8
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in SUPPORTED_EXT)


# ═══════════════════════════════════════════════════════════════════
#  STEP 1: Deduplicate DF and VASC
# ═══════════════════════════════════════════════════════════════════

def deduplicate(class_name):
    folder = SYNTH_ROOT / class_name
    files = list_images(folder)
    log.info("[DEDUP %s] Hashing %d images ...", class_name, len(files))

    hashes = []
    for fp in tqdm(files, desc=f"Hash {class_name}", unit="img"):
        img = Image.open(str(fp)).convert("RGB")
        h = imagehash.phash(img)
        hashes.append((fp, h))

    # Find which files to remove (keep the lower seed)
    remove_set = set()
    for i in range(len(hashes)):
        if hashes[i][0] in remove_set:
            continue
        for j in range(i + 1, len(hashes)):
            if hashes[j][0] in remove_set:
                continue
            dist = hashes[i][1] - hashes[j][1]
            if dist < PHASH_THRESHOLD:
                remove_set.add(hashes[j][0])  # keep lower index, remove higher

    if remove_set:
        dup_dir = folder / "_duplicates"
        dup_dir.mkdir(exist_ok=True)
        for fp in remove_set:
            shutil.move(str(fp), str(dup_dir / fp.name))
        log.info("[DEDUP %s] Moved %d duplicates to %s", class_name, len(remove_set), dup_dir)
    else:
        log.info("[DEDUP %s] No duplicates found.", class_name)

    remaining = len(list_images(folder))
    log.info("[DEDUP %s] %d unique images remaining.", class_name, remaining)
    return remaining


# ═══════════════════════════════════════════════════════════════════
#  STEP 2: Unsharp masking for VASC
# ═══════════════════════════════════════════════════════════════════

def unsharp_mask_folder(class_name, sigma=1.0, strength=0.5):
    """Apply unsharp masking: sharpened = original + strength * (original - blurred)"""
    folder = SYNTH_ROOT / class_name
    files = list_images(folder)
    log.info("[SHARPEN %s] Applying unsharp mask (sigma=%.1f, strength=%.1f) to %d images ...",
             class_name, sigma, strength, len(files))

    for fp in tqdm(files, desc=f"Sharpen {class_name}", unit="img"):
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        blurred = cv2.GaussianBlur(bgr, (0, 0), sigma)
        sharpened = cv2.addWeighted(bgr, 1.0 + strength, blurred, -strength, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        cv2.imwrite(str(fp), sharpened)

    log.info("[SHARPEN %s] Done. %d images sharpened in-place.", class_name, len(files))


# ═══════════════════════════════════════════════════════════════════
#  STEP 3: Re-run quality gate on VASC
# ═══════════════════════════════════════════════════════════════════

def img_sharpness(bgr):
    return cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def img_brightness(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()

def img_saturation(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean()

def img_rgb_means(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    return rgb[:,:,0].mean(), rgb[:,:,1].mean(), rgb[:,:,2].mean()


def rerun_quality_gate(class_name):
    """Re-run the per-image PASS/FAIL grading and update reports."""
    real_dir = REAL_ROOT / class_name
    synth_dir = SYNTH_ROOT / class_name
    real_files = list_images(real_dir)
    synth_files = list_images(synth_dir)

    log.info("[QUALITY %s] Real: %d  Synth: %d", class_name, len(real_files), len(synth_files))

    # Compute real stats
    real_records = []
    for fp in tqdm(real_files, desc=f"Real {class_name}", unit="img"):
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None: continue
        r, g, b = img_rgb_means(bgr)
        real_records.append({"sharpness": img_sharpness(bgr), "brightness": img_brightness(bgr),
                             "saturation": img_saturation(bgr)})

    real_summary = {}
    for key in ["sharpness", "brightness", "saturation"]:
        vals = np.array([r[key] for r in real_records])
        real_summary[key] = {"mean": float(vals.mean()), "std": float(vals.std())}

    # Compute synth stats and grade
    graded = []
    for fp in tqdm(synth_files, desc=f"Synth {class_name}", unit="img"):
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None: continue
        r, g, b = img_rgb_means(bgr)
        rec = {"file": fp.name, "sharpness": img_sharpness(bgr), "brightness": img_brightness(bgr),
               "saturation": img_saturation(bgr), "r_mean": r, "g_mean": g, "b_mean": b}
        reasons = []
        for key in ["sharpness", "brightness", "saturation"]:
            rs = real_summary[key]
            lo = rs["mean"] - 2 * rs["std"]
            hi = rs["mean"] + (3 if key == "sharpness" else 2) * rs["std"]
            if rec[key] < lo or rec[key] > hi:
                reasons.append(f"{key}={rec[key]:.1f} outside [{lo:.1f}, {hi:.1f}]")
        rec["grade"] = "PASS" if not reasons else "FAIL"
        rec["fail_reasons"] = "; ".join(reasons)
        graded.append(rec)

    pass_count = sum(1 for g in graded if g["grade"] == "PASS")
    fail_count = sum(1 for g in graded if g["grade"] == "FAIL")
    pass_rate = round(pass_count / len(graded) * 100, 2) if graded else 0

    log.info("[QUALITY %s] %d PASS / %d FAIL (%.1f%%)", class_name, pass_count, fail_count, pass_rate)

    # Write updated CSV
    csv_path = REPORT_DIR / f"quality_{class_name}_per_image_v2.csv"
    fields = ["file", "grade", "sharpness", "brightness", "saturation", "r_mean", "g_mean", "b_mean", "fail_reasons"]
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(graded)

    # Compute synth summary stats
    synth_summary = {}
    for key in ["sharpness", "brightness", "saturation"]:
        vals = np.array([g[key] for g in graded])
        synth_summary[key] = {"mean": float(vals.mean()), "std": float(vals.std())}

    log.info("[QUALITY %s] Sharpness: Real=%.1f Synth=%.1f (was 73.8 before sharpening)",
             class_name, real_summary["sharpness"]["mean"], synth_summary["sharpness"]["mean"])

    return {"pass_count": pass_count, "fail_count": fail_count, "pass_rate_pct": pass_rate,
            "synth_count": len(synth_files), "sharpness": synth_summary["sharpness"]}


# ═══════════════════════════════════════════════════════════════════
#  STEP 4: Dataset manifest CSV
# ═══════════════════════════════════════════════════════════════════

def generate_manifest():
    """Create final dataset manifest CSV with exact counts per class."""
    log.info("")
    log.info("[MANIFEST] Generating dataset manifest ...")

    rows = []
    for cls in ["AK", "SCC", "DF", "VASC"]:
        real_count = len(list_images(REAL_ROOT / cls))
        synth_count = len(list_images(SYNTH_ROOT / cls))
        total = real_count + synth_count

        # Read quality report if available
        json_path = REPORT_DIR / f"quality_{cls}_summary.json"
        fid = None
        pass_rate = None
        if json_path.exists():
            with open(str(json_path)) as f:
                rpt = json.load(f)
                fid = rpt.get("fid")
                pass_rate = rpt.get("pass_rate_pct")

        rows.append({
            "class": cls,
            "real_images": real_count,
            "synthetic_images": synth_count,
            "total_images": total,
            "synth_ratio": round(synth_count / real_count, 2) if real_count else 0,
            "fid": round(fid, 2) if fid else "N/A",
            "pass_rate_pct": pass_rate if pass_rate else "N/A",
            "gan_model": "StyleGAN2-ADA",
            "resolution": "512x512",
            "truncation_psi": 0.85,
            "generated_date": datetime.now().strftime("%Y-%m-%d"),
        })

    with open(str(MANIFEST_CSV), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    log.info("[MANIFEST] Saved to: %s", MANIFEST_CSV)
    log.info("")
    log.info("%-6s  %6s  %6s  %6s  %8s  %6s  %8s", "Class", "Real", "Synth", "Total", "Ratio", "FID", "PassRate")
    log.info("-" * 60)
    for r in rows:
        log.info("%-6s  %6d  %6d  %6d  %8s  %6s  %8s",
                 r["class"], r["real_images"], r["synthetic_images"], r["total_images"],
                 r["synth_ratio"], r["fid"], r["pass_rate_pct"])

    return rows


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 65)
    log.info("  Post-processing Pipeline")
    log.info("=" * 65)
    log.info("")

    # Step 1: Dedup DF and VASC
    log.info("--- STEP 1: Deduplication ---")
    df_remaining = deduplicate("DF")
    vasc_remaining = deduplicate("VASC")
    log.info("")

    # Step 2: Sharpen VASC
    log.info("--- STEP 2: Unsharp Masking (VASC) ---")
    unsharp_mask_folder("VASC", sigma=1.0, strength=0.5)
    log.info("")

    # Step 3: Re-run quality gate on VASC
    log.info("--- STEP 3: Quality Re-check (VASC) ---")
    vasc_result = rerun_quality_gate("VASC")
    log.info("")

    # Step 4: Dataset manifest
    log.info("--- STEP 4: Dataset Manifest ---")
    manifest = generate_manifest()

    log.info("")
    log.info("=" * 65)
    log.info("  POST-PROCESSING COMPLETE")
    log.info("    DF:   %d unique images (removed duplicates)", df_remaining)
    log.info("    VASC: %d unique images (deduped + sharpened)", vasc_remaining)
    log.info("    VASC pass rate: %.1f%%", vasc_result["pass_rate_pct"])
    log.info("    Manifest: %s", MANIFEST_CSV)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
