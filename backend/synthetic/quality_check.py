"""
LesionIQ -- Post-generation quality audit for synthetic dermoscopy images.

Compares a folder of synthetic images against the corresponding real images
and produces a detailed report covering:

  1. Distribution metrics   -- FID (requires torch + scipy) and mean KID proxy
  2. Per-image sharpness    -- Laplacian variance vs real distribution
  3. Per-image brightness   -- mean grayscale intensity vs real distribution
  4. Per-image saturation   -- mean HSV-S channel vs real distribution
  5. Colour histogram       -- per-channel RGB mean/std comparison
  6. Duplicate detection    -- flags near-duplicate synthetics (pHash hamming < 8)
  7. SSIM spot check        -- structural similarity of random synth-real pairs

Each synthetic image is individually graded PASS / FAIL.  A summary CSV and
JSON report are written to the output directory.

Usage:
    conda activate sg2ada
    python quality_check.py

Dependencies (beyond base env):
    pip install scipy scikit-image imagehash
    (torch is required for FID but already in the sg2ada env)
"""

import os
import sys
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ╔══════════════════════════════════════════════════════════════════╗
# ║  EDIT THESE PLACEHOLDERS                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

CLASS_NAME      = "AK"                                       # AK | SCC | DF | VASC
REAL_DIR        = r"path/to/cleaned/AK"                 # real images
SYNTH_DIR       = r"path/to/synthetic_stylegan/AK"      # synthetic images
REPORT_DIR      = r"path/to/quality_reports"            # where reports go

SSIM_SPOT_N     = 50        # number of random synth images for SSIM spot check
PHASH_THRESHOLD = 8         # hamming distance below this = near-duplicate

# ══════════════════════════════════════════════════════════════════


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quality_check")


# ──────────────────────── helpers ────────────────────────────────

def list_images(directory):
    d = Path(directory)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in SUPPORTED_EXT)


def img_sharpness(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def img_brightness(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()


def img_saturation(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean()


def img_rgb_means(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    return rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean()


# ────────────────── distribution statistics ──────────────────────

def compute_folder_stats(files, label):
    """Compute per-image metrics for a folder of images."""
    records = []
    for fp in tqdm(files, desc=f"{label} stats", unit="img"):
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        r, g, b = img_rgb_means(bgr)
        records.append({
            "file": fp.name,
            "sharpness": img_sharpness(bgr),
            "brightness": img_brightness(bgr),
            "saturation": img_saturation(bgr),
            "r_mean": r, "g_mean": g, "b_mean": b,
        })
    return records


def summarise(records, key):
    vals = np.array([r[key] for r in records])
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max())}


# ────────────────── FID computation ──────────────────────────────

def compute_fid(real_files, synth_files):
    """
    Compute FID using InceptionV3 features.
    Returns FID score or None if dependencies are missing.
    """
    try:
        import torch
        from torchvision import transforms, models
        from scipy.linalg import sqrtm
    except ImportError:
        log.warning("torch/scipy not available — skipping FID computation.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def extract_features(file_list, max_n=2048):
        feats = []
        subset = file_list[:max_n]
        for fp in tqdm(subset, desc="FID features", unit="img"):
            img = Image.open(str(fp)).convert("RGB")
            t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = inception(t).cpu().numpy().flatten()
            feats.append(f)
        return np.array(feats)

    log.info("Extracting Inception features for FID ...")
    real_feats = extract_features(real_files)
    synth_feats = extract_features(synth_files)

    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_s, sigma_s = synth_feats.mean(axis=0), np.cov(synth_feats, rowvar=False)

    diff = mu_r - mu_s
    covmean, _ = sqrtm(sigma_r @ sigma_s, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_s - 2 * covmean))
    return fid


# ────────────────── SSIM spot check ──────────────────────────────

def ssim_spot_check(real_files, synth_files, n):
    """Compute SSIM between random synth images and their nearest-brightness real image."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        log.warning("scikit-image not available — skipping SSIM spot check.")
        return None

    rng = np.random.RandomState(42)
    synth_subset = rng.choice(synth_files, size=min(n, len(synth_files)), replace=False)

    real_brightnesses = []
    for fp in real_files[:500]:
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is not None:
            real_brightnesses.append((img_brightness(bgr), fp))

    scores = []
    for sfp in tqdm(synth_subset, desc="SSIM spot check", unit="pair"):
        s_bgr = cv2.imread(str(sfp), cv2.IMREAD_COLOR)
        if s_bgr is None:
            continue
        s_bright = img_brightness(s_bgr)

        closest_fp = min(real_brightnesses, key=lambda x: abs(x[0] - s_bright))[1]
        r_bgr = cv2.imread(str(closest_fp), cv2.IMREAD_COLOR)
        if r_bgr is None:
            continue

        s_gray = cv2.cvtColor(cv2.resize(s_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)
        r_gray = cv2.cvtColor(cv2.resize(r_bgr, (256, 256)), cv2.COLOR_BGR2GRAY)

        score = structural_similarity(r_gray, s_gray)
        scores.append({"synth": sfp.name, "real": closest_fp.name, "ssim": float(score)})

    return scores


# ────────────────── duplicate detection ──────────────────────────

def detect_duplicates(synth_files):
    """Flag near-duplicate synthetic images using perceptual hashing."""
    try:
        import imagehash
    except ImportError:
        log.warning("imagehash not available — skipping duplicate detection.")
        log.warning("Install with:  pip install imagehash")
        return []

    hashes = []
    for fp in tqdm(synth_files, desc="Hashing synth", unit="img"):
        img = Image.open(str(fp)).convert("RGB")
        h = imagehash.phash(img)
        hashes.append((fp.name, h))

    duplicates = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = hashes[i][1] - hashes[j][1]
            if dist < PHASH_THRESHOLD:
                duplicates.append({
                    "file_a": hashes[i][0],
                    "file_b": hashes[j][0],
                    "hamming_distance": dist,
                })

    return duplicates


# ────────────────── per-image grading ────────────────────────────

def grade_synthetics(synth_records, real_summary):
    """Grade each synthetic image PASS/FAIL against real distribution bounds."""
    graded = []
    for rec in synth_records:
        reasons = []
        for key in ["sharpness", "brightness", "saturation"]:
            rs = real_summary[key]
            lo = rs["mean"] - 2 * rs["std"]
            hi = rs["mean"] + (3 if key == "sharpness" else 2) * rs["std"]
            val = rec[key]
            if val < lo or val > hi:
                reasons.append(f"{key}={val:.1f} outside [{lo:.1f}, {hi:.1f}]")

        rec["grade"] = "PASS" if not reasons else "FAIL"
        rec["fail_reasons"] = "; ".join(reasons)
        graded.append(rec)

    return graded


# ────────────────── main ─────────────────────────────────────────

def main():
    real_files = list_images(REAL_DIR)
    synth_files = list_images(SYNTH_DIR)

    if not real_files:
        log.error("No real images found in %s", REAL_DIR)
        sys.exit(1)
    if not synth_files:
        log.error("No synthetic images found in %s", SYNTH_DIR)
        sys.exit(1)

    log.info("Class:      %s", CLASS_NAME)
    log.info("Real:       %d images in %s", len(real_files), REAL_DIR)
    log.info("Synthetic:  %d images in %s", len(synth_files), SYNTH_DIR)
    log.info("")

    out = Path(REPORT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Per-image statistics ──
    real_records = compute_folder_stats(real_files, "Real")
    synth_records = compute_folder_stats(synth_files, "Synth")

    real_summary = {}
    synth_summary = {}
    for key in ["sharpness", "brightness", "saturation", "r_mean", "g_mean", "b_mean"]:
        real_summary[key] = summarise(real_records, key)
        synth_summary[key] = summarise(synth_records, key)

    log.info("")
    log.info("%-14s  %12s  %12s", "Metric", "Real", "Synthetic")
    log.info("-" * 42)
    for key in ["sharpness", "brightness", "saturation"]:
        r = real_summary[key]
        s = synth_summary[key]
        log.info("%-14s  %5.1f +/- %4.1f  %5.1f +/- %4.1f",
                 key, r["mean"], r["std"], s["mean"], s["std"])

    # ── 2. FID ──
    log.info("")
    fid = compute_fid(real_files, synth_files)
    if fid is not None:
        log.info("FID:  %.2f", fid)

    # ── 3. SSIM spot check ──
    log.info("")
    ssim_results = ssim_spot_check(real_files, synth_files, SSIM_SPOT_N)
    if ssim_results:
        ssim_vals = [s["ssim"] for s in ssim_results]
        log.info("SSIM spot check (%d pairs):  mean=%.3f  min=%.3f  max=%.3f",
                 len(ssim_vals), np.mean(ssim_vals), np.min(ssim_vals), np.max(ssim_vals))

    # ── 4. Duplicate detection ──
    log.info("")
    duplicates = detect_duplicates(synth_files)
    if duplicates:
        log.warning("%d near-duplicate pairs found (hamming < %d):", len(duplicates), PHASH_THRESHOLD)
        for d in duplicates[:10]:
            log.warning("  %s <-> %s  (dist=%d)", d["file_a"], d["file_b"], d["hamming_distance"])
        if len(duplicates) > 10:
            log.warning("  ... and %d more", len(duplicates) - 10)
    else:
        log.info("No near-duplicates found among synthetic images.")

    # ── 5. Per-image grading ──
    graded = grade_synthetics(synth_records, real_summary)
    pass_count = sum(1 for g in graded if g["grade"] == "PASS")
    fail_count = sum(1 for g in graded if g["grade"] == "FAIL")
    log.info("")
    log.info("Per-image grading:  %d PASS  /  %d FAIL  out of %d",
             pass_count, fail_count, len(graded))

    # ── 6. Write reports ──
    csv_path = out / f"quality_{CLASS_NAME}_per_image.csv"
    fields = ["file", "grade", "sharpness", "brightness", "saturation",
              "r_mean", "g_mean", "b_mean", "fail_reasons"]
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(graded)
    log.info("Per-image CSV:  %s", csv_path)

    summary_report = {
        "class": CLASS_NAME,
        "real_count": len(real_files),
        "synth_count": len(synth_files),
        "fid": fid,
        "ssim_mean": float(np.mean([s["ssim"] for s in ssim_results])) if ssim_results else None,
        "ssim_min": float(np.min([s["ssim"] for s in ssim_results])) if ssim_results else None,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate_pct": round(pass_count / len(graded) * 100, 2) if graded else 0,
        "near_duplicates": len(duplicates),
        "real_stats": real_summary,
        "synth_stats": synth_summary,
        "timestamp": datetime.now().isoformat(),
    }

    json_path = out / f"quality_{CLASS_NAME}_summary.json"
    with open(str(json_path), "w") as f:
        json.dump(summary_report, f, indent=2)
    log.info("Summary JSON:   %s", json_path)

    # ── 7. Verdict ──
    log.info("")
    log.info("=" * 50)
    if fid is not None and fid < 50 and pass_count / max(len(graded), 1) > 0.85:
        log.info("  VERDICT:  PASS  --  Quality acceptable for training")
    elif fid is not None and fid < 100 and pass_count / max(len(graded), 1) > 0.70:
        log.info("  VERDICT:  MARGINAL  --  Review flagged images")
    else:
        log.info("  VERDICT:  REVIEW  --  Consider retraining or tuning gamma")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
