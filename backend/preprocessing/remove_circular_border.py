"""
LesionIQ -- Batch circular border removal for dermoscopy images.

Iterates every image in a folder. Images with a circular black
background are detected, cropped to the largest inscribed square, and resized.
Images without a circular border are resized and copied through unchanged.
ALL images end up in the output folder regardless.

Usage:
    # Training set (per-class subfolders)
    python remove_circular_border.py --input path/to/NV --output path/to/processed/NV

    # Validation set (flat folder + CSV update)
    python remove_circular_border.py --input path/to/val_images --output path/to/val_processed --csv path/to/layer0_val.csv

    # Test set
    python remove_circular_border.py --input path/to/test_images --output path/to/test_processed --csv path/to/layer0_test.csv
"""

import os
import sys
import csv
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


TARGET_SIZE  = 512       # output resolution (square)
SAVE_DEBUG   = False     # set True to save overlay images in OUTPUT_DIR/_debug/

# ──────────────────────────────────────────────────────────────

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("border_removal")


# ─────────────────────────── detection ───────────────────────────

def has_circular_border(img_bgr: np.ndarray, min_radius_frac: float = 0.25):
    """
    Check whether the image has a circular black border (dermoscope vignette).

    Returns (cx, cy, radius) if a convincing circle is found, else None.

    Steps:
      1. Grayscale + heavy blur to suppress lesion texture.
      2. Low binary threshold (pixel < 30 => border).
      3. Morphological close to seal small gaps.
      4. Largest contour -> minimum enclosing circle.
      5. Reject if circle is too small (artefact) or covers >95% of the
         image (full-frame, no real border).
    """
    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)
    min_radius = int(min_dim * min_radius_frac)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    cx, cy, radius = int(cx), int(cy), int(radius)

    if radius < min_radius:
        return None

    if radius * 2 > min_dim * 0.95:
        return None

    return (cx, cy, radius)


# ─────────────────────────── geometry ────────────────────────────

def inscribed_square(cx, cy, radius, img_h, img_w):
    """Largest axis-aligned square inside the circle, clamped to image bounds."""
    half_side = int(radius / np.sqrt(2))

    x1 = max(cx - half_side, 0)
    y1 = max(cy - half_side, 0)
    x2 = min(cx + half_side, img_w)
    y2 = min(cy + half_side, img_h)

    side = min(x2 - x1, y2 - y1)
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img_w)
    y2 = min(y2, img_h)

    return x1, y1, x2, y2


# ──────────────────────── per-image logic ────────────────────────

def process_image(src_path, dst_path, target_size, debug_dir=None):
    """
    Load one image, check for circular border, crop if present, resize, save.
    Non-border images are resized and saved as-is.

    Returns "cropped" | "copied" | "error".
    """
    try:
        img_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            log.warning("Could not read: %s", src_path.name)
            return "error"

        circle = has_circular_border(img_bgr)

        if circle is not None:
            cx, cy, radius = circle
            h, w = img_bgr.shape[:2]
            x1, y1, x2, y2 = inscribed_square(cx, cy, radius, h, w)

            if debug_dir is not None:
                overlay = img_bgr.copy()
                cv2.circle(overlay, (cx, cy), radius, (0, 255, 0), 3)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    overlay,
                    f"r={radius} crop={x2 - x1}x{y2 - y1}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )
                cv2.imwrite(str(debug_dir / f"debug_{src_path.stem}.png"), overlay)

            cropped = img_bgr[y1:y2, x1:x2]
            pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            pil = pil.resize((target_size, target_size), Image.LANCZOS)
            pil.save(str(dst_path), "PNG")
            return "cropped"

        else:
            pil = Image.open(str(src_path)).convert("RGB")
            pil = pil.resize((target_size, target_size), Image.LANCZOS)
            pil.save(str(dst_path), "PNG")
            return "copied"

    except Exception as e:
        log.error("Error on %s: %s", src_path.name, e)
        return "error"


# ──────────────────────── main loop ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch circular border removal for dermoscopy images"
    )
    parser.add_argument("--input", required=True, help="Input folder with source images")
    parser.add_argument("--output", required=True, help="Output folder for processed images")
    parser.add_argument("--csv", default=None,
                        help="Optional: CSV file to update image_path column after processing")
    parser.add_argument("--size", type=int, default=TARGET_SIZE,
                        help=f"Output resolution (square, default: {TARGET_SIZE})")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXT
    )

    if not image_files:
        log.error("No images found in %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if SAVE_DEBUG:
        debug_dir = output_dir / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    stats = {"cropped": 0, "copied": 0, "error": 0}

    log.info("Input:  %s  (%d images)", input_dir, len(image_files))
    log.info("Output: %s", output_dir)
    log.info("Target: %dx%d PNG", args.size, args.size)
    log.info("")

    for src_path in tqdm(image_files, desc=input_dir.name, unit="img"):
        dst_path = output_dir / f"{src_path.stem}.png"
        status = process_image(src_path, dst_path, args.size, debug_dir)
        stats[status] += 1

    log.info("")
    log.info("Done — %d cropped (had border), %d copied (no border), %d errors",
             stats["cropped"], stats["copied"], stats["error"])
    log.info("All %d images written to %s", stats["cropped"] + stats["copied"], output_dir)

    # Optionally update CSV image_path column
    if args.csv:
        log.info("Updating image_path in %s ...", args.csv)
        rows = []
        with open(args.csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            for row in reader:
                img_id = row['image']
                row['image_path'] = str(output_dir / f"{img_id}.png")
                rows.append(row)

        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        log.info("Updated %d rows.", len(rows))


if __name__ == "__main__":
    main()
