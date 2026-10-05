#!/usr/bin/env python3
"""Download LesionIQ model checkpoints from GitHub Releases.

Usage:
    python scripts/download_checkpoints.py            # Download all checkpoints
    python scripts/download_checkpoints.py --only best_full.pt
    python scripts/download_checkpoints.py --list     # List available files

The script downloads checkpoint files into backend/checkpoints/ and
verifies their SHA-256 hashes when available.

If a GitHub Release is not yet published, the script will print setup
instructions for creating one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "Astroshreyas1/LesionIQ"
RELEASE_TAG = "v1.0-checkpoints"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "backend" / "checkpoints"

# Files expected in the release. Update SHA-256 hashes after uploading.
CHECKPOINT_MANIFEST = {
    "best_full.pt": {
        "description": "Full hybrid model (EfficientNet-B4 + SwinV2 + metadata MLP)",
        "sha256": None,  # Set after first upload
    },
    "best_image_only.pt": {
        "description": "Dual-backbone, no metadata (EfficientNet-B4 + SwinV2)",
        "sha256": None,
    },
    "best_effnet_only.pt": {
        "description": "EfficientNet-B4 only baseline",
        "sha256": None,
    },
    "best_full_swa.pt": {
        "description": "SWA-averaged full hybrid model",
        "sha256": None,
    },
    "optimal_scales.npy": {
        "description": "Clinical DiffEvo threshold scales",
        "sha256": None,
    },
    "optimal_temperature.npy": {
        "description": "LBFGS-calibrated temperature",
        "sha256": None,
    },
    "mel_safety_threshold.npy": {
        "description": "MEL recall safety threshold (0.265)",
        "sha256": None,
    },
}


def _release_url() -> str:
    return f"https://api.github.com/repos/{REPO}/releases/tags/{RELEASE_TAG}"


def _get_release_assets() -> list[dict]:
    """Fetch the asset list from a GitHub Release."""
    try:
        req = urllib.request.Request(
            _release_url(),
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data.get("assets", [])
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise


def _download_file(url: str, dest: Path, expected_sha256: str | None = None) -> None:
    """Download a file with progress display and optional hash verification."""
    print(f"  Downloading {dest.name} ...")
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/octet-stream"},
    )
    dest.parent.mkdir(parents=True, exist_ok=True)

    sha = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                sha.update(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb = downloaded / (1 << 20)
                    total_mb = total / (1 << 20)
                    print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
        print()

    if expected_sha256 and sha.hexdigest() != expected_sha256:
        print(f"  [WARN] SHA-256 mismatch for {dest.name}")
        print(f"         Expected: {expected_sha256}")
        print(f"         Got:      {sha.hexdigest()}")


def _print_setup_instructions() -> None:
    print(
        "\n"
        "══════════════════════════════════════════════════════════\n"
        "  GitHub Release not found.\n"
        "══════════════════════════════════════════════════════════\n"
        "\n"
        "  To make checkpoints downloadable, create a GitHub Release:\n"
        "\n"
        f"  1. Go to https://github.com/{REPO}/releases/new\n"
        f"  2. Tag: {RELEASE_TAG}\n"
        "  3. Title: 'Model Checkpoints'\n"
        "  4. Upload these files from backend/checkpoints/:\n"
    )
    for name, info in CHECKPOINT_MANIFEST.items():
        print(f"     - {name:30s}  ({info['description']})")
    print(
        "\n"
        "  5. Publish the release\n"
        "  6. Re-run this script to verify downloads work\n"
        "\n"
        "  Alternatively, upload checkpoints to Hugging Face or\n"
        "  Google Drive and update the URLs in this script.\n"
        "══════════════════════════════════════════════════════════\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LesionIQ model checkpoints")
    parser.add_argument(
        "--only", type=str, nargs="+",
        help="Download only these specific files",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available checkpoint files and exit",
    )
    parser.add_argument(
        "--dest", type=str, default=str(CHECKPOINT_DIR),
        help=f"Destination directory (default: {CHECKPOINT_DIR})",
    )
    args = parser.parse_args()

    if args.list:
        print("\nLesionIQ checkpoint manifest:\n")
        for name, info in CHECKPOINT_MANIFEST.items():
            status = "✓" if (CHECKPOINT_DIR / name).exists() else "✗"
            print(f"  {status} {name:30s}  {info['description']}")
        print()
        sys.exit(0)

    dest_dir = Path(args.dest)
    wanted = set(args.only) if args.only else set(CHECKPOINT_MANIFEST.keys())

    # Fetch release assets
    assets = _get_release_assets()
    if not assets:
        _print_setup_instructions()
        sys.exit(1)

    asset_map = {a["name"]: a for a in assets}
    downloaded = 0

    for name in wanted:
        if name not in CHECKPOINT_MANIFEST:
            print(f"  [SKIP] Unknown file: {name}")
            continue

        dest_file = dest_dir / name
        if dest_file.exists():
            print(f"  [SKIP] {name} already exists")
            continue

        asset = asset_map.get(name)
        if not asset:
            print(f"  [MISS] {name} not found in release assets")
            continue

        _download_file(
            asset["browser_download_url"],
            dest_file,
            CHECKPOINT_MANIFEST[name].get("sha256"),
        )
        downloaded += 1

    if downloaded:
        print(f"\n  Downloaded {downloaded} file(s) to {dest_dir}")
    else:
        print("\n  All requested checkpoints are already present.")


if __name__ == "__main__":
    main()
