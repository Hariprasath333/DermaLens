"""One-shot: rerun ONLY per-class temperature calibration and save the file.

Use this when post_training.py succeeded on global T but crashed before
writing per_class_temperatures.npy. Single forward pass over val set,
LBFGS fit, save.
"""
import os
from pathlib import Path
import numpy as np

# Ensure ascii-only output before anything else prints
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from backend.classifier.post_training import (
    CKPT_DIR, calibrate_per_class_temperature, load_model,
)
from backend.classifier.dataloader import get_dataloaders
from backend.classifier.config import BATCH_SIZE


def main():
    print("[STEP] Loading val loader...")
    _, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    print("[STEP] Loading 'full' checkpoint...")
    model = load_model("full")

    print("[STEP] Running per-class temperature calibration...")
    pc_temps = calibrate_per_class_temperature(model, val_loader)

    out_path = Path(CKPT_DIR) / "per_class_temperatures.npy"
    np.save(out_path, pc_temps.astype(np.float32))
    print(f"\n[OK] Saved per-class temperatures -> {out_path}")
    print(f"     values: {np.round(pc_temps, 4).tolist()}")


if __name__ == "__main__":
    main()
