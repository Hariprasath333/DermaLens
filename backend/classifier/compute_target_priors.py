"""
LesionIQ -- Compute target priors for prior-shift adaptation.

Writes two files into backend/checkpoints/:
  target_prior_sld.npy     -- estimated from unlabeled test predictions via SLD
  target_prior_oracle.npy  -- empirical class proportions from test labels

Both are 8-vector numpy arrays summing to 1.

Reads the test set, forward-passes the trained model with current calibration
settings (per-class temperature if available, else global T, else uncalibrated),
then runs SLD against the model's effective training prior to estimate the
test prior without using labels.

Run once after running compute_effective_train_prior.py.

    $env:PYTHONPATH = "C:\\LesionIQ"
    python -m backend.classifier.compute_target_priors
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.classifier.config import BATCH_SIZE, DEVICE, USE_AMP, TEST_CSV
from backend.classifier.dataloader import LesionDataset, VAL_TRANSFORMS
from backend.classifier.inference import build_model, CKPT_DIR
from backend.classifier.prior_adaptation import (
    estimate_test_prior_sld,
    empirical_prior_from_labels,
)


CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


@torch.no_grad()
def _gather_calibrated_probs(model, loader: DataLoader,
                              global_T: float,
                              per_class_T: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass; return (probs, labels) using current calibration."""
    model.eval()
    is_fp16 = next(model.parameters()).dtype == torch.float16
    p_parts, y_parts = [], []
    for batch in loader:
        if len(batch) == 3:
            images, meta, labels = batch
        else:
            images, labels = batch
            meta = None
        images = images.to(DEVICE, non_blocking=True)
        if meta is not None:
            meta = meta.to(DEVICE, non_blocking=True)
        if is_fp16:
            images = images.half()
            if meta is not None:
                meta = meta.half()

        with torch.amp.autocast("cuda", enabled=USE_AMP) if DEVICE == "cuda" \
                else torch.autocast(device_type="cpu", enabled=False):
            out = model(images, meta) if meta is not None else model(images)
        logits = (out[0] if isinstance(out, tuple) else out).float()

        if per_class_T is not None:
            temps = torch.from_numpy(per_class_T).to(logits.device)
            logits = logits / temps
        elif global_T and global_T != 1.0:
            logits = logits / global_T

        p_parts.append(F.softmax(logits, dim=1).cpu().numpy())
        y_parts.append(labels.numpy())
    return np.concatenate(p_parts, 0), np.concatenate(y_parts, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute SLD + oracle target priors on test set.")
    ap.add_argument("--mode", default="full",
                    choices=["full", "image_only", "swin_only", "effnet_only"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    ckpt_dir = Path(CKPT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_prior_path = ckpt_dir / "effective_train_prior.npy"
    if not train_prior_path.exists():
        raise SystemExit(
            f"Missing {train_prior_path}. Run "
            "`python -m backend.classifier.compute_effective_train_prior` first."
        )
    train_prior = np.load(str(train_prior_path)).astype(np.float32)
    print(f"[OK] Loaded effective training prior  (mean={train_prior.mean():.4f})")

    # Load calibration assets exactly as inference does
    pc_path = ckpt_dir / "per_class_temperatures.npy"
    tg_path = ckpt_dir / "optimal_temperature.npy"
    per_class_T = np.load(str(pc_path)).astype(np.float32) if pc_path.exists() else None
    global_T    = float(np.load(str(tg_path))) if tg_path.exists() else 1.0
    if per_class_T is not None:
        print(f"[OK] Using per-class T (mean={per_class_T.mean():.4f})")
    else:
        print(f"[OK] Using global T={global_T:.4f}")

    print(f"[STEP] Building model ({args.mode}) ...")
    model = build_model(args.mode, args.checkpoint)

    print(f"[STEP] Loading test data: {TEST_CSV}")
    dataset = LesionDataset(TEST_CSV, image_dir="", transform=VAL_TRANSFORMS)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=(DEVICE == "cuda"))

    print(f"[STEP] Forward pass on {len(dataset)} test images ...")
    probs, labels = _gather_calibrated_probs(model, loader, global_T, per_class_T)
    print(f"[OK]   Collected: probs {probs.shape}, labels {labels.shape}")

    # SLD estimation
    sld_prior = estimate_test_prior_sld(probs, train_prior, max_iter=500)
    sld_path = ckpt_dir / "target_prior_sld.npy"
    np.save(sld_path, sld_prior.astype(np.float32))
    print(f"[OK] Saved SLD target prior   -> {sld_path}")

    # Oracle (uses labels — only valid when labels are available)
    oracle_prior = empirical_prior_from_labels(labels, n_classes=probs.shape[1])
    oracle_path = ckpt_dir / "target_prior_oracle.npy"
    np.save(oracle_path, oracle_prior.astype(np.float32))
    print(f"[OK] Saved oracle target prior -> {oracle_path}")

    # Print comparison
    print("\n  Class   Effective-train   Oracle-test   SLD-test")
    print("  -----   ---------------   -----------   --------")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<6s}  {train_prior[i]:>14.4f}   {oracle_prior[i]:>11.4f}   {sld_prior[i]:>8.4f}")

    err = np.abs(sld_prior - oracle_prior).max()
    print(f"\n  SLD vs Oracle L-inf error: {err:.4f}")
    print("  (Lower = SLD recovered the true test prior well; ~0.03 is excellent.)")


if __name__ == "__main__":
    main()
