"""
LesionIQ -- Compute the model's effective training prior.

Forward-passes the training set through the full model and writes the
mean softmax output to ``backend/checkpoints/effective_train_prior.npy``.
This is the prior the prior-shift adapter compares against; it accounts
for the WeightedRandomSampler effect during training (which makes the
model's prior much closer to uniform than the raw class counts).

Run once after training (or any time after a checkpoint changes):

    $env:PYTHONPATH = "C:\\LesionIQ"
    python -m backend.classifier.compute_effective_train_prior
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.classifier.config import BATCH_SIZE, DEVICE, USE_AMP
from backend.classifier.dataloader import LesionDataset, VAL_TRANSFORMS
from backend.classifier.config import TRAIN_CSV
from backend.classifier.inference import build_model, CKPT_DIR
from backend.classifier.prior_adaptation import effective_train_prior


@torch.no_grad()
def _gather_softmax(model, loader: DataLoader) -> np.ndarray:
    """Forward pass the entire loader and return (N, K) softmax probs."""
    model.eval()
    is_fp16 = next(model.parameters()).dtype == torch.float16
    parts = []
    for batch in loader:
        if len(batch) == 3:
            images, meta, _ = batch
        else:
            images, _ = batch
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
        logits = out[0] if isinstance(out, tuple) else out
        parts.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute effective training prior.")
    ap.add_argument("--mode", default="full",
                    choices=["full", "image_only", "swin_only", "effnet_only"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--output", default=str(Path(CKPT_DIR) / "effective_train_prior.npy"))
    args = ap.parse_args()

    print(f"[STEP] Building model ({args.mode}) ...")
    model = build_model(args.mode, args.checkpoint)

    # Use VAL_TRANSFORMS (deterministic) on the *training* CSV so the
    # softmax averaging reflects the data distribution rather than the
    # augmentation distribution.
    print(f"[STEP] Loading training data (no augmentation): {TRAIN_CSV}")
    dataset = LesionDataset(TRAIN_CSV, image_dir="", transform=VAL_TRANSFORMS)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers,
                         pin_memory=(DEVICE == "cuda"))

    print(f"[STEP] Forward pass on {len(dataset)} training images ...")
    probs = _gather_softmax(model, loader)
    print(f"[OK]   Collected softmax outputs: shape={probs.shape}")

    prior = effective_train_prior(probs)
    np.save(args.output, prior)

    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    print("\n[OK] Saved effective training prior ->", args.output)
    print("\n  Class   Effective prior")
    print("  -----   ---------------")
    for c, p in zip(class_names, prior):
        bar = "#" * int(round(p * 100))
        print(f"  {c:<6s}  {p:.4f}  {bar}")


if __name__ == "__main__":
    main()
