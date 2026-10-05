"""
LesionIQ Hybrid Classifier — Main Entry Point
================================================
Usage:
    python train_classifier.py                          # train hybrid (default)
    python train_classifier.py --mode effnet_only       # EfficientNet-B4 only
    python train_classifier.py --mode swin_only         # Swin-Base only
    python train_classifier.py --eval-only --checkpoint path/to/best_model.pt
    python train_classifier.py --fix-swa --checkpoint path/to/best_model.pt
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel

from backend.classifier.config import (
    BATCH_SIZE, DEVICE, FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING,
    OUTPUT_DIR, SEED,
)
from backend.classifier.dataloader import LABEL_COLS, META_COLS, get_dataloaders
from backend.classifier.models import LesionIQHybrid
# train.py pulls in `wandb` at import time; skip it for eval-only / fix-swa
# runs so missing optional deps don't block evaluation.
try:
    from backend.classifier.train import FocalLoss, _validate, train  # noqa: F401
except ModuleNotFoundError as _train_import_err:
    _MISSING_TRAIN_DEPS = _train_import_err
    FocalLoss = _validate = train = None
else:
    _MISSING_TRAIN_DEPS = None
from backend.classifier.evaluate import evaluate
from backend.classifier.explainability import run_explainability

# Known training class distribution (real + synthetic)
CLASS_COUNTS = {
    0: 3640,   # MEL
    1: 10304,  # NV
    2: 2668,   # BCC
    3: 1500,   # AK
    4: 2126,   # BKL
    5: 588,    # DF
    6: 615,    # VASC
    7: 1305,   # SCC
}


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _compute_class_weights() -> np.ndarray:
    total = sum(CLASS_COUNTS.values())
    num_classes = len(CLASS_COUNTS)
    weights = np.zeros(num_classes, dtype=np.float32)
    for cls, count in CLASS_COUNTS.items():
        weights[cls] = total / (num_classes * count)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="LesionIQ Hybrid Classifier")
    parser.add_argument("--mode", choices=["effnet_only", "swin_only", "image_only", "full"],
                        default="full", help="Architecture mode (default: full)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; run evaluation + explainability only")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a saved checkpoint (.pt)")
    parser.add_argument("--skip-explain", action="store_true",
                        help="Skip the explainability suite after evaluation")
    parser.add_argument("--fix-swa", action="store_true",
                        help="Re-run SWA BN update on existing checkpoint (no retraining)")
    args = parser.parse_args()

    _set_seed()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    class_names = LABEL_COLS
    class_weights = _compute_class_weights()

    # ── Model ─────────────────────────────────────────────────
    model = LesionIQHybrid(meta_dim=len(META_COLS), mode=args.mode)

    # ── Load checkpoint (if provided) ─────────────────────────
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[CKPT] Loaded {args.checkpoint}  "
              f"(epoch={ckpt.get('epoch')}, val_f1={ckpt.get('val_f1', '?')})")

    # ── Fix SWA (re-run BN update only) ───────────────────────
    if args.fix_swa:
        if _MISSING_TRAIN_DEPS is not None:
            print(f"ERROR: --fix-swa requires the training stack "
                  f"(missing: {_MISSING_TRAIN_DEPS}). Install wandb or run "
                  f"--eval-only instead.")
            sys.exit(1)
        if not args.checkpoint:
            print("ERROR: --fix-swa requires --checkpoint path/to/best_model.pt")
            sys.exit(1)
        print("\n  Re-running SWA batch norm update with metadata...")
        swa_model = AveragedModel(model).to(DEVICE)
        swa_model.train()

        momenta = {}
        for module in swa_model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
                module.momentum = None
                module.num_batches_tracked *= 0

        with torch.no_grad():
            for images, meta, labels in train_loader:
                images = images.to(DEVICE, non_blocking=True)
                meta = meta.to(DEVICE, non_blocking=True)
                swa_model(images, meta)

        for module, momentum in momenta.items():
            module.momentum = momentum

        swa_path = str(Path(OUTPUT_DIR) / "checkpoints" / "best_model_swa.pt")
        torch.save({
            "model_state_dict": swa_model.module.state_dict(),
            "mode": args.mode,
            "swa": True,
        }, swa_path)
        print(f"  SWA checkpoint saved -> {swa_path}")

        alpha = torch.tensor(FOCAL_ALPHA, dtype=torch.float32).to(DEVICE)
        criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha,
                              label_smoothing=LABEL_SMOOTHING).to(DEVICE)
        swa_loss, swa_acc, swa_f1, swa_auc = _validate(swa_model, val_loader, criterion, DEVICE)
        print(f"  SWA model: val_f1={swa_f1:.4f}  val_auc={swa_auc:.4f}  val_acc={swa_acc:.4f}")
        print("\nDone.")
        return

    # ── Train ─────────────────────────────────────────────────
    if not args.eval_only:
        if _MISSING_TRAIN_DEPS is not None:
            print(f"ERROR: training requires the training stack "
                  f"(missing: {_MISSING_TRAIN_DEPS}). Install wandb or pass "
                  f"--eval-only.")
            sys.exit(1)
        best_path = train(model, train_loader, val_loader, class_weights)
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[CKPT] Reloaded best checkpoint for evaluation")

    model.to(DEVICE)

    # ── Evaluate ──────────────────────────────────────────────
    report = evaluate(model, test_loader, class_names)

    # ── Explainability ────────────────────────────────────────
    if not args.skip_explain:
        run_explainability(model, test_loader, class_names, META_COLS)

    # ── Save run config ───────────────────────────────────────
    run_info = {
        "mode": args.mode,
        "seed": SEED,
        "device": DEVICE,
        "class_names": class_names,
        "meta_dim": len(META_COLS),
        "meta_features": META_COLS,
    }
    info_path = Path(OUTPUT_DIR) / "run_info.json"
    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\nDone. All outputs → {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
