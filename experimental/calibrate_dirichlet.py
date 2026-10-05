"""
LesionIQ — Dirichlet Calibration
================================

Replaces the single-scalar temperature scaling (T=0.75) with a full Dirichlet
calibration. A scalar T sharpens *all* class distributions uniformly, which
hurts the rare classes (SCC, DF, VASC) — they get sharpened toward overconfident
wrong predictions because LBFGS optimization on imbalanced val data is dominated
by NV/MEL gradients.

Dirichlet calibration fits a full K×K affine transformation on the val logits:

    z'_cal = W · log(softmax(z)) + b           (W: KxK, b: K)
    p_cal  = softmax(z'_cal)

This captures:
  • per-class temperature  (diagonal of W)
  • cross-class correlations  (off-diagonal — fixes structured SCC→BCC confusion)
  • per-class bias  (b)

References
----------
  Kull et al., "Beyond temperature scaling: Obtaining well-calibrated
  multiclass probabilities with Dirichlet calibration", NeurIPS 2019.

Usage
-----
  python -m backend.classifier.calibrate_dirichlet \\
      --checkpoint backend/checkpoints/best_full.pt \\
      --mode full

  This writes:
    backend/checkpoints/dirichlet_cal.npz   ← (W, b) for inference loading
    backend/checkpoints/val_logits.npz      ← raw logits cache (for ECE plots)
    backend/output/reports/calibration_report.json

After running, edit backend/.env (or environment) to set:
    LESIONIQ_USE_DIRICHLET=1

so the inference pipeline picks up the new calibration in place of
optimal_temperature.npy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.classifier.config import DEVICE, OUTPUT_DIR, NUM_CLASSES
from backend.classifier.dataloader import get_dataloaders
from backend.classifier.inference import build_model

# Resolved checkpoint directory (mirrors inference.py)
from backend.classifier.inference import CKPT_DIR
import importlib  # noqa: F401  (kept for symmetry with inference imports)


# ---------------------------------------------------------------------------
#  Logit collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def gather_logits(model, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Run val/test set forward pass once and return (logits, labels).

    Returns
    -------
    logits : (N, K) float32
    labels : (N,) int64
    """
    model.eval()
    all_logits, all_labels = [], []

    is_fp16 = next(model.parameters()).dtype == torch.float16

    for batch in loader:
        # LesionDataset returns (image, meta, label)
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

        logits = model(images, meta) if meta is not None else model(images)

        # Some training-mode models return (logits, aux) — take primary head
        if isinstance(logits, tuple):
            logits = logits[0]

        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_logits, 0), np.concatenate(all_labels, 0)


# ---------------------------------------------------------------------------
#  Expected Calibration Error (ECE) + per-class ECE
# ---------------------------------------------------------------------------

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 15) -> float:
    """Standard ECE: bin by max-prob confidence, compare confidence vs accuracy."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else \
               (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def per_class_ece(probs: np.ndarray, labels: np.ndarray,
                  n_classes: int = NUM_CLASSES, n_bins: int = 15) -> dict:
    """Per-class ECE: bin each class's predicted probability against its one-vs-rest accuracy."""
    out = {}
    for c in range(n_classes):
        p_c = probs[:, c]
        y_c = (labels == c).astype(np.float32)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece_c, n = 0.0, len(labels)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (p_c > lo) & (p_c <= hi) if i > 0 else \
                   (p_c >= lo) & (p_c <= hi)
            if mask.sum() == 0:
                continue
            bin_acc = y_c[mask].mean()
            bin_conf = p_c[mask].mean()
            ece_c += (mask.sum() / n) * abs(bin_acc - bin_conf)
        out[c] = float(ece_c)
    return out


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> float:
    """Worst-case calibration gap across bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else \
               (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        mce = max(mce, abs(accuracies[mask].mean() - confidences[mask].mean()))
    return float(mce)


# ---------------------------------------------------------------------------
#  Dirichlet calibration — pure-torch implementation
# ---------------------------------------------------------------------------
#
#  netcal's DirichletCalibration supports the same model but its scipy-backend
#  optimizer is slow on 5k samples and breaks on Windows for some installs.
#  We use a torch-LBFGS implementation directly — same math, ~1s to fit.
#
#  Model: z_cal = W @ log(p) + b
#         where p = softmax(z_raw)
#         W: (K, K), b: (K,)
#  Fit by minimizing cross-entropy on val labels.
#  Regularize off-diagonal of W to prevent overfitting on 64 params / ~5k samples.
# ---------------------------------------------------------------------------

class DirichletCalibrator:
    def __init__(self, n_classes: int = NUM_CLASSES, l2_off_diag: float = 0.01,
                 l2_bias: float = 0.01):
        self.K = n_classes
        self.l2_off_diag = l2_off_diag
        self.l2_bias = l2_bias
        # Initialize W ≈ I (identity → no change)
        self.W = torch.eye(n_classes, dtype=torch.float32)
        self.b = torch.zeros(n_classes, dtype=torch.float32)

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            max_iter: int = 500, lr: float = 0.1, verbose: bool = True) -> "DirichletCalibrator":
        """Fit on (logits, labels) using LBFGS."""
        z = torch.from_numpy(logits).float()
        y = torch.from_numpy(labels).long()

        # Use log-softmax as input — this is the Dirichlet-calibration parameterisation
        log_p = F.log_softmax(z, dim=1)

        W = self.W.clone().requires_grad_(True)
        b = self.b.clone().requires_grad_(True)

        optimizer = torch.optim.LBFGS([W, b], lr=lr, max_iter=max_iter,
                                       tolerance_grad=1e-7, tolerance_change=1e-9,
                                       line_search_fn="strong_wolfe")
        off_diag_mask = 1.0 - torch.eye(self.K)

        def closure():
            optimizer.zero_grad()
            z_cal = log_p @ W.t() + b
            loss = F.cross_entropy(z_cal, y)
            # L2 regularization on off-diagonal of W and on b — keeps it near identity
            reg = self.l2_off_diag * ((W * off_diag_mask) ** 2).sum() \
                  + self.l2_bias * (b ** 2).sum()
            total = loss + reg
            total.backward()
            return total

        optimizer.step(closure)

        self.W = W.detach()
        self.b = b.detach()

        if verbose:
            with torch.no_grad():
                z_cal = log_p @ self.W.t() + self.b
                loss_before = F.cross_entropy(z, y).item()
                loss_after = F.cross_entropy(z_cal, y).item()
                print(f"  NLL  before: {loss_before:.4f}  →  after: {loss_after:.4f}")

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply calibration. Returns calibrated probabilities (N, K)."""
        z = torch.from_numpy(logits).float()
        log_p = F.log_softmax(z, dim=1)
        z_cal = log_p @ self.W.t() + self.b
        return F.softmax(z_cal, dim=1).numpy()

    def save(self, path: Path) -> None:
        np.savez(path, W=self.W.numpy(), b=self.b.numpy())

    @classmethod
    def load(cls, path: Path) -> "DirichletCalibrator":
        d = np.load(path)
        cal = cls(n_classes=d["W"].shape[0])
        cal.W = torch.from_numpy(d["W"]).float()
        cal.b = torch.from_numpy(d["b"]).float()
        return cal


# ---------------------------------------------------------------------------
#  Comparison: raw / global-T / Dirichlet
# ---------------------------------------------------------------------------

def fit_global_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """LBFGS for a single scalar T (same as current pipeline) — for ablation."""
    z = torch.from_numpy(logits).float()
    y = torch.from_numpy(labels).long()
    T = torch.tensor([1.0], requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / T.clamp(min=0.05), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().item())


def evaluate_calibration(name: str, probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute ECE / MCE / per-class ECE / NLL for a probability set."""
    eps = 1e-12
    nll = -np.log(probs[np.arange(len(labels)), labels].clip(min=eps)).mean()
    ece = expected_calibration_error(probs, labels)
    mce = maximum_calibration_error(probs, labels)
    pce = per_class_ece(probs, labels)
    acc = (probs.argmax(1) == labels).mean()
    return {
        "name": name,
        "accuracy": round(float(acc), 4),
        "nll": round(float(nll), 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "per_class_ece": {int(k): round(v, 4) for k, v in pce.items()},
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fit Dirichlet calibration on val set.")
    ap.add_argument("--mode", default="full",
                    choices=["full", "image_only", "swin_only", "effnet_only"])
    ap.add_argument("--checkpoint", default=None,
                    help="Override default checkpoint path for the mode.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cache", action="store_true",
                    help="Load logits from cache if present, skip forward pass.")
    args = ap.parse_args()

    ckpt_dir = Path(CKPT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(OUTPUT_DIR) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    cache_path = ckpt_dir / f"val_logits_{args.mode}.npz"

    # ── 1. Gather val logits ────────────────────────────────────────────────
    if args.cache and cache_path.exists():
        print(f"[CACHE] Loading val logits from {cache_path}")
        d = np.load(cache_path)
        val_logits, val_labels = d["logits"], d["labels"]
    else:
        print(f"[STEP] Building model ({args.mode}) ...")
        model = build_model(args.mode, args.checkpoint)

        print("[STEP] Building val DataLoader ...")
        _, val_loader, _ = get_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers
        )

        print(f"[STEP] Forward pass on val set (batch={args.batch_size}) ...")
        val_logits, val_labels = gather_logits(model, val_loader)
        print(f"       Collected {len(val_labels)} samples, "
              f"logits shape {val_logits.shape}")

        np.savez(cache_path, logits=val_logits, labels=val_labels)
        print(f"[OK]   Cached → {cache_path}")
        # Free GPU before fitting
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 2. Compare three calibrations ───────────────────────────────────────
    print("\n[STEP] Evaluating raw / global-T / Dirichlet calibrations on val set\n")

    # Raw
    probs_raw = F.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    raw_metrics = evaluate_calibration("raw", probs_raw, val_labels)

    # Global temperature (for ablation)
    T = fit_global_temperature(val_logits, val_labels)
    probs_T = F.softmax(torch.from_numpy(val_logits) / T, dim=1).numpy()
    T_metrics = evaluate_calibration(f"global_T={T:.3f}", probs_T, val_labels)

    # Dirichlet
    print("[STEP] Fitting Dirichlet calibration ...")
    dcal = DirichletCalibrator(n_classes=val_logits.shape[1])
    dcal.fit(val_logits, val_labels, verbose=True)
    probs_D = dcal.transform(val_logits)
    D_metrics = evaluate_calibration("dirichlet", probs_D, val_labels)

    # ── 3. Save calibration + report ────────────────────────────────────────
    dcal_path = ckpt_dir / "dirichlet_cal.npz"
    dcal.save(dcal_path)
    print(f"\n[OK]   Saved Dirichlet calibration → {dcal_path}")

    report = {
        "mode": args.mode,
        "n_val_samples": int(len(val_labels)),
        "n_classes": int(val_logits.shape[1]),
        "global_temperature": round(T, 4),
        "calibrations": [raw_metrics, T_metrics, D_metrics],
        "improvement_summary": {
            "ece_reduction_global_T": round(raw_metrics["ece"] - T_metrics["ece"], 4),
            "ece_reduction_dirichlet": round(raw_metrics["ece"] - D_metrics["ece"], 4),
            "ece_dirichlet_vs_global_T": round(T_metrics["ece"] - D_metrics["ece"], 4),
        }
    }
    report_path = report_dir / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK]   Saved calibration report → {report_path}")

    # ── 4. Print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(" CALIBRATION COMPARISON  (val set)")
    print("=" * 72)
    header = f"{'method':<22} {'NLL':>8} {'ECE':>8} {'MCE':>8} {'Acc':>8}"
    print(header)
    print("-" * len(header))
    for m in [raw_metrics, T_metrics, D_metrics]:
        print(f"{m['name']:<22} {m['nll']:>8.4f} {m['ece']:>8.4f} "
              f"{m['mce']:>8.4f} {m['accuracy']:>8.4f}")
    print("=" * 72)
    print("\n Per-class ECE")
    print("-" * 72)
    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    print(f"{'class':<8}", end="")
    for m in [raw_metrics, T_metrics, D_metrics]:
        print(f" {m['name'][:14]:>14}", end="")
    print()
    print("-" * 72)
    for c in range(len(class_names)):
        print(f"{class_names[c]:<8}", end="")
        for m in [raw_metrics, T_metrics, D_metrics]:
            print(f" {m['per_class_ece'][c]:>14.4f}", end="")
        print()
    print("=" * 72)
    print("\nNext: set LESIONIQ_USE_DIRICHLET=1 in your environment "
          "or .env to enable\n      the new calibration in the inference pipeline.")


if __name__ == "__main__":
    main()
