"""
LesionIQ Hybrid Classifier — Evaluation / Clinical Test Suite
==============================================================
Runs on the held-out test set and produces:
  • Accuracy, Macro-F1, Per-class F1, AUC-ROC, Precision, Recall
  • Sensitivity & Specificity per class
  • Brier score (calibration)
  • Confusion matrix (PNG)
  • Full classification report (JSON)
  • Console summary table
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    brier_score_loss,
)

from backend.classifier.config import DEVICE, USE_AMP, OUTPUT_DIR, NUM_CLASSES, CONFIDENCE_THRESHOLD
from backend.classifier.models import LesionIQHybrid

# ── Collect predictions ──────────────────────────────────────

@torch.no_grad()
def _gather_predictions(
    model: LesionIQHybrid,
    loader: DataLoader,
    device: str = DEVICE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (all_probs  [N, C], all_preds  [N], all_labels  [N])."""
    model.eval()
    probs_list, labels_list = [], []

    for images, meta, labels in loader:
        images = images.to(device, non_blocking=True)
        meta   = meta.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits, _ = model(images, meta)

        probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        labels_list.append(labels.numpy())

    all_probs  = np.concatenate(probs_list, axis=0)
    all_preds  = all_probs.argmax(axis=1)
    all_labels = np.concatenate(labels_list, axis=0)
    return all_probs, all_preds, all_labels

# ── Per-class sensitivity & specificity ──────────────────────

def _sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sens, spec = {}, {}
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens[c] = float(tp / (tp + fn)) if (tp + fn) else 0.0
        spec[c] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return sens, spec

# ── Brier score (multi-class) ────────────────────────────────

def _brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1])[y_true]
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())

# ── Expected Calibration Error ───────────────────────────────

def _compute_ece(probs: np.ndarray, labels: np.ndarray,
                 n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) with equal-width confidence bins.

    ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|

    Args:
        probs:  (N, K) predicted probabilities.
        labels: (N,)  integer ground-truth class indices.
        n_bins: number of equal-width bins in [0, 1].

    Returns:
        Scalar ECE in [0, 1].
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # include left edge only for first bin; right-open otherwise
        mask = (confidences >= lo) & (confidences < hi) if i < n_bins - 1 \
               else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _compute_per_class_ece(probs: np.ndarray, labels: np.ndarray,
                            n_bins: int = 15) -> Dict[int, float]:
    """Per-class ECE: one-vs-rest calibration error for each class."""
    result = {}
    n = len(labels)
    for c in range(probs.shape[1]):
        p_c = probs[:, c]
        y_c = (labels == c).astype(np.float32)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece_c = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (p_c >= lo) & (p_c < hi) if i < n_bins - 1 \
                   else (p_c >= lo) & (p_c <= hi)
            if mask.sum() == 0:
                continue
            bin_acc  = y_c[mask].mean()
            bin_conf = p_c[mask].mean()
            ece_c += (mask.sum() / n) * abs(bin_acc - bin_conf)
        result[c] = float(ece_c)
    return result


def _plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray,
                               title: str, save_path: str,
                               n_bins: int = 15) -> None:
    """Save a reliability diagram (confidence vs accuracy per bin)."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(np.float32)

    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_accs, bin_confs, bin_counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi) if i < n_bins - 1 \
               else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_accs.append(accuracies[mask].mean())
        bin_confs.append(confidences[mask].mean())
        bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(bin_centers, bin_accs, width=1.0 / n_bins,
           alpha=0.7, color="steelblue", edgecolor="white", label="Accuracy")
    ax.step(bin_confs, bin_accs, where="mid",
            color="firebrick", lw=1.5, label="Acc per bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ece = _compute_ece(probs, labels, n_bins)
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ── Confusion matrix plot ────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ── Main evaluation entry point ──────────────────────────────

def _load_calibration_assets(ckpt_dir: Path):
    """Load whichever temperature calibration files exist.

    Returns (global_T, per_class_T) where missing values are None / 1.0.
    """
    global_T = 1.0
    t_path   = ckpt_dir / "optimal_temperature.npy"
    if t_path.exists():
        global_T = float(np.load(str(t_path)))

    pc_T  = None
    pc_path = ckpt_dir / "per_class_temperatures.npy"
    if pc_path.exists():
        pc_T = np.load(str(pc_path)).astype(np.float32)

    return global_T, pc_T


def evaluate(
    model: LesionIQHybrid,
    test_loader: DataLoader,
    class_names: List[str],
    device: str = DEVICE,
) -> Dict:
    """Run full clinical evaluation suite on the test set.

    Returns a metrics dictionary and saves artefacts to OUTPUT_DIR/reports/.
    Includes ECE (Expected Calibration Error) and reliability diagrams for:
      - raw uncalibrated probabilities
      - global temperature scaling
      - per-class temperature scaling (if available)
    """
    report_dir = Path(OUTPUT_DIR) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load logits so we can compute calibrated variants without re-running model
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for images, meta, labels_batch in test_loader:
            images = images.to(device, non_blocking=True)
            meta   = meta.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = model(images, meta)
                logits_b = out[0] if isinstance(out, tuple) else out
            logits_list.append(logits_b.float().cpu())
            labels_list.append(labels_batch)
    all_logits = torch.cat(logits_list).float()
    labels     = torch.cat(labels_list).numpy()

    # ── Probabilities: raw ────────────────────────────────────
    probs     = F.softmax(all_logits, dim=1).numpy()
    preds     = probs.argmax(axis=1)
    num_cls   = probs.shape[1]

    # ── Load calibration assets ───────────────────────────────
    from backend.classifier.inference import CKPT_DIR as _CKPT_DIR
    global_T, pc_T = _load_calibration_assets(Path(_CKPT_DIR))

    # Global-T calibrated probabilities
    probs_global = F.softmax(all_logits / global_T, dim=1).numpy()

    # Per-class-T calibrated probabilities (or fall back to global)
    if pc_T is not None:
        probs_pc = F.softmax(
            all_logits / torch.from_numpy(pc_T), dim=1).numpy()
    else:
        probs_pc = probs_global   # same as global if file absent

    # ── ECE comparison ────────────────────────────────────────
    ece_raw    = _compute_ece(probs,        labels)
    ece_global = _compute_ece(probs_global, labels)
    ece_pc     = _compute_ece(probs_pc,     labels)

    pc_ece_raw    = _compute_per_class_ece(probs,        labels)
    pc_ece_global = _compute_per_class_ece(probs_global, labels)
    pc_ece_pc     = _compute_per_class_ece(probs_pc,     labels)

    # Reliability diagrams
    _plot_reliability_diagram(
        probs,        labels, "Raw (uncalibrated)",
        str(report_dir / "reliability_raw.png"))
    _plot_reliability_diagram(
        probs_global, labels, f"Global T={global_T:.2f}",
        str(report_dir / "reliability_global_T.png"))
    if pc_T is not None:
        _plot_reliability_diagram(
            probs_pc, labels, "Per-class temperature",
            str(report_dir / "reliability_per_class_T.png"))

    # ── Core metrics (use raw probs for F1/AUC as before) ─────
    acc       = accuracy_score(labels, preds)
    macro_f1  = f1_score(labels, preds, average="macro")
    per_f1    = f1_score(labels, preds, average=None).tolist()
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels, preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    sens, spec = _sensitivity_specificity(labels, preds, num_cls)
    brier = _brier_score_multiclass(labels, probs)

    # ── Confidence analysis ───────────────────────────────────
    max_conf = probs.max(axis=1)
    uncertain_mask = max_conf < CONFIDENCE_THRESHOLD
    n_uncertain = int(uncertain_mask.sum())
    pct_uncertain = n_uncertain / len(labels) * 100

    # ── Confusion matrix ──────────────────────────────────────
    cm = confusion_matrix(labels, preds, labels=list(range(num_cls)))
    cm_path = str(report_dir / "confusion_matrix.png")
    _plot_confusion_matrix(cm, class_names, cm_path)

    # ── Build report dict ─────────────────────────────────────
    report = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "auc_roc_ovr": round(auc, 4) if not np.isnan(auc) else None,
        "brier_score": round(brier, 5),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "uncertain_predictions": n_uncertain,
        "uncertain_pct": round(pct_uncertain, 2),
        # ── Calibration ───────────────────────────────────────
        "calibration": {
            "ece_raw":    round(ece_raw,    4),
            "ece_global_T": round(ece_global, 4),
            "ece_per_class_T": round(ece_pc,  4),
            "global_T": round(global_T, 4),
            "per_class_T_available": pc_T is not None,
            "per_class_T": (
                {class_names[i]: round(float(pc_T[i]), 4) for i in range(len(pc_T))}
                if pc_T is not None else None),
            "per_class_ece_raw": {
                class_names[i]: round(pc_ece_raw[i], 4)
                for i in range(num_cls)},
            "per_class_ece_global_T": {
                class_names[i]: round(pc_ece_global[i], 4)
                for i in range(num_cls)},
            "per_class_ece_per_class_T": {
                class_names[i]: round(pc_ece_pc[i], 4)
                for i in range(num_cls)},
        },
        "per_class": {},
    }
    for i, name in enumerate(class_names):
        report["per_class"][name] = {
            "f1": round(per_f1[i], 4),
            "sensitivity": round(sens[i], 4),
            "specificity": round(spec[i], 4),
        }

    report["sklearn_classification_report"] = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0,
    )

    # ── Save JSON ─────────────────────────────────────────────
    json_path = str(report_dir / "classification_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Console summary ───────────────────────────────────────
    print(f"\n{'='*62}")
    print(f" TEST RESULTS")
    print(f"{'='*62}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Macro Prec.   : {precision:.4f}")
    print(f"  Macro Recall  : {recall:.4f}")
    print(f"  AUC-ROC (OVR) : {auc:.4f}")
    print(f"  Brier Score   : {brier:.5f}")
    print(f"  Uncertain     : {n_uncertain}/{len(labels)} ({pct_uncertain:.1f}%)"
          f"  [threshold={CONFIDENCE_THRESHOLD}]")
    print(f"{'─'*62}")
    print(f"  Calibration (ECE, 15 bins):")
    print(f"    Raw (uncalibrated) : {ece_raw:.4f}")
    print(f"    Global T={global_T:.2f}       : {ece_global:.4f}")
    pc_label = f"Per-class T (x{num_cls})" if pc_T is not None else "Per-class T (N/A)"
    print(f"    {pc_label:<20} : {ece_pc:.4f}")
    print(f"{'─'*62}")
    print(f"  {'Class':<22} {'F1':>6}  {'Sens':>6}  {'Spec':>6}  {'ECE(raw)':>9}")
    print(f"  {'─'*55}")
    for i, name in enumerate(class_names):
        print(f"  {name:<22} {per_f1[i]:6.4f}  {sens[i]:6.4f}  {spec[i]:6.4f}"
              f"  {pc_ece_raw[i]:9.4f}")
    print(f"{'='*62}")
    print(f"  Confusion matrix    → {cm_path}")
    print(f"  Reliability diagram → {report_dir}/reliability_*.png")
    print(f"  Full report         → {json_path}\n")

    return report
