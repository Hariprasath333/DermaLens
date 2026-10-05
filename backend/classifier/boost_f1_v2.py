"""
LesionIQ -- Clinical F1 Boost v2
==================================
Targeted fixes for AK/SCC confusion and MEL recall safety.

Changes from v1:
  - Asymmetric DiffEvo bounds: AK [1.5, 6.0], SCC [1.5, 6.0]
  - MEL recall safety: post-hoc threshold to force recall >= 0.85
  - Confusion matrix: AK<->SCC misclassification analysis
  - Clinical-aware scoring: penalizes MEL/SCC misses more

Version History:
  v1 (boost_f1.py): Symmetric bounds [0.3, 4.0] for all classes.
  v2 (this file): Asymmetric AK/SCC bounds, clinical-weighted F1, MEL safety.
  v3 (boost_f1_v3.py): Confusion matrix reframe — BCC suppress, SCC boost, MEL 80%.
  -> v3 is the current production version.

Usage:
  python boost_f1_v2.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (f1_score, roc_auc_score, classification_report,
                             confusion_matrix, recall_score, precision_score)
from scipy.optimize import differential_evolution

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.classifier.config import DEVICE, NUM_CLASSES, USE_AMP, OUTPUT_DIR, BATCH_SIZE
from backend.classifier.models import LesionIQHybrid
from backend.classifier.dataloader import get_dataloaders

CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
MODES = ["effnet_only", "image_only", "full"]

# Class indices
MEL, NV, BCC, AK, BKL, DF, VASC, SCC = range(8)


# =================================================================
#  Load model + get probabilities (with 4-way TTA)
# =================================================================

def load_model(mode, checkpoint_name=None):
    if checkpoint_name is None:
        checkpoint_name = f"best_{mode}.pt"
    path = os.path.join(CKPT_DIR, checkpoint_name)
    model = LesionIQHybrid(mode=mode).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[LOADED] {mode} from {path} (F1={ckpt.get('val_f1', '?'):.4f})")
    return model


@torch.no_grad()
def get_all_probs(model, loader):
    all_probs, all_labels = [], []
    for images, meta, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        meta = meta.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            def _fwd(x):
                out = model(x, meta)
                return out[0] if isinstance(out, tuple) else out
            logits = (_fwd(images) +
                      _fwd(torch.flip(images, dims=[3])) +
                      _fwd(torch.flip(images, dims=[2])) +
                      _fwd(torch.flip(images, dims=[2, 3]))) / 4.0
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())
    return np.concatenate(all_probs), np.array(all_labels)


@torch.no_grad()
def get_all_logits(model, loader):
    """Like get_all_probs but returns raw logits (no softmax).
    Used for logit-space ensemble averaging before a single final softmax."""
    all_logits, all_labels = [], []
    for images, meta, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        meta = meta.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            def _fwd(x):
                out = model(x, meta)
                return out[0] if isinstance(out, tuple) else out
            logits = (_fwd(images) +
                      _fwd(torch.flip(images, dims=[3])) +
                      _fwd(torch.flip(images, dims=[2])) +
                      _fwd(torch.flip(images, dims=[2, 3]))) / 4.0
        all_logits.append(logits.float().cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.concatenate(all_logits), np.array(all_labels)


# =================================================================
#  ANALYSIS: AK <-> SCC confusion matrix
# =================================================================

def analyze_confusion(probs, labels, scales=None):
    """Print confusion matrix with focus on AK/SCC/MEL misclassifications."""
    if scales is not None:
        preds = (probs * scales).argmax(axis=1)
    else:
        preds = probs.argmax(axis=1)

    cm = confusion_matrix(labels, preds, labels=range(NUM_CLASSES))

    print("\n  Full Confusion Matrix:")
    print(f"  {'':>5s}", end="")
    for name in CLASS_NAMES:
        print(f"  {name:>5s}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>5s}", end="")
        for j in range(NUM_CLASSES):
            marker = " " if i == j else ""
            if cm[i][j] > 0 and i != j:
                marker = "*" if cm[i][j] >= 10 else " "
            print(f"  {cm[i][j]:5d}{marker}", end="")
        print(f"  (n={cm[i].sum()})")

    # AK/SCC specific analysis
    print(f"\n  --- AK <-> SCC Confusion ---")
    ak_total = (labels == AK).sum()
    scc_total = (labels == SCC).sum()
    ak_as_scc = cm[AK][SCC]
    scc_as_ak = cm[SCC][AK]
    ak_as_mel = cm[AK][MEL]
    scc_as_mel = cm[SCC][MEL]
    scc_as_bcc = cm[SCC][BCC]
    ak_as_bkl = cm[AK][BKL]
    print(f"  AK  predicted as SCC: {ak_as_scc:3d}/{ak_total} ({100*ak_as_scc/ak_total:.1f}%)")
    print(f"  SCC predicted as AK:  {scc_as_ak:3d}/{scc_total} ({100*scc_as_ak/scc_total:.1f}%)")
    print(f"  AK  predicted as MEL: {ak_as_mel:3d}/{ak_total} ({100*ak_as_mel/ak_total:.1f}%)")
    print(f"  AK  predicted as BKL: {ak_as_bkl:3d}/{ak_total} ({100*ak_as_bkl/ak_total:.1f}%)")
    print(f"  SCC predicted as MEL: {scc_as_mel:3d}/{scc_total} ({100*scc_as_mel/scc_total:.1f}%)")
    print(f"  SCC predicted as BCC: {scc_as_bcc:3d}/{scc_total} ({100*scc_as_bcc/scc_total:.1f}%)")

    # MEL analysis
    print(f"\n  --- MEL Recall Analysis ---")
    mel_total = (labels == MEL).sum()
    mel_recall = cm[MEL][MEL] / mel_total
    mel_as_nv = cm[MEL][NV]
    mel_as_bkl = cm[MEL][BKL]
    print(f"  MEL recall: {mel_recall:.4f} ({cm[MEL][MEL]}/{mel_total})")
    print(f"  MEL predicted as NV:  {mel_as_nv:3d}/{mel_total} ({100*mel_as_nv/mel_total:.1f}%)")
    print(f"  MEL predicted as BKL: {mel_as_bkl:3d}/{mel_total} ({100*mel_as_bkl/mel_total:.1f}%)")

    return cm


# =================================================================
#  METHOD: Asymmetric DiffEvo (boosted AK/SCC bounds)
# =================================================================

def asymmetric_diffevo(probs, labels, tag=""):
    """DiffEvo with wider search bounds for AK and SCC."""
    print(f"\n{'='*60}")
    print(f" Asymmetric DiffEvo {tag}")
    print(f"{'='*60}")

    baseline_f1 = f1_score(labels, probs.argmax(1), average="macro")
    print(f"  Baseline macro-F1: {baseline_f1:.4f}")

    def neg_f1(scales):
        preds = (probs * scales).argmax(axis=1)
        return -f1_score(labels, preds, average="macro")

    # Asymmetric bounds -- force higher scaling for rare/confused classes
    bounds = [
        (0.3, 4.0),   # MEL
        (0.3, 4.0),   # NV
        (0.3, 4.0),   # BCC
        (1.5, 6.0),   # AK  -- force higher (confused with SCC/BKL)
        (0.3, 4.0),   # BKL
        (0.5, 5.0),   # DF  -- also rare
        (0.3, 4.0),   # VASC
        (1.5, 6.0),   # SCC -- force higher (worst F1, malignant)
    ]

    result = differential_evolution(
        neg_f1, bounds,
        seed=42,
        maxiter=1000,   # more iterations for asymmetric search
        popsize=40,     # larger population
        tol=1e-7,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True,
    )

    best_scales = result.x
    tuned_f1 = -result.fun
    delta = tuned_f1 - baseline_f1

    print(f"  Tuned macro-F1:   {tuned_f1:.4f}  (delta = {delta:+.4f})")
    print(f"  Scales:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:>5s}: {best_scales[i]:.4f}")

    return best_scales, tuned_f1


# =================================================================
#  METHOD: Clinical-aware scoring (penalize MEL/SCC misses)
# =================================================================

def clinical_aware_diffevo(probs, labels, tag=""):
    """Optimize a clinical-weighted F1 that penalizes MEL/SCC misses."""
    print(f"\n{'='*60}")
    print(f" Clinical-Aware DiffEvo {tag}")
    print(f"{'='*60}")

    baseline_f1 = f1_score(labels, probs.argmax(1), average="macro")
    print(f"  Baseline macro-F1: {baseline_f1:.4f}")

    # Clinical weights: MEL and SCC misses are more dangerous
    # Higher weight = optimizer tries harder to get that class right
    clinical_weights = np.array([2.0, 0.5, 1.0, 1.5, 1.0, 1.0, 1.0, 2.5])
    #                            MEL  NV   BCC  AK   BKL  DF   VASC SCC

    def neg_clinical_f1(scales):
        preds = (probs * scales).argmax(axis=1)
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        # Weighted macro-F1
        weighted_f1 = np.average(per_class_f1, weights=clinical_weights)
        return -weighted_f1

    bounds = [
        (0.3, 5.0),   # MEL -- high clinical importance
        (0.2, 3.0),   # NV  -- dominant class, can be suppressed
        (0.3, 4.0),   # BCC
        (1.5, 6.0),   # AK
        (0.3, 4.0),   # BKL
        (0.5, 5.0),   # DF
        (0.3, 4.0),   # VASC
        (1.5, 6.0),   # SCC -- highest clinical importance
    ]

    result = differential_evolution(
        neg_clinical_f1, bounds,
        seed=42,
        maxiter=1000,
        popsize=40,
        tol=1e-7,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True,
    )

    best_scales = result.x
    preds = (probs * best_scales).argmax(axis=1)
    tuned_f1 = f1_score(labels, preds, average="macro")  # standard macro for comparison
    per_class = f1_score(labels, preds, average=None, zero_division=0)
    clinical_f1 = np.average(per_class, weights=clinical_weights)

    print(f"  Standard macro-F1: {tuned_f1:.4f}  (delta = {tuned_f1 - baseline_f1:+.4f})")
    print(f"  Clinical-weighted: {clinical_f1:.4f}")
    print(f"  Per-class F1:")
    for i, name in enumerate(CLASS_NAMES):
        w = clinical_weights[i]
        print(f"    {name:>5s}: F1={per_class[i]:.4f}  (weight={w:.1f})")
    print(f"  Scales:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:>5s}: {best_scales[i]:.4f}")

    return best_scales, tuned_f1, per_class


# =================================================================
#  MEL Recall Safety Threshold
# =================================================================

def mel_recall_safety(probs, labels, scales, target_recall=0.75):
    """Post-hoc: find MEL probability threshold to boost recall.
    
    Searches from high to low threshold, finding the highest threshold
    (most selective) that still achieves the target recall.
    Rejects solutions that drop macro-F1 below a floor.
    """
    print(f"\n{'='*60}")
    print(f" MEL Recall Safety (target >= {target_recall:.0%})")
    print(f"{'='*60}")

    scaled_probs = probs * scales
    # Use RAW probs for MEL threshold (not scaled), to get meaningful values
    mel_probs_raw = probs[:, MEL]

    # Current recall
    current_preds = scaled_probs.argmax(axis=1)
    current_mel_recall = recall_score(labels == MEL, current_preds == MEL)
    current_mel_precision = precision_score(labels == MEL, current_preds == MEL, zero_division=0)
    current_macro_f1 = f1_score(labels, current_preds, average="macro")
    print(f"  Current MEL: recall={current_mel_recall:.4f}, precision={current_mel_precision:.4f}")
    print(f"  Current macro-F1: {current_macro_f1:.4f}")

    if current_mel_recall >= target_recall:
        print(f"  Already meets target. No adjustment needed.")
        return scales, None

    # Search from HIGH to LOW threshold (most selective first)
    # This finds the HIGHEST threshold that achieves target recall
    macro_f1_floor = current_macro_f1 - 0.02  # max acceptable F1 drop
    best_threshold = None
    best_f1 = 0

    for threshold in np.arange(0.95, 0.01, -0.005):
        mel_mask = mel_probs_raw >= threshold
        adjusted_preds = current_preds.copy()
        adjusted_preds[mel_mask] = MEL
        recall = recall_score(labels == MEL, adjusted_preds == MEL)
        macro = f1_score(labels, adjusted_preds, average="macro")

        if recall >= target_recall and macro >= macro_f1_floor:
            best_threshold = threshold
            best_f1 = macro
            break  # highest threshold that works = least collateral damage

    if best_threshold is None:
        # Try with relaxed F1 floor
        print(f"  Cannot achieve {target_recall:.0%} recall within F1 floor ({macro_f1_floor:.4f}).")
        print(f"  Trying relaxed search (no F1 floor)...")
        for threshold in np.arange(0.95, 0.01, -0.005):
            mel_mask = mel_probs_raw >= threshold
            adjusted_preds = current_preds.copy()
            adjusted_preds[mel_mask] = MEL
            recall = recall_score(labels == MEL, adjusted_preds == MEL)
            macro = f1_score(labels, adjusted_preds, average="macro")
            if recall >= target_recall:
                best_threshold = threshold
                best_f1 = macro
                break

    if best_threshold is None:
        print(f"  WARNING: Cannot achieve {target_recall:.0%} recall at any threshold.")
        return scales, None

    # Apply and report
    adjusted_preds = current_preds.copy()
    adjusted_preds[mel_probs_raw >= best_threshold] = MEL
    new_recall = recall_score(labels == MEL, adjusted_preds == MEL)
    new_precision = precision_score(labels == MEL, adjusted_preds == MEL, zero_division=0)
    new_macro_f1 = f1_score(labels, adjusted_preds, average="macro")

    print(f"  MEL threshold (raw prob): {best_threshold:.3f}")
    print(f"  MEL recall:    {current_mel_recall:.4f} -> {new_recall:.4f}")
    print(f"  MEL precision: {current_mel_precision:.4f} -> {new_precision:.4f}")
    print(f"  Macro-F1:      {current_macro_f1:.4f} -> {new_macro_f1:.4f}  (delta = {new_macro_f1 - current_macro_f1:+.4f})")

    if new_macro_f1 < macro_f1_floor:
        print(f"  WARNING: F1 drop exceeds floor. Discarding MEL override.")
        return scales, None

    return scales, best_threshold


# =================================================================
#  Print classification report
# =================================================================

def print_report(probs, labels, scales, mel_threshold=None, title=""):
    preds = (probs * scales).argmax(axis=1)
    if mel_threshold is not None:
        # MEL threshold operates on RAW probs (not scaled)
        mel_probs_raw = probs[:, MEL]
        preds[mel_probs_raw >= mel_threshold] = MEL
    f1 = f1_score(labels, preds, average="macro")
    print(f"\n  {title} -- Macro-F1: {f1:.4f}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    return f1


# =================================================================
#  MAIN
# =================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" LesionIQ -- Clinical F1 Boost v2")
    print("=" * 60)

    _, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    # ── Build ensemble probabilities ─────────────────────────
    print("\n  Loading all models for ensemble...")
    all_model_logits = []
    for mode in MODES:
        try:
            m = load_model(mode, f"best_{mode}.pt")
            # collect raw logits for logit-space ensemble
            logits, labels = get_all_logits(m, val_loader)
            all_model_logits.append(logits)
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [SKIP] {mode}: {e}")

    # Logit-space ensemble: average raw logits before a single softmax
    ensemble_probs = torch.softmax(
        torch.from_numpy(np.mean(all_model_logits, axis=0)), dim=1).numpy()
    ens_baseline = f1_score(labels, ensemble_probs.argmax(1), average="macro")
    print(f"\n  Ensemble baseline: {ens_baseline:.4f}")

    # ── Step 0: Confusion matrix analysis (baseline) ─────────
    print("\n" + "#" * 60)
    print(" BASELINE CONFUSION ANALYSIS")
    print("#" * 60)
    analyze_confusion(ensemble_probs, labels)

    # ── Step 1: Asymmetric DiffEvo ───────────────────────────
    scales_asym, f1_asym = asymmetric_diffevo(ensemble_probs, labels, "(ensemble)")
    print_report(ensemble_probs, labels, scales_asym, title="Asymmetric DiffEvo")

    # ── Step 2: Clinical-aware DiffEvo ───────────────────────
    scales_clin, f1_clin, per_class_clin = clinical_aware_diffevo(ensemble_probs, labels, "(ensemble)")
    print_report(ensemble_probs, labels, scales_clin, title="Clinical-Aware DiffEvo")

    # ── Step 3: Pick best, then apply MEL safety ─────────────
    if f1_asym >= f1_clin:
        best_scales, best_f1, best_name = scales_asym, f1_asym, "Asymmetric"
    else:
        best_scales, best_f1, best_name = scales_clin, f1_clin, "Clinical"

    print(f"\n  Best method: {best_name} (F1={best_f1:.4f})")

    # Confusion after optimization
    print("\n" + "#" * 60)
    print(f" CONFUSION AFTER {best_name.upper()} OPTIMIZATION")
    print("#" * 60)
    analyze_confusion(ensemble_probs, labels, best_scales)

    # ── Step 4: MEL recall safety ────────────────────────────
    final_scales, mel_threshold = mel_recall_safety(ensemble_probs, labels, best_scales, target_recall=0.85)

    # ── Final report ─────────────────────────────────────────
    print("\n" + "#" * 60)
    print(" FINAL CONFIGURATION")
    print("#" * 60)
    final_f1 = print_report(ensemble_probs, labels, final_scales, mel_threshold,
                            title="FINAL: Ensemble + Tuning + MEL Safety")

    # ── Save ─────────────────────────────────────────────────
    scale_path = os.path.join(CKPT_DIR, "optimal_scales.npy")
    np.save(scale_path, final_scales)
    print(f"\n  Scales saved -> {scale_path}")

    if mel_threshold is not None:
        thresh_path = os.path.join(CKPT_DIR, "mel_safety_threshold.npy")
        np.save(thresh_path, mel_threshold)
        print(f"  MEL threshold saved -> {thresh_path}")

    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"  Ensemble baseline:              {ens_baseline:.4f}")
    print(f"  Asymmetric DiffEvo:             {f1_asym:.4f}")
    print(f"  Clinical-Aware DiffEvo:         {f1_clin:.4f}")
    print(f"  Final (best + MEL safety):      {final_f1:.4f}")
    print(f"{'='*60}")
