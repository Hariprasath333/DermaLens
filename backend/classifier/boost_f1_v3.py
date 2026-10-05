"""
LesionIQ -- Clinical F1 Boost v3 (Confusion Matrix Reframe)
=============================================================
Targets the ACTUAL dominant failure modes from the confusion matrix:

  Priority 1: MEL→NV  22%  — MEL recall constraint >= 80%
  Priority 2: SCC→BCC 31%  — BCC suppression + SCC boost in DiffEvo
  Priority 3: AK→BKL  20%  — AK scale boost (carried over from v2)

Changes from v2:
  - BCC upper bound capped at 3.0 (was 4.0) to stop it absorbing SCC
  - SCC lower bound raised to 2.0, upper to 7.0 (was [1.5, 6.0])
  - NV upper bound capped at 2.0 to prevent over-suppression
  - MEL recall target lowered to 80% (85% was too aggressive)
  - Clinical weights: SCC 2.5→3.0, MEL 2.0→2.5
  - Full before/after confusion matrix with SCC→BCC tracking

Version History:
  v1 (boost_f1.py): Symmetric bounds [0.3, 4.0] for all classes.
  v2 (boost_f1_v2.py): Asymmetric AK/SCC bounds, clinical-weighted F1, MEL safety.
  v3 (this file): Confusion matrix reframe -- BCC suppress, SCC boost, MEL 80%.
  -> v3 is the current production version.

Usage:
  python backend/classifier/boost_f1_v3.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (f1_score, classification_report,
                             confusion_matrix, recall_score, precision_score)
from scipy.optimize import differential_evolution

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.classifier.config import DEVICE, NUM_CLASSES, USE_AMP, BATCH_SIZE, OUTPUT_DIR
from backend.classifier.models import LesionIQHybrid
from backend.classifier.dataloader import get_dataloaders

REPO_CKPT_DIR = str(BACKEND_ROOT / "checkpoints")
TRAIN_CKPT_DIR = str(Path(OUTPUT_DIR) / "checkpoints")

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
MODES = ["effnet_only", "image_only", "full"]
MEL, NV, BCC, AK, BKL, DF, VASC, SCC = range(8)
MALIGNANT = {MEL, BCC, AK, SCC}


# =================================================================
#  Load model + get probabilities (with 4-way TTA)
# =================================================================

def load_model(mode, ckpt_dir=TRAIN_CKPT_DIR):
    path = os.path.join(ckpt_dir, f"best_{mode}.pt")
    model = LesionIQHybrid(mode=mode).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    f1 = ckpt.get('val_f1', 0)
    print(f"  [LOADED] {mode} from {path} (F1={f1:.4f})")
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
#  Confusion matrix analysis (focused on actual failure modes)
# =================================================================

def analyze_confusion(probs, labels, scales=None, mel_threshold=None, title=""):
    if scales is not None:
        preds = (probs * scales).argmax(axis=1)
    else:
        preds = probs.argmax(axis=1)

    if mel_threshold is not None:
        preds[probs[:, MEL] >= mel_threshold] = MEL

    cm = confusion_matrix(labels, preds, labels=range(NUM_CLASSES))

    print(f"\n  {'='*62}")
    print(f"  {title}")
    print(f"  {'='*62}")

    # Full matrix
    print(f"\n  {'':>5s}", end="")
    for name in CLASS_NAMES:
        print(f"  {name:>5s}", end="")
    print("   Total")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>5s}", end="")
        for j in range(NUM_CLASSES):
            val = cm[i][j]
            if i == j:
                print(f"  [{val:4d}]", end="")  # brackets for correct
            elif val >= 10:
                print(f"  *{val:4d} ", end="")   # star for big misses
            else:
                print(f"   {val:4d} ", end="")
        print(f"   {cm[i].sum()}")

    # Key failure mode tracking
    print(f"\n  --- Priority Failure Modes ---")
    mel_total = (labels == MEL).sum()
    scc_total = (labels == SCC).sum()
    ak_total = (labels == AK).sum()

    mel_as_nv = cm[MEL][NV]
    scc_as_bcc = cm[SCC][BCC]
    ak_as_bkl = cm[AK][BKL]
    mel_recall = cm[MEL][MEL] / mel_total
    scc_recall = cm[SCC][SCC] / scc_total

    print(f"  [!!] MEL->NV:  {mel_as_nv:3d}/{mel_total} ({100*mel_as_nv/mel_total:.1f}%)  "
          f"MEL recall: {mel_recall:.3f}")
    print(f"  [!!] SCC->BCC: {scc_as_bcc:3d}/{scc_total} ({100*scc_as_bcc/scc_total:.1f}%)  "
          f"SCC recall: {scc_recall:.3f}")
    print(f"  [! ] AK->BKL:  {ak_as_bkl:3d}/{ak_total} ({100*ak_as_bkl/ak_total:.1f}%)")

    macro_f1 = f1_score(labels, preds, average="macro")
    print(f"\n  Macro-F1: {macro_f1:.4f}")
    return cm, macro_f1


# =================================================================
#  METHOD 1: Revised Asymmetric DiffEvo
# =================================================================

def revised_asymmetric_diffevo(probs, labels):
    print(f"\n{'='*62}")
    print(f" Method 1: Revised Asymmetric DiffEvo")
    print(f"{'='*62}")

    baseline_f1 = f1_score(labels, probs.argmax(1), average="macro")
    print(f"  Baseline macro-F1: {baseline_f1:.4f}")

    def neg_f1(scales):
        preds = (probs * scales).argmax(axis=1)
        return -f1_score(labels, preds, average="macro")

    # REVISED bounds: BCC suppressed, SCC boosted aggressively
    bounds = [
        (0.5, 4.0),   # MEL  — protected by recall constraint later
        (0.3, 2.0),   # NV   — cap to prevent over-suppression
        (0.3, 3.0),   # BCC  — absorbing SCC, needs suppression (was 4.0)
        (1.5, 6.0),   # AK   — chronically underconfident
        (0.3, 3.0),   # BKL  — absorbing AK
        (0.5, 5.0),   # DF   — small class needs boost
        (0.3, 3.0),   # VASC — already strong at 0.81
        (2.0, 7.0),   # SCC  — aggressive boost to compete with BCC (was [1.5,6.0])
    ]

    result = differential_evolution(
        neg_f1, bounds,
        seed=42, maxiter=1500, popsize=50,
        tol=1e-8, mutation=(0.5, 1.5), recombination=0.9, polish=True,
    )

    best_scales = result.x
    tuned_f1 = -result.fun

    print(f"  Tuned macro-F1: {tuned_f1:.4f}  (delta = {tuned_f1 - baseline_f1:+.4f})")
    print(f"  Scales:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:>5s}: {best_scales[i]:.4f}")

    return best_scales, tuned_f1


# =================================================================
#  METHOD 2: Clinical-Weighted DiffEvo (updated weights)
# =================================================================

def clinical_weighted_diffevo(probs, labels):
    print(f"\n{'='*62}")
    print(f" Method 2: Clinical-Weighted DiffEvo")
    print(f"{'='*62}")

    baseline_f1 = f1_score(labels, probs.argmax(1), average="macro")

    # Updated clinical weights: SCC and MEL bumped
    clinical_weights = np.array([2.5, 0.5, 1.0, 1.5, 1.0, 0.8, 0.8, 3.0])
    #                            MEL  NV   BCC  AK   BKL  DF   VASC SCC

    def neg_clinical_f1(scales):
        preds = (probs * scales).argmax(axis=1)
        per_class = f1_score(labels, preds, average=None, zero_division=0)
        return -np.average(per_class, weights=clinical_weights)

    bounds = [
        (0.5, 5.0),   # MEL
        (0.2, 2.0),   # NV
        (0.3, 3.0),   # BCC  — suppressed
        (1.5, 6.0),   # AK
        (0.3, 3.0),   # BKL
        (0.5, 5.0),   # DF
        (0.3, 3.0),   # VASC
        (2.0, 7.0),   # SCC  — aggressive boost
    ]

    result = differential_evolution(
        neg_clinical_f1, bounds,
        seed=42, maxiter=1500, popsize=50,
        tol=1e-8, mutation=(0.5, 1.5), recombination=0.9, polish=True,
    )

    best_scales = result.x
    preds = (probs * best_scales).argmax(axis=1)
    macro_f1 = f1_score(labels, preds, average="macro")
    per_class = f1_score(labels, preds, average=None, zero_division=0)
    clinical_f1 = np.average(per_class, weights=clinical_weights)

    print(f"  Standard macro-F1:  {macro_f1:.4f}  (delta = {macro_f1 - baseline_f1:+.4f})")
    print(f"  Clinical-weighted:  {clinical_f1:.4f}")
    print(f"  Per-class F1:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:>5s}: F1={per_class[i]:.4f}  (w={clinical_weights[i]:.1f})")
    print(f"  Scales:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:>5s}: {best_scales[i]:.4f}")

    return best_scales, macro_f1


# =================================================================
#  MEL Recall Safety (target >= 80%)
# =================================================================

def mel_recall_safety(probs, labels, scales, target_recall=0.80):
    print(f"\n{'='*62}")
    print(f" MEL Recall Safety (target >= {target_recall:.0%})")
    print(f"{'='*62}")

    scaled_probs = probs * scales
    mel_raw = probs[:, MEL]
    current_preds = scaled_probs.argmax(axis=1)

    current_recall = recall_score(labels == MEL, current_preds == MEL)
    current_prec = precision_score(labels == MEL, current_preds == MEL, zero_division=0)
    current_f1 = f1_score(labels, current_preds, average="macro")

    print(f"  Current MEL: recall={current_recall:.4f}, precision={current_prec:.4f}")
    print(f"  Current macro-F1: {current_f1:.4f}")

    if current_recall >= target_recall:
        print(f"  Already meets target. No adjustment needed.")
        return scales, None

    # Max acceptable F1 drop: 0.015
    f1_floor = current_f1 - 0.015
    best_threshold = None

    for t in np.arange(0.95, 0.01, -0.005):
        adjusted = current_preds.copy()
        adjusted[mel_raw >= t] = MEL
        rec = recall_score(labels == MEL, adjusted == MEL)
        macro = f1_score(labels, adjusted, average="macro")
        if rec >= target_recall and macro >= f1_floor:
            best_threshold = t
            break

    if best_threshold is None:
        # Relaxed search (no F1 floor)
        print(f"  Cannot meet target within F1 floor ({f1_floor:.4f}). Relaxing...")
        for t in np.arange(0.95, 0.01, -0.005):
            adjusted = current_preds.copy()
            adjusted[mel_raw >= t] = MEL
            rec = recall_score(labels == MEL, adjusted == MEL)
            if rec >= target_recall:
                best_threshold = t
                break

    if best_threshold is None:
        print(f"  WARNING: Cannot achieve {target_recall:.0%} recall.")
        return scales, None

    # Report
    adjusted = current_preds.copy()
    adjusted[mel_raw >= best_threshold] = MEL
    new_recall = recall_score(labels == MEL, adjusted == MEL)
    new_prec = precision_score(labels == MEL, adjusted == MEL, zero_division=0)
    new_f1 = f1_score(labels, adjusted, average="macro")

    print(f"  MEL threshold (raw prob): {best_threshold:.3f}")
    print(f"  MEL recall:    {current_recall:.4f} -> {new_recall:.4f}")
    print(f"  MEL precision: {current_prec:.4f} -> {new_prec:.4f}")
    print(f"  Macro-F1:      {current_f1:.4f} -> {new_f1:.4f}  ({new_f1 - current_f1:+.4f})")

    return scales, best_threshold


# =================================================================
#  MAIN
# =================================================================

if __name__ == "__main__":
    print("=" * 62)
    print(" LesionIQ -- Clinical F1 Boost v3 (Confusion Matrix Reframe)")
    print("=" * 62)

    _, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    # ── Build ensemble ──────────────────────────────────────────
    print("\n  Loading models for ensemble...")
    all_logits = []
    labels = None
    for mode in MODES:
        try:
            m = load_model(mode)
            # collect raw logits for logit-space ensemble
            logits, labels = get_all_logits(m, val_loader)
            all_logits.append(logits)
            del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [SKIP] {mode}: {e}")

    # Logit-space ensemble: average raw logits before a single softmax
    ensemble_probs = torch.softmax(
        torch.from_numpy(np.mean(all_logits, axis=0)), dim=1).numpy()
    ens_baseline = f1_score(labels, ensemble_probs.argmax(1), average="macro")
    print(f"\n  Ensemble models: {len(all_logits)}")
    print(f"  Ensemble baseline macro-F1: {ens_baseline:.4f}")

    # ── BEFORE: confusion matrix ────────────────────────────────
    analyze_confusion(ensemble_probs, labels, title="BASELINE (no scaling)")

    # ── Method 1: Revised Asymmetric DiffEvo ────────────────────
    scales_asym, f1_asym = revised_asymmetric_diffevo(ensemble_probs, labels)
    analyze_confusion(ensemble_probs, labels, scales_asym,
                      title="AFTER Revised Asymmetric DiffEvo")

    # ── Method 2: Clinical-Weighted DiffEvo ─────────────────────
    scales_clin, f1_clin = clinical_weighted_diffevo(ensemble_probs, labels)
    analyze_confusion(ensemble_probs, labels, scales_clin,
                      title="AFTER Clinical-Weighted DiffEvo")

    # ── Pick best ───────────────────────────────────────────────
    if f1_asym >= f1_clin:
        best_scales, best_f1, best_name = scales_asym, f1_asym, "Revised Asymmetric"
    else:
        best_scales, best_f1, best_name = scales_clin, f1_clin, "Clinical-Weighted"

    print(f"\n  >>> Best method: {best_name} (F1={best_f1:.4f})")

    # ── MEL Recall Safety ───────────────────────────────────────
    final_scales, mel_threshold = mel_recall_safety(
        ensemble_probs, labels, best_scales, target_recall=0.80)

    # ── FINAL confusion matrix ──────────────────────────────────
    analyze_confusion(ensemble_probs, labels, final_scales, mel_threshold,
                      title="FINAL: Best DiffEvo + MEL Safety")

    # ── Final classification report ─────────────────────────────
    final_preds = (ensemble_probs * final_scales).argmax(axis=1)
    if mel_threshold is not None:
        final_preds[ensemble_probs[:, MEL] >= mel_threshold] = MEL
    final_f1 = f1_score(labels, final_preds, average="macro")
    print(f"\n  FINAL Classification Report:")
    print(classification_report(labels, final_preds,
                                target_names=CLASS_NAMES, digits=4))

    # ── Save to BOTH checkpoint directories ─────────────────────
    for ckpt_dir in [TRAIN_CKPT_DIR, REPO_CKPT_DIR]:
        os.makedirs(ckpt_dir, exist_ok=True)

        scale_path = os.path.join(ckpt_dir, "optimal_scales.npy")
        np.save(scale_path, final_scales)
        print(f"  Scales -> {scale_path}")

        if mel_threshold is not None:
            thresh_path = os.path.join(ckpt_dir, "mel_safety_threshold.npy")
            np.save(thresh_path, mel_threshold)
            print(f"  MEL threshold -> {thresh_path}")

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f" SUMMARY")
    print(f"{'='*62}")
    print(f"  Ensemble baseline:       {ens_baseline:.4f}")
    print(f"  Revised Asymmetric:      {f1_asym:.4f}")
    print(f"  Clinical-Weighted:       {f1_clin:.4f}")
    print(f"  Final (+ MEL safety):    {final_f1:.4f}")
    if mel_threshold is not None:
        print(f"  MEL safety threshold:    {mel_threshold:.3f}")
    print(f"  Best method:             {best_name}")
    print(f"{'='*62}")
