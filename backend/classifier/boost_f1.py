"""
LesionIQ -- Aggressive Threshold Tuning (v1)
==============================================
Multiple approaches to push macro-F1 above 0.60:
  1. Differential Evolution (global optimizer, not local like Nelder-Mead)
  2. Greedy per-class grid search (optimize one class at a time)
  3. K-fold cross-validated thresholds

Runs on both single-model and ensemble probabilities.

Version History:
  v1 (this file): Symmetric bounds [0.3, 4.0] for all classes. DiffEvo won.
  v2 (boost_f1_v2.py): Asymmetric AK/SCC bounds, clinical-weighted F1, MEL safety.
  v3 (boost_f1_v3.py): Confusion matrix reframe — BCC suppress, SCC boost, MEL 80%.
  -> v3 is the current production version.

Usage:
  python boost_f1.py
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from scipy.optimize import differential_evolution
from sklearn.model_selection import StratifiedKFold

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
#  METHOD 1: Differential Evolution (global optimizer)
# =================================================================

def differential_evolution_tuning(probs, labels, tag=""):
    """Global optimizer -- much more robust than Nelder-Mead."""
    print(f"\n{'='*60}")
    print(f" Method 1: Differential Evolution {tag}")
    print(f"{'='*60}")

    baseline_f1 = f1_score(labels, probs.argmax(1), average="macro")
    print(f"  Baseline macro-F1: {baseline_f1:.4f}")

    def neg_f1(scales):
        preds = (probs * scales).argmax(axis=1)
        return -f1_score(labels, preds, average="macro")

    # Bounds: each scale between 0.3 and 4.0
    bounds = [(0.3, 4.0)] * NUM_CLASSES
    result = differential_evolution(
        neg_f1, bounds,
        seed=42,
        maxiter=500,
        popsize=30,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True,  # Local polish after global search
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
#  METHOD 2: Greedy per-class grid search
# =================================================================

def greedy_grid_search(probs, labels, tag=""):
    """Optimize one class at a time -- fewer DOF = less overfitting."""
    print(f"\n{'='*60}")
    print(f" Method 2: Greedy Per-Class Grid Search {tag}")
    print(f"{'='*60}")

    baseline_f1 = f1_score(labels, probs.argmax(1), average="macro")
    print(f"  Baseline macro-F1: {baseline_f1:.4f}")

    scales = np.ones(NUM_CLASSES)
    grid = np.arange(0.3, 4.01, 0.05)

    # Multiple passes -- interactions between classes mean one pass isn't enough
    for pass_num in range(3):
        improved = False
        for c in range(NUM_CLASSES):
            best_s = scales[c]
            best_f1 = f1_score(labels, (probs * scales).argmax(1), average="macro")
            for s in grid:
                trial = scales.copy()
                trial[c] = s
                trial_f1 = f1_score(labels, (probs * trial).argmax(1), average="macro")
                if trial_f1 > best_f1:
                    best_f1 = trial_f1
                    best_s = s
                    improved = True
            scales[c] = best_s
        current_f1 = f1_score(labels, (probs * scales).argmax(1), average="macro")
        print(f"  Pass {pass_num+1}: macro-F1 = {current_f1:.4f}")
        if not improved:
            break

    tuned_f1 = f1_score(labels, (probs * scales).argmax(1), average="macro")
    delta = tuned_f1 - baseline_f1
    print(f"  Tuned macro-F1:   {tuned_f1:.4f}  (delta = {delta:+.4f})")
    print(f"  Scales:")
    for i, name in enumerate(CLASS_NAMES):
        marker = " *" if abs(scales[i] - 1.0) > 0.1 else ""
        print(f"    {name:>5s}: {scales[i]:.4f}{marker}")

    return scales, tuned_f1


# =================================================================
#  METHOD 3: K-fold cross-validated thresholds
# =================================================================

def kfold_cv_thresholds(probs, labels, k=5, tag=""):
    """Train thresholds on K-1 folds, evaluate on held-out fold.
    Returns the average performance across folds (unbiased estimate)."""
    print(f"\n{'='*60}")
    print(f" Method 3: {k}-Fold CV Thresholds {tag}")
    print(f"{'='*60}")

    baseline_f1 = f1_score(labels, probs.argmax(1), average="macro")
    print(f"  Baseline macro-F1: {baseline_f1:.4f}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_f1s_baseline = []
    fold_f1s_tuned = []
    all_fold_scales = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(probs, labels)):
        p_train, l_train = probs[train_idx], labels[train_idx]
        p_val, l_val = probs[val_idx], labels[val_idx]

        # Optimize on train fold using greedy grid search (fast + low DOF)
        scales = np.ones(NUM_CLASSES)
        grid = np.arange(0.3, 4.01, 0.1)
        for _ in range(2):
            for c in range(NUM_CLASSES):
                best_s = scales[c]
                best_f1 = f1_score(l_train, (p_train * scales).argmax(1), average="macro")
                for s in grid:
                    trial = scales.copy()
                    trial[c] = s
                    trial_f1 = f1_score(l_train, (p_train * trial).argmax(1), average="macro")
                    if trial_f1 > best_f1:
                        best_f1 = trial_f1
                        best_s = s
                scales[c] = best_s

        # Evaluate on held-out fold
        baseline_fold = f1_score(l_val, p_val.argmax(1), average="macro")
        tuned_fold = f1_score(l_val, (p_val * scales).argmax(1), average="macro")
        fold_f1s_baseline.append(baseline_fold)
        fold_f1s_tuned.append(tuned_fold)
        all_fold_scales.append(scales)
        print(f"  Fold {fold+1}: baseline={baseline_fold:.4f}  tuned={tuned_fold:.4f}  delta={tuned_fold-baseline_fold:+.4f}")

    avg_baseline = np.mean(fold_f1s_baseline)
    avg_tuned = np.mean(fold_f1s_tuned)
    avg_scales = np.mean(all_fold_scales, axis=0)

    print(f"\n  CV Average: baseline={avg_baseline:.4f}  tuned={avg_tuned:.4f}  delta={avg_tuned-avg_baseline:+.4f}")
    print(f"  Averaged scales:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:>5s}: {avg_scales[i]:.4f}")

    # Apply averaged scales to full dataset
    full_tuned_f1 = f1_score(labels, (probs * avg_scales).argmax(1), average="macro")
    print(f"  Full-set with avg scales: {full_tuned_f1:.4f}")

    return avg_scales, avg_tuned


# =================================================================
#  Print classification report
# =================================================================

def print_report(probs, labels, scales, title=""):
    preds = (probs * scales).argmax(axis=1)
    f1 = f1_score(labels, preds, average="macro")
    print(f"\n  {title} -- Macro-F1: {f1:.4f}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    return f1


# =================================================================
#  MAIN
# =================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" LesionIQ -- Aggressive F1 Boost")
    print("=" * 60)

    _, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    # ── Single model (full) ──────────────────────────────────
    print("\n\n" + "#" * 60)
    print(" SINGLE MODEL: full")
    print("#" * 60)

    model = load_model("full")
    probs, labels = get_all_probs(model, val_loader)
    del model; torch.cuda.empty_cache()

    s1, f1_de = differential_evolution_tuning(probs, labels, "(single)")
    s2, f1_gs = greedy_grid_search(probs, labels, "(single)")
    s3, f1_cv = kfold_cv_thresholds(probs, labels, k=5, tag="(single)")

    # Pick best method
    best_single = max([(s1, f1_de, "DiffEvo"), (s2, f1_gs, "Greedy"), (s3, f1_cv, "CV")],
                      key=lambda x: x[1])
    print(f"\n  >> Best single-model method: {best_single[2]} (F1={best_single[1]:.4f})")
    print_report(probs, labels, best_single[0], f"Single model + {best_single[2]}")

    # ── Ensemble ─────────────────────────────────────────────
    print("\n\n" + "#" * 60)
    print(" ENSEMBLE (3 models)")
    print("#" * 60)

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

    if len(all_model_logits) >= 2:
        # Logit-space ensemble: average raw logits before a single softmax
        ensemble_probs = torch.softmax(
            torch.from_numpy(np.mean(all_model_logits, axis=0)), dim=1).numpy()
        ens_baseline = f1_score(labels, ensemble_probs.argmax(1), average="macro")
        print(f"\n  Ensemble baseline: {ens_baseline:.4f}")

        s1e, f1_de_e = differential_evolution_tuning(ensemble_probs, labels, "(ensemble)")
        s2e, f1_gs_e = greedy_grid_search(ensemble_probs, labels, "(ensemble)")
        s3e, f1_cv_e = kfold_cv_thresholds(ensemble_probs, labels, k=5, tag="(ensemble)")

        best_ens = max([(s1e, f1_de_e, "DiffEvo"), (s2e, f1_gs_e, "Greedy"), (s3e, f1_cv_e, "CV")],
                       key=lambda x: x[1])
        print(f"\n  >> Best ensemble method: {best_ens[2]} (F1={best_ens[1]:.4f})")
        print_report(ensemble_probs, labels, best_ens[0], f"Ensemble + {best_ens[2]}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  Single model baseline:   0.5924")
    print(f"  Single model best:       {best_single[1]:.4f} ({best_single[2]})")
    if len(all_model_logits) >= 2:
        print(f"  Ensemble baseline:       {ens_baseline:.4f}")
        print(f"  Ensemble best:           {best_ens[1]:.4f} ({best_ens[2]})")

        # Save the winning scales
        winner_scales = best_ens[0] if best_ens[1] >= best_single[1] else best_single[0]
        winner_f1 = max(best_ens[1], best_single[1])
        winner_name = f"ensemble_{best_ens[2]}" if best_ens[1] >= best_single[1] else f"single_{best_single[2]}"
    else:
        winner_scales = best_single[0]
        winner_f1 = best_single[1]
        winner_name = f"single_{best_single[2]}"

    scale_path = os.path.join(CKPT_DIR, "optimal_scales.npy")
    np.save(scale_path, winner_scales)
    print(f"\n  Winner: {winner_name} (F1={winner_f1:.4f})")
    print(f"  Saved -> {scale_path}")
    print(f"{'='*60}")
