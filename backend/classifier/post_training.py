"""
LesionIQ -- Post-Training Optimization Suite
=============================================
Run AFTER all 4 ablation experiments finish.

Features:
  1. Per-class threshold tuning (free F1 boost)
  2. Ensemble all 4 checkpoints
  3. Temperature scaling (calibration)

Usage:
  python post_training.py
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from scipy.optimize import minimize
from backend.classifier.config import DEVICE, NUM_CLASSES, USE_AMP, OUTPUT_DIR, BATCH_SIZE
from backend.classifier.models import LesionIQHybrid
from backend.classifier.dataloader import get_dataloaders

# Use the same checkpoint location as inference.py (backend/checkpoints/).
# Fall back to OUTPUT_DIR/checkpoints/ for backward compatibility with older
# training runs that may have saved there.
_PRIMARY_CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
_LEGACY_CKPT_DIR  = Path(OUTPUT_DIR) / "checkpoints"
CKPT_DIR = str(_PRIMARY_CKPT_DIR if _PRIMARY_CKPT_DIR.exists() else _LEGACY_CKPT_DIR)
MODES = ["effnet_only", "swin_only", "image_only", "full"]


# =================================================================
#  UTILITY: Load a model from checkpoint
# =================================================================

def load_model(mode, checkpoint_name=None):
    """Load a trained model from its checkpoint."""
    if checkpoint_name is None:
        checkpoint_name = f"best_{mode}.pt"
    path = os.path.join(CKPT_DIR, checkpoint_name)
    if not os.path.exists(path):
        # Try the generic best_model.pt
        path = os.path.join(CKPT_DIR, "best_model.pt")
    
    model = LesionIQHybrid(mode=mode).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[LOADED] {mode} from {path} (F1={ckpt.get('val_f1', '?'):.4f})")
    return model


# =================================================================
#  UTILITY: Get all probabilities from a model on val set
# =================================================================

@torch.no_grad()
def get_all_probs(model, loader):
    """Run model on entire loader, return (probs, labels)."""
    all_probs, all_labels = [], []
    for images, meta, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        meta   = meta.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            # 4-way TTA
            def _fwd(x):
                out = model(x, meta)
                return out[0] if isinstance(out, tuple) else out

            logits = (_fwd(images) +
                      _fwd(torch.flip(images, dims=[3])) +
                      _fwd(torch.flip(images, dims=[2])) +
                      _fwd(torch.flip(images, dims=[2, 3]))) / 4.0

        # float32 before softmax -- AMP float16 softmax overflows
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
        meta   = meta.to(DEVICE, non_blocking=True)
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
#  1. PER-CLASS THRESHOLD TUNING
# =================================================================

def optimize_thresholds(probs, labels, num_classes=NUM_CLASSES):
    """Find optimal per-class thresholds that maximize macro-F1.
    
    Uses an 80/20 split of the validation data:
      - val_tune (80%): used to optimize scales via Nelder-Mead
      - val_check (20%): held-out sanity check to detect overfitting
    
    If tuned F1 on val_check is worse than baseline, reverts to uniform scales.
    """
    print("\n" + "=" * 60)
    print(" Per-Class Threshold Tuning (with overfit protection)")
    print("=" * 60)
    
    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    
    # --- Stratified 80/20 split of val data ---
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tune_idx, check_idx = next(splitter.split(probs, labels))
    
    probs_tune,  labels_tune  = probs[tune_idx],  labels[tune_idx]
    probs_check, labels_check = probs[check_idx], labels[check_idx]
    
    print(f"\n  Split: tune={len(tune_idx)} samples, check={len(check_idx)} samples")
    tune_counts = np.bincount(labels_tune, minlength=num_classes)
    check_counts = np.bincount(labels_check, minlength=num_classes)
    for i, name in enumerate(class_names):
        print(f"    {name:>5s}: tune={tune_counts[i]:4d}  check={check_counts[i]:4d}")
    
    # --- Baseline on both splits ---
    base_f1_tune  = f1_score(labels_tune,  probs_tune.argmax(1),  average="macro")
    base_f1_check = f1_score(labels_check, probs_check.argmax(1), average="macro")
    base_f1_full  = f1_score(labels,       probs.argmax(1),       average="macro")
    print(f"\n  Baseline (argmax):")
    print(f"    Full val:  Macro-F1 = {base_f1_full:.4f}")
    print(f"    Tune set:  Macro-F1 = {base_f1_tune:.4f}")
    print(f"    Check set: Macro-F1 = {base_f1_check:.4f}")
    
    # --- Optimize on tune split only ---
    def neg_f1(scales):
        scaled = probs_tune * scales
        preds = scaled.argmax(axis=1)
        return -f1_score(labels_tune, preds, average="macro")
    
    x0 = np.ones(num_classes)
    result = minimize(neg_f1, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-5})
    
    best_scales = result.x
    
    # --- Evaluate on HELD-OUT check split ---
    tuned_f1_tune  = f1_score(labels_tune,  (probs_tune * best_scales).argmax(1),  average="macro")
    tuned_f1_check = f1_score(labels_check, (probs_check * best_scales).argmax(1), average="macro")
    tuned_f1_full  = f1_score(labels,       (probs * best_scales).argmax(1),       average="macro")
    
    delta_tune  = tuned_f1_tune  - base_f1_tune
    delta_check = tuned_f1_check - base_f1_check
    delta_full  = tuned_f1_full  - base_f1_full
    
    print(f"\n  After tuning:")
    print(f"    Tune set:  Macro-F1 = {tuned_f1_tune:.4f}  (delta = +{delta_tune:.4f})")
    print(f"    Check set: Macro-F1 = {tuned_f1_check:.4f}  (delta = {delta_check:+.4f})")
    print(f"    Full val:  Macro-F1 = {tuned_f1_full:.4f}  (delta = {delta_full:+.4f})")
    
    # --- Diagnostic delta interpretation ---
    print(f"\n  +== DIAGNOSTIC ======================================+")
    print(f"  |  Threshold gain (check set): {delta_check:+.4f}                   |")
    if delta_check < 0:
        print(f"  |  WARNING: Nelder-Mead OVERFIT the tune set.          |")
        print(f"  |  Reverting to uniform thresholds (safe default).     |")
        best_scales = np.ones(num_classes)
    elif delta_check < 0.03:
        print(f"  |  Small gain -- model already well-calibrated OR val  |")
        print(f"  |  set too small. Using tuned scales cautiously.       |")
    elif delta_check > 0.15:
        print(f"  |  WARNING: Suspiciously large -- possible NM overfit. |")
        print(f"  |  Falling back to conservative grid search.           |")
        # Fallback: simple grid search over [0.5, 1.0, 1.5, 2.0] per class
        best_grid_f1 = base_f1_check
        best_grid_scales = np.ones(num_classes)
        for c in range(num_classes):
            for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
                trial = np.ones(num_classes)
                trial[c] = s
                trial_f1 = f1_score(labels_check, (probs_check * trial).argmax(1), average="macro")
                if trial_f1 > best_grid_f1:
                    best_grid_f1 = trial_f1
                    best_grid_scales = trial.copy()
        best_scales = best_grid_scales
        print(f"  |  Grid search check F1: {best_grid_f1:.4f}                   |")
    else:
        print(f"  |  OK: Healthy gain. Scales are reliable.              |")
    print(f"  +====================================================+")
    
    # Final scales
    print(f"\n  Final scales per class:")
    for i, name in enumerate(class_names):
        print(f"    {name:>5s}: {best_scales[i]:.4f}")
    
    final_f1 = f1_score(labels, (probs * best_scales).argmax(1), average="macro")
    final_auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    print(f"\n  Final Full-Val Macro-F1: {final_f1:.4f}  (was {base_f1_full:.4f})")
    print(f"  Macro AUC: {final_auc:.4f} (unchanged by thresholds)")
    
    print(f"\n  Classification Report (final):")
    final_preds = (probs * best_scales).argmax(axis=1)
    print(classification_report(labels, final_preds, target_names=class_names, digits=4))
    
    return best_scales, final_f1


# =================================================================
#  2. ENSEMBLE ALL 4 MODELS
# =================================================================

def ensemble_predictions(val_loader):
    """Load all 4 ablation models and average their predictions."""
    print("\n" + "="*60)
    print(" Ensemble (All 4 Ablation Models)")
    print("="*60)

    all_model_logits = []
    available_modes = []

    for mode in MODES:
        try:
            model = load_model(mode, f"best_{mode}.pt")
            # collect raw logits for logit-space ensemble
            logits, labels = get_all_logits(model, val_loader)
            all_model_logits.append(logits)
            available_modes.append(mode)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [SKIP] {mode}: {e}")

    if len(all_model_logits) < 2:
        print("  Need at least 2 models for ensemble. Skipping.")
        return None, None

    # Logit-space ensemble: average raw logits before a single softmax
    ensemble_probs = torch.softmax(
        torch.from_numpy(np.mean(all_model_logits, axis=0)), dim=1).numpy()
    ensemble_preds = ensemble_probs.argmax(axis=1)
    ensemble_f1 = f1_score(labels, ensemble_preds, average="macro")
    ensemble_auc = roc_auc_score(labels, ensemble_probs, multi_class='ovr', average='macro')

    print(f"\n  Ensemble ({len(available_modes)} models): Macro-F1 = {ensemble_f1:.4f}  AUC = {ensemble_auc:.4f}")
    print(f"  Models used: {available_modes}")

    # Also try weighted ensemble (optimize weights in logit-space)
    def neg_f1_weights(weights):
        weights = np.abs(weights) / np.sum(np.abs(weights))  # Normalize
        weighted_logits = sum(w * l for w, l in zip(weights, all_model_logits))
        probs = torch.softmax(torch.from_numpy(weighted_logits), dim=1).numpy()
        preds = probs.argmax(axis=1)
        return -f1_score(labels, preds, average="macro")

    w0 = np.ones(len(all_model_logits)) / len(all_model_logits)
    result = minimize(neg_f1_weights, w0, method='Nelder-Mead',
                      options={'maxiter': 2000})

    best_weights = np.abs(result.x) / np.sum(np.abs(result.x))
    weighted_logits = sum(w * l for w, l in zip(best_weights, all_model_logits))
    weighted_probs = torch.softmax(torch.from_numpy(weighted_logits), dim=1).numpy()
    weighted_preds = weighted_probs.argmax(axis=1)
    weighted_f1 = f1_score(labels, weighted_preds, average="macro")

    print(f"  Weighted Ensemble:  Macro-F1 = {weighted_f1:.4f}")
    print(f"  Optimal weights:")
    for mode, w in zip(available_modes, best_weights):
        print(f"    {mode:>12s}: {w:.4f}")

    return ensemble_probs, labels


# =================================================================
#  3. TEMPERATURE SCALING (Calibration)
# =================================================================

def _collect_logits(model, val_loader):
    """Forward-pass the entire val loader; return (logits float32, labels)."""
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, meta, labels in val_loader:
            images = images.to(DEVICE)
            meta   = meta.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                output = model(images, meta)
                logits = output[0] if isinstance(output, tuple) else output
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits).float(), torch.cat(all_labels)


def calibrate_temperature(model, val_loader):
    """Find optimal global temperature on validation set via grid search.

    Grid search is more robust than LBFGS for temperature scaling,
    especially with float16 logits from AMP training.
    """
    print("\n" + "="*60)
    print(" Global Temperature Scaling (Calibration)")
    print("="*60)

    all_logits, all_labels = _collect_logits(model, val_loader)

    # Before calibration
    probs_before = F.softmax(all_logits, dim=1).numpy()
    nll = nn.CrossEntropyLoss()
    nll_before = nll(all_logits, all_labels).item()

    # Grid search over temperature values (more robust than LBFGS)
    best_temp = 1.0
    best_nll  = nll_before
    for t in np.arange(0.5, 5.01, 0.05):
        trial_nll = nll(all_logits / t, all_labels).item()
        if trial_nll < best_nll:
            best_nll = trial_nll
            best_temp = t

    optimal_temp = round(best_temp, 2)
    probs_after  = F.softmax(all_logits / optimal_temp, dim=1).numpy()

    f1_before = f1_score(all_labels.numpy(), probs_before.argmax(1), average="macro")
    f1_after  = f1_score(all_labels.numpy(), probs_after.argmax(1),  average="macro")

    print(f"\n  Optimal global temperature: {optimal_temp:.2f}")
    print(f"  NLL: {nll_before:.4f} -> {best_nll:.4f}")
    print(f"  F1 before: {f1_before:.4f}  F1 after: {f1_after:.4f}")
    print(f"  (Temperature scaling improves calibration, not discriminative F1.)")

    return optimal_temp


# =================================================================
#  3b. PER-CLASS TEMPERATURE SCALING
# =================================================================

class PerClassTemperatureScaler:
    """8 independent scalar temperatures — one per output class dimension.

    Motivation: A single global temperature is biased toward the dominant
    validation classes (NV, MEL ~69% combined), causing it to over-sharpen
    rare-class logits (SCC, DF, VASC) that are already poorly calibrated.
    Per-class temperatures let each class independently find its optimal
    sharpness.

    Implementation:
        scaled_logits[k] = logits[k] / T_k      for each class k
        probs = softmax(scaled_logits)

    Temperatures are constrained to be >= 0.1 to prevent division by zero
    or extreme sharpening of a single class.
    """

    def __init__(self, n_classes: int = NUM_CLASSES):
        self.n_classes    = n_classes
        self.temperatures = np.ones(n_classes, dtype=np.float32)

    def fit(self, all_logits: torch.Tensor, all_labels: torch.Tensor,
            max_iter: int = 500) -> "PerClassTemperatureScaler":
        """Fit via LBFGS minimising cross-entropy NLL on val logits."""
        temps = torch.ones(self.n_classes, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [temps], lr=0.1, max_iter=max_iter,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            scaled = all_logits / temps.clamp(min=0.1)
            loss   = F.cross_entropy(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperatures = temps.detach().clamp(min=0.1).numpy().astype(np.float32)
        return self

    def transform(self, logits_np: np.ndarray) -> np.ndarray:
        """Apply per-class temperatures. (N,K) logits → (N,K) probs."""
        z      = torch.from_numpy(logits_np).float()
        temps  = torch.from_numpy(self.temperatures)
        scaled = z / temps                          # broadcast (N,K) / (K,)
        return F.softmax(scaled, dim=1).numpy()

    def save(self, path: str) -> None:
        np.save(path, self.temperatures)

    @classmethod
    def load(cls, path: str) -> "PerClassTemperatureScaler":
        obj = cls()
        obj.temperatures = np.load(path).astype(np.float32)
        obj.n_classes    = len(obj.temperatures)
        return obj


def calibrate_per_class_temperature(model, val_loader,
                                     class_names=None) -> np.ndarray:
    """Fit 8 per-class temperatures on the validation set.

    Returns the temperatures as a (K,) numpy array and prints a comparison
    table against the global temperature result.
    """
    if class_names is None:
        class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    print("\n" + "="*60)
    print(" Per-Class Temperature Scaling (Calibration)")
    print("="*60)

    all_logits, all_labels = _collect_logits(model, val_loader)

    # Baseline NLL
    nll_fn    = nn.CrossEntropyLoss()
    nll_raw   = nll_fn(all_logits, all_labels).item()

    # Global temperature (for comparison only — does not re-save)
    best_global, best_nll_g = 1.0, nll_raw
    for t in np.arange(0.5, 5.01, 0.05):
        trial = nll_fn(all_logits / t, all_labels).item()
        if trial < best_nll_g:
            best_nll_g, best_global = trial, t

    # Per-class temperatures
    scaler = PerClassTemperatureScaler(n_classes=all_logits.shape[1])
    scaler.fit(all_logits, all_labels)

    nll_pc = nll_fn(
        all_logits / torch.from_numpy(scaler.temperatures).clamp(min=0.1),
        all_labels,
    ).item()

    # F1 comparison
    probs_raw    = F.softmax(all_logits, dim=1).numpy()
    probs_global = F.softmax(all_logits / best_global, dim=1).numpy()
    probs_pc     = scaler.transform(all_logits.numpy())
    labels_np    = all_labels.numpy()

    f1_raw    = f1_score(labels_np, probs_raw.argmax(1),    average="macro")
    f1_global = f1_score(labels_np, probs_global.argmax(1), average="macro")
    f1_pc     = f1_score(labels_np, probs_pc.argmax(1),     average="macro")

    print(f"\n  {'':22s}  {'NLL':>8}  {'Macro-F1':>10}")
    print(f"  {'Raw (no calibration)':22s}  {nll_raw:>8.4f}  {f1_raw:>10.4f}")
    print(f"  {'Global T={:.2f}'.format(best_global):22s}  {best_nll_g:>8.4f}  {f1_global:>10.4f}")
    print(f"  {'Per-class T (x8)':22s}  {nll_pc:>8.4f}  {f1_pc:>10.4f}")
    print(f"\n  Per-class temperatures:")
    for name, t in zip(class_names, scaler.temperatures):
        # ascii bar (avoid Unicode chars; Windows cp1252 console crashes on them)
        bar = "#" * int(t * 10)
        print(f"    {name:<6s}  T={t:.3f}  {bar}")

    return scaler.temperatures


# =================================================================
#  MAIN
# =================================================================

if __name__ == "__main__":
    print("="*60)
    print(" LesionIQ Post-Training Optimization Suite")
    print("="*60)
    
    # Load data
    _, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    # --- 1. Per-class threshold tuning on best model ---
    try:
        best_model = load_model("full")
        probs, labels = get_all_probs(best_model, val_loader)
        scales, tuned_f1 = optimize_thresholds(probs, labels)
        
        # Save scales for inference
        np.save(os.path.join(CKPT_DIR, "optimal_scales.npy"), scales)
        print(f"  Saved optimal scales -> {os.path.join(CKPT_DIR, 'optimal_scales.npy')}")
    except Exception as e:
        print(f"  Threshold tuning failed: {e}")
        import traceback; traceback.print_exc()
    
    # --- 2. Ensemble ---
    try:
        ensemble_probs, labels = ensemble_predictions(val_loader)
        if ensemble_probs is not None:
            # Apply threshold tuning to ensemble
            scales, ensemble_tuned_f1 = optimize_thresholds(ensemble_probs, labels)
    except Exception as e:
        print(f"  Ensemble failed: {e}")
        import traceback; traceback.print_exc()
    
    # --- 3a. Global temperature scaling ---
    try:
        best_model = load_model("full")
        optimal_temp = calibrate_temperature(best_model, val_loader)
        np.save(os.path.join(CKPT_DIR, "optimal_temperature.npy"), optimal_temp)
        print(f"  Saved global temperature -> {os.path.join(CKPT_DIR, 'optimal_temperature.npy')}")
    except Exception as e:
        print(f"  Global temperature scaling failed: {e}")
        import traceback; traceback.print_exc()

    # --- 3b. Per-class temperature scaling ---
    try:
        best_model = load_model("full")
        pc_temps = calibrate_per_class_temperature(best_model, val_loader)
        pc_path  = os.path.join(CKPT_DIR, "per_class_temperatures.npy")
        np.save(pc_path, pc_temps)
        print(f"  Saved per-class temperatures -> {pc_path}")
    except Exception as e:
        print(f"  Per-class temperature scaling failed: {e}")
        import traceback; traceback.print_exc()
    
    print("\n" + "="*60)
    print(" Post-Training Optimization Complete!")
    print("="*60)
