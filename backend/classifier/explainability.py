"""
LesionIQ Hybrid Classifier — Clinical Explainability Suite
============================================================
Produces the following artefacts for clinical audit:

1. **Grad-CAM++** heatmaps on EfficientNet-B4 branch
2. **Swin attention rollout** visualisations
3. **SHAP** summary for the metadata branch
4. **Per-class calibration curves** (reliability diagrams)
5. **Uncertain-prediction report** (confidence < threshold)
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.classifier.config import (
    DEVICE, USE_AMP, OUTPUT_DIR, NUM_CLASSES,
    CONFIDENCE_THRESHOLD, NUM_EXPLAINABILITY_SAMPLES,
    SHAP_BACKGROUND_SAMPLES,
)
from backend.classifier.models import LesionIQHybrid

# =====================================================================
#  1. Grad-CAM++  (EfficientNet-B4 branch)
# =====================================================================

class _GradCAMPP:
    """Grad-CAM++ for a target convolutional layer."""

    def __init__(self, model: LesionIQHybrid, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    @torch.no_grad()
    def __call__(
        self, image: torch.Tensor, metadata: torch.Tensor, class_idx: Optional[int] = None,
    ) -> np.ndarray:
        self.model.eval()
        with torch.enable_grad():
            image = image.clone().requires_grad_(True)
            logits, _ = self.model(image, metadata)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            score = logits[0, class_idx]
            self.model.zero_grad()
            score.backward()

        grads = self.gradients[0]   # (C, H, W)
        acts  = self.activations[0] # (C, H, W)

        alpha_num   = grads.pow(2)
        alpha_denom = 2.0 * grads.pow(2) + (acts * grads.pow(3)).sum(dim=(1, 2), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num / alpha_denom

        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        return cam


def _overlay_heatmap(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a CAM heatmap on the original image (both in [0, 1] range)."""
    import cv2
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1].astype(np.float32) / 255.0
    overlay = alpha * heatmap + (1 - alpha) * image_np
    overlay = np.clip(overlay, 0, 1)
    return overlay


def generate_gradcam(
    model: LesionIQHybrid,
    loader: DataLoader,
    class_names: List[str],
    n_samples: int = NUM_EXPLAINABILITY_SAMPLES,
    device: str = DEVICE,
) -> None:
    """Save Grad-CAM++ heatmap overlays for a sample of test images."""
    if model.mode not in ("hybrid", "efficientnet"):
        print("[GRADCAM] Skipped — no EfficientNet branch in this mode.")
        return

    save_dir = Path(OUTPUT_DIR) / "explainability" / "gradcam"
    save_dir.mkdir(parents=True, exist_ok=True)

    target_layer = model.get_efficientnet_target_layer()
    gradcam = _GradCAMPP(model, target_layer)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    count = 0
    for images, meta, labels in loader:
        for i in range(images.size(0)):
            if count >= n_samples:
                return
            img_t = images[i:i+1].to(device)
            meta_t = meta[i:i+1].to(device)

            cam = gradcam(img_t, meta_t)

            img_np = images[i].cpu() * std + mean
            img_np = img_np.permute(1, 2, 0).numpy().clip(0, 1)

            overlay = _overlay_heatmap(img_np, cam)
            pred = model(img_t, meta_t)[0].argmax(1).item()

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(overlay); axes[1].set_title(
                f"Grad-CAM++ (pred={class_names[pred]}, true={class_names[labels[i]]})"
            ); axes[1].axis("off")
            plt.tight_layout()
            fig.savefig(str(save_dir / f"gradcam_{count:03d}.png"), dpi=120)
            plt.close(fig)
            count += 1

    print(f"[GRADCAM] Saved {count} heatmaps → {save_dir}")

# =====================================================================
#  2. Swin Attention Rollout
# =====================================================================

def _extract_swin_attention(model: LesionIQHybrid, image: torch.Tensor) -> List[np.ndarray]:
    """Register hooks on every Swin attention layer and collect weights."""
    if model.mode not in ("hybrid", "swin"):
        return []

    attn_weights = []

    def _hook(module, inp, out):
        if hasattr(module, "attn_drop"):
            with torch.no_grad():
                q, k = inp[0], inp[0]
                B, N, C = q.shape
                scale = (C // getattr(module, "num_heads", 1)) ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                attn_weights.append(attn.cpu().numpy())

    hooks = []
    for name, module in model.swin.backbone.named_modules():
        if "attn" in name.lower() and hasattr(module, "qkv"):
            hooks.append(module.register_forward_hook(_hook))

    with torch.no_grad():
        model.swin(image)

    for h in hooks:
        h.remove()
    return attn_weights


def generate_attention_maps(
    model: LesionIQHybrid,
    loader: DataLoader,
    class_names: List[str],
    n_samples: int = NUM_EXPLAINABILITY_SAMPLES,
    device: str = DEVICE,
) -> None:
    """Save Swin attention visualisations for a sample of test images."""
    if model.mode not in ("hybrid", "swin"):
        print("[ATTN] Skipped — no Swin branch in this mode.")
        return

    save_dir = Path(OUTPUT_DIR) / "explainability" / "attention_maps"
    save_dir.mkdir(parents=True, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    model.eval()
    count = 0
    for images, meta, labels in loader:
        for i in range(images.size(0)):
            if count >= n_samples:
                return

            img_t = images[i:i+1].to(device)
            img_224 = F.interpolate(img_t, size=(224, 224), mode="bilinear",
                                    align_corners=False)
            attn_list = _extract_swin_attention(model, img_224)

            if not attn_list:
                print("[ATTN] Could not extract attention weights; skipping.")
                return

            last_attn = attn_list[-1]
            attn_map = last_attn.mean(axis=(0, 1))
            side = int(np.sqrt(attn_map.shape[0]))
            if side * side == attn_map.shape[0]:
                attn_vis = attn_map.mean(axis=-1).reshape(side, side)
            else:
                attn_vis = attn_map.mean(axis=-1)
                attn_vis = attn_vis[:side*side].reshape(side, side)

            img_np = images[i].cpu() * std + mean
            img_np = img_np.permute(1, 2, 0).numpy().clip(0, 1)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
            im = axes[1].imshow(attn_vis, cmap="inferno")
            axes[1].set_title(f"Swin Attention (true={class_names[labels[i]]})")
            axes[1].axis("off")
            plt.colorbar(im, ax=axes[1], fraction=0.046)
            plt.tight_layout()
            fig.savefig(str(save_dir / f"attn_{count:03d}.png"), dpi=120)
            plt.close(fig)
            count += 1

    print(f"[ATTN] Saved {count} maps → {save_dir}")

# =====================================================================
#  3. SHAP — Metadata Feature Importance
# =====================================================================

def generate_shap_analysis(
    model: LesionIQHybrid,
    loader: DataLoader,
    meta_feature_names: List[str],
    n_background: int = SHAP_BACKGROUND_SAMPLES,
    device: str = DEVICE,
) -> None:
    """Run SHAP KernelExplainer on the metadata MLP and save summary plots."""
    try:
        import shap
    except ImportError:
        print("[SHAP] shap package not installed — skipping.")
        return

    save_dir = Path(OUTPUT_DIR) / "explainability" / "shap"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect metadata tensors from the loader
    all_meta = []
    for _, meta, _ in loader:
        all_meta.append(meta.numpy())
    all_meta = np.concatenate(all_meta, axis=0)

    bg_idx = np.random.choice(len(all_meta), min(n_background, len(all_meta)), replace=False)
    background = all_meta[bg_idx]

    def _meta_predict(meta_np: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            meta_t = torch.tensor(meta_np, dtype=torch.float32).to(device)
            embed = model.metadata_mlp(meta_t)
            aux_logits = model.meta_aux_head(embed)
            return F.softmax(aux_logits, dim=1).cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(_meta_predict, background)

        test_sample = all_meta[:min(50, len(all_meta))]
        shap_values = explainer.shap_values(test_sample)

    # Summary bar plot
    fig = plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        stacked = np.abs(np.stack(shap_values, axis=0)).mean(axis=0)
    else:
        stacked = np.abs(shap_values)
    mean_importance = stacked.mean(axis=0)
    order = np.argsort(mean_importance)[::-1]

    plt.barh(range(len(order)), mean_importance[order])
    plt.yticks(range(len(order)), [meta_feature_names[i] for i in order])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Metadata Feature Importance (SHAP)")
    plt.tight_layout()
    fig.savefig(str(save_dir / "shap_summary.png"), dpi=150)
    plt.close(fig)

    # Save raw SHAP values
    np.savez(str(save_dir / "shap_values.npz"),
             shap_values=shap_values if not isinstance(shap_values, list)
             else np.stack(shap_values, axis=0),
             feature_names=np.array(meta_feature_names))

    print(f"[SHAP] Summary plot + raw values → {save_dir}")

# =====================================================================
#  4. Per-Class Calibration Curves
# =====================================================================

def generate_calibration_curves(
    model: LesionIQHybrid,
    loader: DataLoader,
    class_names: List[str],
    device: str = DEVICE,
) -> None:
    """Plot per-class reliability diagrams and save to disk."""
    from sklearn.calibration import calibration_curve

    save_dir = Path(OUTPUT_DIR) / "explainability" / "calibration"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_probs, all_labels = [], []
    for images, meta, labels in loader:
        images = images.to(device, non_blocking=True)
        meta   = meta.to(device, non_blocking=True)
        with torch.no_grad():
            out = model(images, meta)
            logits = out[0] if isinstance(out, tuple) else out
        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    n_cls = probs.shape[1]
    cols = min(4, n_cls)
    rows = (n_cls + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)

    for c in range(n_cls):
        ax = axes[c // cols, c % cols]
        binary = (labels == c).astype(int)
        try:
            frac_pos, mean_pred = calibration_curve(binary, probs[:, c], n_bins=10)
            ax.plot(mean_pred, frac_pos, marker="o", label="Model")
        except ValueError:
            pass
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
        ax.set_title(class_names[c], fontsize=10)
        ax.set_xlabel("Mean predicted prob.")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=8)

    for c in range(n_cls, rows * cols):
        axes[c // cols, c % cols].axis("off")

    plt.suptitle("Per-Class Calibration Curves", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = str(save_dir / "calibration_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[CALIB] Calibration curves → {path}")

# =====================================================================
#  5. Uncertain-Prediction Report
# =====================================================================

def generate_uncertainty_report(
    model: LesionIQHybrid,
    loader: DataLoader,
    class_names: List[str],
    threshold: float = CONFIDENCE_THRESHOLD,
    device: str = DEVICE,
) -> None:
    """Flag low-confidence predictions and save as CSV + JSON summary."""
    save_dir = Path(OUTPUT_DIR) / "explainability"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    records = []
    idx = 0
    for images, meta, labels in loader:
        images = images.to(device, non_blocking=True)
        meta   = meta.to(device, non_blocking=True)
        with torch.no_grad():
            logits, _ = model(images, meta)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        for i in range(images.size(0)):
            max_p = probs[i].max()
            pred  = probs[i].argmax()
            if max_p < threshold:
                records.append({
                    "index": idx + i,
                    "true_label": class_names[labels[i]],
                    "predicted": class_names[pred],
                    "confidence": round(float(max_p), 4),
                    "top3": {class_names[j]: round(float(probs[i, j]), 4)
                             for j in probs[i].argsort()[-3:][::-1]},
                })
        idx += images.size(0)

    csv_path = str(save_dir / "uncertain_predictions.csv")
    import csv
    if records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "true_label",
                                                    "predicted", "confidence"])
            writer.writeheader()
            for r in records:
                writer.writerow({k: r[k] for k in writer.fieldnames})

    json_path = str(save_dir / "uncertain_predictions.json")
    with open(json_path, "w") as f:
        json.dump({"threshold": threshold, "total_uncertain": len(records),
                    "predictions": records}, f, indent=2)

    print(f"[UNCERTAIN] {len(records)} uncertain predictions → {csv_path}")

# =====================================================================
#  Master entry point
# =====================================================================

def run_explainability(
    model: LesionIQHybrid,
    test_loader: DataLoader,
    class_names: List[str],
    meta_feature_names: List[str],
    device: str = DEVICE,
) -> None:
    """Run all five explainability modules."""
    model.to(device).eval()
    print(f"\n{'='*62}")
    print(f" EXPLAINABILITY SUITE")
    print(f"{'='*62}\n")

    generate_gradcam(model, test_loader, class_names, device=device)
    generate_attention_maps(model, test_loader, class_names, device=device)
    generate_shap_analysis(model, test_loader, meta_feature_names, device=device)
    generate_calibration_curves(model, test_loader, class_names, device=device)
    generate_uncertainty_report(model, test_loader, class_names, device=device)

    print(f"\n  All explainability artefacts → {Path(OUTPUT_DIR) / 'explainability'}\n")
