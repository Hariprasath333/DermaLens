"""
LesionIQ Hybrid Classifier — Training Loop
============================================
Focal loss  ·  AdamW  ·  Cosine-annealing  ·  Layer-wise LR decay (Swin)
Mixed precision (AMP)  ·  Early stopping on macro-F1
Auxiliary metadata loss to keep the metadata branch relevant.
"""

import csv
import json
import os
import time
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score

from backend.classifier.config import (
    DEVICE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE, COSINE_T_MAX,
    SWIN_LR_DECAY, FOCAL_GAMMA, FOCAL_ALPHA, USE_AMP,
    META_AUX_WEIGHT, META_LR_SCALE, OUTPUT_DIR, GRAD_ACCUM_STEPS,
    LABEL_SMOOTHING,
)
from backend.classifier.models import LesionIQHybrid

# ── Focal loss ────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class focal loss with per-class alpha weighting and label smoothing."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal
        return focal.mean()

# ── Parameter groups with layer-wise LR decay ────────────────

def _swin_lr_groups(model: LesionIQHybrid, base_lr: float,
                    decay: float) -> List[dict]:
    """Assign decreasing LR to deeper Swin layers."""
    groups = []
    if not hasattr(model, 'swin'):
        return groups
        
    swin = model.swin
    layer_names = []
    for name, _ in swin.named_parameters():
        depth_tag = name.split(".")[0]
        if depth_tag not in layer_names:
            layer_names.append(depth_tag)

    num_layers = len(layer_names)
    for i, tag in enumerate(layer_names):
        lr_scale = decay ** (num_layers - 1 - i)
        params = [p for n, p in swin.named_parameters() if n.startswith(tag)]
        if params:
            groups.append({"params": params, "lr": base_lr * lr_scale})
    return groups


def build_optimizer(model: LesionIQHybrid) -> torch.optim.Optimizer:
    param_groups = []

    if hasattr(model, 'effnet'):
        param_groups.append({
            "params": list(model.effnet.parameters()),
            "lr": LR * 0.1,
            "name": "efficientnet",
        })

    if hasattr(model, 'swin'):
        param_groups.extend(_swin_lr_groups(model, LR * 0.1, SWIN_LR_DECAY))

    if hasattr(model, 'meta_mlp'):
        param_groups.append({
            "params": list(model.meta_mlp.parameters()),
            "lr": LR * META_LR_SCALE,
            "name": "metadata",
        })
        
    if hasattr(model, 'classifier'):
        param_groups.append({
            "params": list(model.classifier.parameters()),
            "lr": LR,
            "name": "fusion",
        })

    return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

# ── CutMix helper (better than MixUp for dermoscopy) ─────────

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix_data(x, y, alpha=1.0):
    """CutMix: cut a region from one image and paste into another.
    More realistic than MixUp for dermoscopy — preserves real texture."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x_cut = x.clone()
    x_cut[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
    # Adjust lambda to the actual area ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3)))
    return x_cut, y[index], lam

def cutmix_criterion(criterion, pred, y_orig, y_cut, lam):
    """Compute loss for CutMix blended targets."""
    return lam * criterion(pred, y_orig) + (1 - lam) * criterion(pred, y_cut)

# ── Training step ─────────────────────────────────────────────

def _train_one_epoch(
    model: LesionIQHybrid,
    loader: DataLoader,
    criterion: FocalLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
) -> float:
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    for i, (images, meta, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        meta   = meta.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply CutMix (30% of batches) — kept low to protect rare classes
        # DF/VASC have so few samples that cutting their only discriminative
        # region (e.g., vascular pattern) can destroy the learning signal
        use_cutmix = np.random.random() < 0.30
        if use_cutmix:
            mixed_images, labels_cut, lam = cutmix_data(images, labels)
        else:
            mixed_images, labels_cut, lam = images, labels, 1.0

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            output = model(mixed_images, meta, return_aux=True)
            if isinstance(output, tuple):
                logits, meta_aux = output
            else:
                logits, meta_aux = output, None
                
            if use_cutmix:
                main_loss = cutmix_criterion(criterion, logits, labels, labels_cut, lam)
            else:
                main_loss = criterion(logits, labels)
                
            if meta_aux is not None:
                if use_cutmix:
                    aux_loss = cutmix_criterion(criterion, meta_aux, labels, labels_cut, lam)
                else:
                    aux_loss = criterion(meta_aux, labels)
                loss = main_loss + META_AUX_WEIGHT * aux_loss
            else:
                loss = main_loss
                
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        
        if ((i + 1) % GRAD_ACCUM_STEPS == 0) or ((i + 1) == len(loader)):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += (loss.item() * GRAD_ACCUM_STEPS) * images.size(0)

    return running_loss / len(loader.dataset)

# ── Validation step with TTA ─────────────────────────────────

@torch.no_grad()
def _validate(
    model: LesionIQHybrid,
    loader: DataLoader,
    criterion: FocalLoss,
    device: str,
) -> Tuple[float, float, float, float]:
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, meta, labels in loader:
        images = images.to(device, non_blocking=True)
        meta   = meta.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            # Multi-TTA: 4 predictions averaged
            def _forward(x):
                out = model(x, meta)
                return out[0] if isinstance(out, tuple) else out
            
            logits_orig  = _forward(images)                          # Original
            logits_hflip = _forward(torch.flip(images, dims=[3]))    # H-flip
            logits_vflip = _forward(torch.flip(images, dims=[2]))    # V-flip
            logits_hvflip = _forward(torch.flip(images, dims=[2,3])) # H+V flip
            
            logits_avg = (logits_orig + logits_hflip + logits_vflip + logits_hvflip) / 4.0
                
            loss = criterion(logits_avg, labels)

        running_loss += loss.item() * images.size(0)
        # Cast to float32 BEFORE softmax — AMP autocast produces float16
        # logits, and softmax on float16 overflows to 0.0/1.0 which kills AUC
        probs = torch.softmax(logits_avg.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.extend(logits_avg.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    
    # Macro AUC (one-vs-rest, threshold-independent)
    all_probs = np.concatenate(all_probs, axis=0)
    try:
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError as e:
        print(f"  [WARN] AUC computation failed: {e}")
        macro_auc = 0.0
    
    return avg_loss, acc, macro_f1, macro_auc

# ── Full training routine ────────────────────────────────────

def train(
    model: LesionIQHybrid,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: np.ndarray,
    epochs: int = EPOCHS,
    device: str = DEVICE,
) -> str:
    """Train the model and return the path to the best checkpoint."""
    out = Path(OUTPUT_DIR)
    ckpt_dir = out / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = out / "logs";        log_dir.mkdir(parents=True, exist_ok=True)

    # Loss
    alpha = FOCAL_ALPHA
    if alpha is None:
        alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        alpha = torch.tensor(alpha, dtype=torch.float32).to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha,
                          label_smoothing=LABEL_SMOOTHING).to(device)

    model = model.to(device)
    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=COSINE_T_MAX)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    # Progressive unfreezing + SWA schedule:
    #   Epochs  1–15: Backbones frozen, head learns to fuse features
    #   Epoch  15:    Unfreeze last backbone stage at 0.1x head LR
    #   Epochs 20+:   SWA averages the adapted weights
    UNFREEZE_EPOCH = 15
    SWA_START = 20
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=LR * 0.5)
    swa_active = False
    unfrozen = False

    best_f1 = 0.0
    best_path = str(ckpt_dir / "best_model.pt")
    wait = 0
    patience = PATIENCE  # local copy — will be increased after unfreezing
    log_rows = []

    print(f"\n{'='*60}")
    print(f" Training  |  mode={model.mode}  |  epochs={epochs}  |  device={device}")
    print(f" Unfreeze at epoch {UNFREEZE_EPOCH}  |  SWA at epoch {SWA_START}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # --- Progressive unfreezing at UNFREEZE_EPOCH ---
        if epoch == UNFREEZE_EPOCH and not unfrozen and model.mode in ('image_only', 'full'):
            print(f"\n  >>> UNFREEZING last backbone stage at epoch {epoch} <<<")
            unfrozen_count = 0
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    # Unfreeze last stage of each backbone
                    should_unfreeze = False
                    if 'effnet' in name and ('blocks.6' in name or 'conv_head' in name or 'bn2' in name):
                        should_unfreeze = True
                    if 'swin' in name and ('layers.3' in name or 'norm' in name):
                        should_unfreeze = True
                    if should_unfreeze:
                        param.requires_grad = True
                        unfrozen_count += 1
            
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  Unfroze {unfrozen_count} param groups | Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")
            
            # Rebuild optimizer with unfrozen params at low LR
            # build_optimizer gives backbones LR*0.1, head gets LR
            optimizer = build_optimizer(model)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - epoch)
            
            # Log LR assignments for verification
            for pg in optimizer.param_groups:
                pg_name = pg.get('name', 'swin_layer')
                print(f"    LR: {pg['lr']:.6f}  ({pg_name}, {len(pg['params'])} params)")
            
            # Reset patience AND increase it — SWA needs 10-15 epochs of
            # averaging, so early stop must not fire before epoch 35
            wait = 0
            patience = 15  # was 10 from config
            print(f"  Patience increased to {patience} (SWA needs room)")
            unfrozen = True

        t0 = time.time()
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
        )
        val_loss, val_acc, val_f1, val_auc = _validate(model, val_loader, criterion, device)
        
        # Switch to SWA scheduler after SWA_START
        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_active = True
        else:
            scheduler.step()
        
        elapsed = time.time() - t0

        row = dict(epoch=epoch, train_loss=round(train_loss, 5),
                   val_loss=round(val_loss, 5), val_acc=round(val_acc, 4),
                   val_f1=round(val_f1, 4), val_auc=round(val_auc, 4),
                   time_s=round(elapsed, 1))
        log_rows.append(row)
        wandb.log(row)

        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_auc": val_auc,
                "mode": model.mode,
            }, best_path)
            marker = "  * BEST"
        else:
            wait += 1

        swa_tag = " [SWA]" if epoch >= SWA_START else ""
        print(f"  Epoch {epoch:3d}/{epochs}  |  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
              f"val_auc={val_auc:.4f}  "
              f"({elapsed:.1f}s){marker}{swa_tag}")

        if wait >= patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    # SWA: update batch norm stats and save SWA checkpoint
    if swa_active:
        print("\n  Updating SWA batch normalization statistics...")
        # Custom BN update — PyTorch's update_bn only passes images (input[0]),
        # discarding metadata. In 'full' mode this causes a shape mismatch
        # (2816 vs 2848) because the metadata branch is skipped.
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
                images = images.to(device, non_blocking=True)
                meta = meta.to(device, non_blocking=True)
                swa_model(images, meta)

        for module, momentum in momenta.items():
            module.momentum = momentum
        swa_path = str(ckpt_dir / "best_model_swa.pt")
        
        # Evaluate SWA model
        swa_loss, swa_acc, swa_f1, swa_auc = _validate(swa_model, val_loader, criterion, device)
        print(f"  SWA model: val_f1={swa_f1:.4f}  val_auc={swa_auc:.4f}  val_acc={swa_acc:.4f}")
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": swa_model.module.state_dict(),
            "val_f1": swa_f1,
            "val_auc": swa_auc,
            "mode": model.mode,
            "swa": True,
        }, swa_path)
        print(f"  SWA checkpoint saved -> {swa_path}")
        
        # If SWA is better, use it as the best
        if swa_f1 > best_f1:
            print(f"  SWA improved F1: {best_f1:.4f} -> {swa_f1:.4f}!")
            best_f1 = swa_f1
            import shutil
            shutil.copy(swa_path, best_path)

    # Save training log
    log_path = str(log_dir / "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n  Training log saved -> {log_path}")
    print(f"  Best checkpoint (F1={best_f1:.4f}) -> {best_path}")

    return best_path
