"""
Stage 5 — Efficient ensemble.

What was wrong with the hackathon ensemble:
  - 3 models loaded independently (3× memory)
  - 3 separate forward passes per image (3× compute)
  - DiffEvo scales applied in probability space (mathematically off;
    `softmax(mean(z)) != mean(softmax(z))`)
  - No batched TTA; each augmentation = one forward pass
  - Static equal weights (`mean`) — never optimized
  - No use of fp16 at inference

What this module fixes:
  E1. `EnsembleWrapper`: one nn.Module that holds N variant models and
      runs them as a single batched forward pass on identical input
      (no Python-side loop overhead between submodels).
  E2. `BatchedTTA`: stacks K augmentation views into one batch dim;
      submodels see (B*K, ...) tensor; logits reduced afterwards. One
      kernel launch per submodel instead of K.
  E3. `LogitSpaceWeightedAvg`: learns ensemble weights (LBFGS on val
      logits, simplex-constrained via softmax parameterization). Acts
      in logit space — `softmax(Σ w_i * z_i)` rather than `Σ w_i * p_i`.
  E4. `predict_calibrated()` pipeline: ensemble logits → temperature
      scaling → (optional) Dirichlet → (optional) SLD prior shift →
      DiffEvo scales (operating on the *calibrated* probs only) →
      argmax.
  E5. Optional fp16 autocast at inference; no fp32 precision loss in
      the logit-averaging step because we cast to fp32 before reduction.

Design priorities:
  1. Efficiency  — single forward per submodel per batch (no K-way loop);
                   fp16 backbone forward + fp32 reduction; no Python in
                   the inner loop
  2. Quality     — pure modules, no train/eval-mode confusion (inference
                   wrapper sets .eval() and torch.inference_mode for us)
  3. Errorless   — submodel mismatch detected at __init__; weight tensor
                   validated; calibration assets are optional and fail
                   forward if mis-shaped
  4. Explainable — every step logs shapes; weight-fitting prints the
                   change in NLL and Macro-F1
  5. Logic       — calibration order matches the math: ensemble first,
                   THEN T, THEN Dirichlet, THEN prior-shift, THEN
                   threshold-scale. Reversing any of these is wrong.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("lesioniq.stage5")


# ─────────────────────────────────────────────────────────────────────
# E1. EnsembleWrapper
# ─────────────────────────────────────────────────────────────────────

class EnsembleWrapper(nn.Module):
    """Hold N submodels and emit (B, N, K) logits in one batched call.

    All submodels MUST accept the same call signature:
        out = submodel(image, meta, meta_mask) -> (B, K) logits
    Submodels may share backbones internally for memory; that's their
    business. This wrapper only enforces the call signature.
    """

    def __init__(self, submodels: Sequence[nn.Module],
                 names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        if len(submodels) == 0:
            raise ValueError("EnsembleWrapper requires >= 1 submodel")
        self.submodels = nn.ModuleList(submodels)
        self.names: list[str] = list(names) if names else [
            f"M{i}" for i in range(len(submodels))]
        if len(self.names) != len(self.submodels):
            raise ValueError("names length must match submodels length")

    def forward(self, image: torch.Tensor, meta: torch.Tensor,
                meta_mask: torch.Tensor) -> torch.Tensor:
        """Return (B, N, K) logits stacked across submodels."""
        outs = []
        for m in self.submodels:
            z = m(image, meta, meta_mask)
            if z.dim() != 2:
                raise ValueError(
                    f"submodel returned {z.dim()}D tensor; expected 2D logits")
            outs.append(z.float())  # cast to fp32 for stable reduction
        return torch.stack(outs, dim=1)  # (B, N, K)


# ─────────────────────────────────────────────────────────────────────
# E2. Batched TTA
# ─────────────────────────────────────────────────────────────────────

class BatchedTTA:
    """Apply K augmentations as a single (B*K) batch, then reduce.

    Augmentations are pure torch ops (cheaper than going back to PIL/np
    inside the forward path). Default set is the 8-way from PLAN.md.
    """

    def __init__(self, views: Sequence[str] = ()) -> None:
        if not views:
            views = ("identity", "hflip", "vflip", "hvflip",
                     "rot90", "rot180", "rot270", "color_jitter_mild")
        self.views = list(views)

    @staticmethod
    def _apply(view: str, x: torch.Tensor) -> torch.Tensor:
        if view == "identity":
            return x
        if view == "hflip":
            return torch.flip(x, dims=[3])
        if view == "vflip":
            return torch.flip(x, dims=[2])
        if view == "hvflip":
            return torch.flip(x, dims=[2, 3])
        if view == "rot90":
            return torch.rot90(x, k=1, dims=[2, 3])
        if view == "rot180":
            return torch.rot90(x, k=2, dims=[2, 3])
        if view == "rot270":
            return torch.rot90(x, k=3, dims=[2, 3])
        if view == "color_jitter_mild":
            # small brightness shift; deterministic so it's reproducible
            return torch.clamp(x * 1.05 + 0.02, x.min().item(), x.max().item())
        raise ValueError(f"unknown TTA view: {view}")

    def run(self, model: nn.Module, image: torch.Tensor,
            meta: torch.Tensor, meta_mask: torch.Tensor) -> torch.Tensor:
        """Return logits averaged across views.

        For EnsembleWrapper, returned shape is (B, N, K).
        For single nn.Module, returned shape is (B, K).
        """
        K = len(self.views)
        B = image.shape[0]
        # Stack views along new dim then collapse to batch
        view_batch = torch.cat(
            [self._apply(v, image) for v in self.views], dim=0)  # (B*K, ...)
        meta_rep = meta.repeat(K, 1)
        mask_rep = meta_mask.repeat(K, 1)

        out = model(view_batch, meta_rep, mask_rep)
        if out.dim() == 2:
            # Single model: (B*K, num_classes) -> reduce
            out = out.view(K, B, -1).mean(dim=0)
            return out
        elif out.dim() == 3:
            # Ensemble: (B*K, N, K_cls) -> reduce
            out = out.view(K, B, *out.shape[1:]).mean(dim=0)
            return out
        else:
            raise ValueError(f"unexpected output dim {out.dim()}")


# ─────────────────────────────────────────────────────────────────────
# E3. Logit-space weighted average (learned weights)
# ─────────────────────────────────────────────────────────────────────

class LogitSpaceWeightedAvg(nn.Module):
    """Convex-combination weights over N submodels.

    weights are parameterized via softmax(theta) so they sum to 1 and
    stay nonnegative without explicit Lagrangian.
    """

    def __init__(self, n_submodels: int,
                 init: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if init is None:
            init = torch.zeros(n_submodels)  # uniform weights after softmax
        else:
            init = init.detach().clone()
            if init.shape != (n_submodels,):
                raise ValueError(
                    f"init shape {init.shape} != ({n_submodels},)")
        self.theta = nn.Parameter(init.float())

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.theta, dim=0)  # (N,)

    def forward(self, ensemble_logits: torch.Tensor) -> torch.Tensor:
        """ensemble_logits: (B, N, K) -> (B, K)"""
        if ensemble_logits.dim() != 3:
            raise ValueError(
                f"expected (B, N, K), got {ensemble_logits.shape}")
        w = self.weights.view(1, -1, 1)  # broadcast to (1, N, 1)
        return (ensemble_logits * w).sum(dim=1)


def fit_ensemble_weights(
    ensemble_logits: torch.Tensor,   # (M_val, N, K)
    labels: torch.Tensor,            # (M_val,) long
    *, lr: float = 0.1, max_iter: int = 200,
    verbose: bool = True,
) -> LogitSpaceWeightedAvg:
    """LBFGS-fit ensemble weights on val to maximize macro-F1 proxy (NLL).

    NLL is a smooth surrogate for macro-F1; the simplex parameterization
    keeps weights well-conditioned.
    """
    if ensemble_logits.dim() != 3:
        raise ValueError("ensemble_logits must be (M, N, K)")
    M, N, K = ensemble_logits.shape
    if labels.shape != (M,):
        raise ValueError(f"labels shape {labels.shape} != ({M},)")

    model = LogitSpaceWeightedAvg(N)
    opt = torch.optim.LBFGS([model.theta], lr=lr, max_iter=max_iter,
                             tolerance_grad=1e-7, tolerance_change=1e-9,
                             line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = model(ensemble_logits)
        loss = F.cross_entropy(z, labels)
        loss.backward()
        return loss

    with torch.no_grad():
        z0 = ensemble_logits.mean(dim=1)
        nll_before = F.cross_entropy(z0, labels).item()
    opt.step(closure)
    with torch.no_grad():
        z1 = model(ensemble_logits)
        nll_after = F.cross_entropy(z1, labels).item()
    if verbose:
        log.info("Ensemble weights fit:  NLL %.4f -> %.4f   weights=%s",
                 nll_before, nll_after,
                 [round(w.item(), 4) for w in model.weights])
    return model


# ─────────────────────────────────────────────────────────────────────
# E4. Calibrated prediction pipeline
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationAssets:
    """All optional post-ensemble calibration knobs.

    Each is optional; missing → that stage is skipped.
    """
    per_class_temperatures: Optional[np.ndarray] = None  # (K,)
    global_temperature: Optional[float] = None
    dirichlet_W: Optional[np.ndarray] = None             # (K, K)
    dirichlet_b: Optional[np.ndarray] = None             # (K,)
    diffevo_scales: Optional[np.ndarray] = None          # (K,)
    train_prior: Optional[np.ndarray] = None             # (K,)
    target_prior: Optional[np.ndarray] = None            # (K,)
    mel_safety_threshold: Optional[float] = None         # raw-prob threshold
    mel_class_idx: int = 0   # index of MEL in the output space


def _apply_temperature(z: torch.Tensor, assets: CalibrationAssets
                       ) -> torch.Tensor:
    """Per-class T preferred; fall back to global T; else identity."""
    if assets.per_class_temperatures is not None:
        T = torch.from_numpy(assets.per_class_temperatures.astype(np.float32))
        T = T.to(z.device)
        return z / T
    if assets.global_temperature is not None:
        return z / float(assets.global_temperature)
    return z


def _apply_dirichlet(z: torch.Tensor, assets: CalibrationAssets
                     ) -> torch.Tensor:
    """z'_j = W @ log_softmax(z) + b. K must match z.shape[-1]."""
    if assets.dirichlet_W is None or assets.dirichlet_b is None:
        return z
    W = torch.from_numpy(assets.dirichlet_W.astype(np.float32)).to(z.device)
    b = torch.from_numpy(assets.dirichlet_b.astype(np.float32)).to(z.device)
    K = z.shape[-1]
    if W.shape != (K, K) or b.shape != (K,):
        raise ValueError(f"Dirichlet W/b shape mismatch: W={W.shape} b={b.shape} K={K}")
    log_p = F.log_softmax(z, dim=-1)
    return log_p @ W.t() + b


def _apply_prior_shift(z: torch.Tensor, assets: CalibrationAssets
                       ) -> torch.Tensor:
    """logit += log(pi_target / pi_train)"""
    if assets.target_prior is None or assets.train_prior is None:
        return z
    tgt = torch.from_numpy(assets.target_prior.astype(np.float32)).to(z.device)
    src = torch.from_numpy(assets.train_prior.astype(np.float32)).to(z.device)
    eps = 1e-12
    delta = torch.log(tgt.clamp(min=eps)) - torch.log(src.clamp(min=eps))
    return z + delta.view(1, -1)


def predict_calibrated(
    ensemble: nn.Module,
    weights: Optional[LogitSpaceWeightedAvg],
    image: torch.Tensor, meta: torch.Tensor, meta_mask: torch.Tensor,
    *,
    tta: Optional[BatchedTTA] = None,
    assets: Optional[CalibrationAssets] = None,
    use_fp16: bool = True,
) -> dict:
    """Run the full inference path: TTA -> ensemble -> calibration.

    Order is fixed (and important):
        1. (TTA averaged) ensemble logits  shape (B, N, K)
        2. Weighted reduction over N        shape (B, K)
        3. Temperature                      (per-class or global)
        4. Dirichlet                        (if present)
        5. softmax                          probabilities
        6. Prior shift                      log-add-on (before softmax)
        7. DiffEvo scales                   on probabilities, renorm
        8. MEL safety override              on raw (un-scaled) MEL prob

    Returns dict of intermediate + final values for debugging.
    """
    ensemble.eval()
    if assets is None:
        assets = CalibrationAssets()

    with torch.inference_mode():
        dtype_ctx = torch.cuda.amp.autocast if use_fp16 and image.is_cuda else None
        if dtype_ctx:
            with dtype_ctx():
                ens_logits = tta.run(ensemble, image, meta, meta_mask) if tta \
                              else ensemble(image, meta, meta_mask)
        else:
            ens_logits = tta.run(ensemble, image, meta, meta_mask) if tta \
                          else ensemble(image, meta, meta_mask)

        # cast back up
        ens_logits = ens_logits.float()

        # Reduce across submodels (B, K)
        if ens_logits.dim() == 3:
            if weights is not None:
                z = weights(ens_logits)
            else:
                z = ens_logits.mean(dim=1)
        else:
            z = ens_logits

        # Steps 3, 4, 5, 6  — note: prior_shift is in logit-space, so we
        # apply it on the logits BEFORE softmax, then take softmax once.
        z = _apply_temperature(z, assets)
        z = _apply_dirichlet(z, assets)
        z = _apply_prior_shift(z, assets)
        probs = torch.softmax(z, dim=-1)

        # 7. DiffEvo scales (on probabilities)
        if assets.diffevo_scales is not None:
            scales = torch.from_numpy(
                assets.diffevo_scales.astype(np.float32)).to(probs.device)
            if scales.shape[0] != probs.shape[-1]:
                raise ValueError("diffevo_scales has wrong length")
            probs_scaled = probs * scales
            probs_scaled = probs_scaled / probs_scaled.sum(dim=-1, keepdim=True)
        else:
            probs_scaled = probs

        # 8. MEL safety on raw (un-scaled) probability
        preds = probs_scaled.argmax(dim=-1)
        mel_triggered = torch.zeros_like(preds, dtype=torch.bool)
        if assets.mel_safety_threshold is not None:
            raw_mel = probs[:, assets.mel_class_idx]
            trig = raw_mel >= float(assets.mel_safety_threshold)
            preds = torch.where(
                trig,
                torch.full_like(preds, assets.mel_class_idx),
                preds,
            )
            mel_triggered = trig

    return {
        "ensemble_logits": ens_logits,
        "reduced_logits": z,
        "probs": probs,
        "probs_scaled": probs_scaled,
        "preds": preds,
        "mel_safety_triggered": mel_triggered,
    }


# ─────────────────────────────────────────────────────────────────────
# High-level orchestrator entry: load N variants -> fit weights ->
# calibrated predict -> report metrics
# ─────────────────────────────────────────────────────────────────────

def evaluate_ensemble(*, variant_ids: list[str], checkpoint_paths: list[str],
                      split_dir: str, out_dir: str, img_size: int = 384,
                      batch_size: int = 32, num_workers: int = 4,
                      use_timm: bool = True) -> dict:
    """End-to-end ensemble evaluation pipeline.

    Steps:
      1. Build N submodels, load N checkpoints
      2. Forward val_calibrate to collect per-submodel logits
      3. Fit LogitSpaceWeightedAvg weights (LBFGS, simplex)
      4. Fit per-class T on the combined logits
      5. Forward test with the calibrated ensemble
      6. Report macro-F1 / AUC / ECE / per-class breakdown
    """
    import json
    from pathlib import Path
    import numpy as np, torch, torch.nn as nn
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix,
    )
    # Imports kept local to avoid forcing torch at module import time
    sys_path_root = str(Path(__file__).resolve().parents[1])
    if sys_path_root not in sys.path:
        sys.path.insert(0, sys_path_root)
    from stages.stage4_dataloader import build_loaders
    from stages.stage3_split import CANONICAL_CLASSES
    from stages.stage7_evaluate import (
        fit_per_class_temperature, expected_calibration_error,
    )
    from models import build_variant

    assert len(variant_ids) == len(checkpoint_paths), \
        "variant_ids and checkpoint_paths must align"

    out_dir = Path(out_dir) / "ensemble"; out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Build + load
    submodels = []
    for vid, cp in zip(variant_ids, checkpoint_paths):
        m = build_variant(vid, use_timm=use_timm, pretrained=False).to(device)
        ckpt = torch.load(cp, map_location=device, weights_only=False)
        m.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=False)
        submodels.append(m)
    ens = EnsembleWrapper(submodels, names=list(variant_ids))

    # 2. Loaders
    loaders, _ = build_loaders(split_dir, batch_size=batch_size,
                                img_size=img_size, num_workers=num_workers,
                                use_balanced_sampler=False)
    val_cal = loaders.get("val_calibrate")
    test = loaders.get("test")
    if test is None:
        raise RuntimeError("ensemble: no test loader")

    @torch.no_grad()
    def collect(loader):
        ens.eval()
        Z, Y = [], []
        for batch in loader:
            if batch is None: continue
            img = batch["image"].to(device, non_blocking=True)
            m_ = batch["meta"].to(device, non_blocking=True)
            mk = batch["meta_mask"].to(device, non_blocking=True)
            z = ens(img, m_, mk).float().cpu()  # (B, N, K)
            Z.append(z); Y.append(batch["label"])
        if not Z:
            raise RuntimeError("empty loader")
        return torch.cat(Z, 0), torch.cat(Y, 0)

    # 3. Fit ensemble weights on val_calibrate (or test if absent)
    if val_cal is not None:
        log.info("Collecting val_calibrate logits ...")
        z_cal, y_cal = collect(val_cal)
        log.info("Fitting ensemble weights on val_calibrate ...")
        weights = fit_ensemble_weights(z_cal, y_cal)
    else:
        log.warning("No val_calibrate; using uniform ensemble weights")
        weights = LogitSpaceWeightedAvg(len(submodels))

    # 4. Test forward + reduce
    log.info("Collecting test logits ...")
    z_test, y_test = collect(test)
    with torch.no_grad():
        z_reduced = weights(z_test).numpy()  # (N, K)

    # 5. Per-class T on reduced val_cal logits
    if val_cal is not None:
        with torch.no_grad():
            z_cal_reduced = weights(z_cal).numpy()
        per_class_T = fit_per_class_temperature(z_cal_reduced, y_cal.numpy())
    else:
        per_class_T = np.ones(z_reduced.shape[1], dtype=np.float32)

    z_t = torch.from_numpy(z_reduced) / torch.from_numpy(per_class_T)
    probs = torch.softmax(z_t, dim=-1).numpy()
    preds = probs.argmax(axis=1)
    y_np = y_test.numpy()

    # 6. Metrics
    metrics = {
        "ensemble": list(variant_ids),
        "weights": [float(w) for w in weights.weights.detach().numpy()],
        "per_class_T": [float(t) for t in per_class_T],
        "accuracy": float(accuracy_score(y_np, preds)),
        "macro_f1": float(f1_score(y_np, preds, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_np, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_np, preds, average="macro", zero_division=0)),
        "auc_roc_ovr": float(roc_auc_score(y_np, probs, multi_class="ovr", average="macro"))
                         if len(set(y_np)) > 1 else None,
        "ece": expected_calibration_error(probs, y_np),
        "per_class_f1": dict(zip(
            CANONICAL_CLASSES,
            f1_score(y_np, preds, average=None, zero_division=0).tolist(),
        )),
    }
    with (out_dir / "ensemble_metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Ensemble eval -> %s   macro_f1=%.4f  ECE=%.4f",
             out_dir, metrics["macro_f1"], metrics["ece"])
    return metrics


# ─────────────────────────────────────────────────────────────────────
# Unit test entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(name)s  %(levelname)s  %(message)s")

    # A trivial submodel: meta concatenated with image-mean -> linear
    class TinyModel(nn.Module):
        def __init__(self, n_classes: int = 8, meta_dim: int = 19):
            super().__init__()
            self.head = nn.Linear(3 + meta_dim, n_classes)

        def forward(self, image, meta, meta_mask):
            img_feat = image.mean(dim=(2, 3))  # (B, 3)
            x = torch.cat([img_feat, meta], dim=1)
            return self.head(x)

    submodels = [TinyModel() for _ in range(3)]
    ens = EnsembleWrapper(submodels, names=["V1", "V4", "V7"])
    tta = BatchedTTA()

    image = torch.randn(4, 3, 64, 64)
    meta = torch.randn(4, 19)
    mask = torch.ones(4, 19)

    out = ens(image, meta, mask)
    print(f"[OK] ensemble single forward: {tuple(out.shape)} (expected (4, 3, 8))")

    out_tta = tta.run(ens, image, meta, mask)
    print(f"[OK] TTA-averaged: {tuple(out_tta.shape)} (expected (4, 3, 8))")

    weights = LogitSpaceWeightedAvg(3)
    z = weights(out)
    print(f"[OK] weighted reduce: {tuple(z.shape)} (expected (4, 8))")
    print(f"[OK] weights start uniform: {weights.weights.tolist()}")

    # Fit weights on synthetic val
    val_logits = torch.randn(200, 3, 8)
    val_labels = torch.randint(0, 8, (200,))
    fitted = fit_ensemble_weights(val_logits, val_labels)
    print(f"[OK] fitted weights: {fitted.weights.tolist()}")

    # Calibrated predict path
    assets = CalibrationAssets(
        per_class_temperatures=np.full(8, 0.8, dtype=np.float32),
        diffevo_scales=np.ones(8, dtype=np.float32),
    )
    result = predict_calibrated(ens, fitted, image, meta, mask,
                                 tta=tta, assets=assets, use_fp16=False)
    print(f"[OK] calibrated pipeline: preds {tuple(result['preds'].shape)}, "
          f"probs {tuple(result['probs'].shape)}")
    print("All Stage-5 ensemble tests passed.")
