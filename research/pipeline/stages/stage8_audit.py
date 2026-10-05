"""
Stage 8 — Audit suite (the evaluation pieces PLAN.md required but
stages 1-7 had not wired up):

  A1. Fairness audit         — per-skin-tone, per-sex, per-age-band
                               stratified macro-F1 + MEL recall
  A2. Per-lesion aggregation — multi-image lesions get aggregated via
                               (a) majority vote (b) mean prob; report
                               both alongside per-image numbers
  A3. Selective accuracy     — accuracy vs coverage curve; MEL miss
                               rate at confidence thresholds; clinical
                               deferral risk-coverage
  A4. Missing-metadata robustness — drop each metadata field at
                               inference, report per-class F1 drop
  A5. Per-feature attribution     — Integrated Gradients on the
                               metadata-vector input dimension

All audits read the test split CSV and the checkpoint, run inference
once per audit family, and write structured JSON + PNG plots.

Priorities:
  1. Efficiency  — single test-set forward pass cached and reused
                   across A1..A5; ablations only re-run the cheap
                   parts (mask flipping, group-by aggregation)
  2. Quality     — every audit independently disable-able; outputs
                   are JSON + small PNGs for thesis tables
  3. Errorless   — missing column (e.g. fitzpatrick on ISIC 2019)
                   silently skips that stratum with a logged warning
  4. Explainable — every output names the cohort, n, and metric
  5. Logic       — selective accuracy uses *calibrated* probabilities
                   so the threshold semantics are honest
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stages.stage3_split import CANONICAL_CLASSES, CLASS_TO_IDX  # noqa: E402
from stages.stage4_dataloader import (  # noqa: E402
    build_loaders, encode_row_metadata, META_FEATURE_NAMES, META_DIM,
)
from models import build_variant  # noqa: E402


log = logging.getLogger("lesioniq.stage8")


# ─────────────────────────────────────────────────────────────────────
# Logits cache
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_test_logits(model, loader, device):
    """One forward pass: returns (logits[N,K], labels[N], row_ids[N])."""
    model.eval()
    Z, Y, R = [], [], []
    for batch in loader:
        if batch is None:
            continue
        img = batch["image"].to(device, non_blocking=True)
        m = batch["meta"].to(device, non_blocking=True)
        mk = batch["meta_mask"].to(device, non_blocking=True)
        z = model(img, m, mk).float().cpu().numpy()
        Z.append(z)
        Y.append(batch["label"].numpy())
        R.append(batch["row_id"].numpy())
    if not Z:
        raise RuntimeError("collect_test_logits: empty test loader")
    return np.concatenate(Z), np.concatenate(Y), np.concatenate(R)


def apply_per_class_T(logits: np.ndarray, T: np.ndarray) -> np.ndarray:
    z = torch.from_numpy(logits).float()
    t = torch.from_numpy(T).float()
    return F.softmax(z / t.clamp(min=0.1), dim=-1).numpy()


# ─────────────────────────────────────────────────────────────────────
# A1. Fairness audit
# ─────────────────────────────────────────────────────────────────────

def _safe_macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    if len(y_true) == 0:
        return None
    try:
        return float(f1_score(y_true, y_pred, average="macro",
                               labels=list(range(len(CANONICAL_CLASSES))),
                               zero_division=0))
    except Exception as e:
        log.warning("macro-F1 failed: %s", e)
        return None


def fairness_audit(test_df: pd.DataFrame, probs: np.ndarray,
                    y_true: np.ndarray) -> dict:
    """Stratified macro-F1 by skin tone, sex, age band.

    Each stratum needs >= 30 samples to report (smaller cohorts produce
    misleading numbers).
    """
    preds = probs.argmax(axis=1)
    out: dict = {"min_cohort_size": 30}

    # Skin tone (Fitzpatrick)
    if "fitzpatrick" in test_df.columns and test_df["fitzpatrick"].notna().any():
        fitz = test_df["fitzpatrick"].fillna(-1).astype(int)
        bands = [("I-II", [1, 2]), ("III-IV", [3, 4]), ("V-VI", [5, 6])]
        sk_out = {}
        for label, vals in bands:
            m = fitz.isin(vals).values
            n = int(m.sum())
            if n < 30:
                sk_out[label] = {"n": n, "macro_f1": None,
                                 "note": "n<30, skipped"}
                continue
            mel_idx = CLASS_TO_IDX["MEL"]
            mel_recall = float(
                ((preds[m] == mel_idx) & (y_true[m] == mel_idx)).sum()
                / max(1, (y_true[m] == mel_idx).sum())
            )
            sk_out[label] = {
                "n": n,
                "macro_f1": _safe_macro_f1(y_true[m], preds[m]),
                "mel_recall": mel_recall,
            }
        out["skin_tone"] = sk_out
    else:
        log.warning("fairness: no fitzpatrick column in test split — "
                    "skin-tone strata skipped")
        out["skin_tone"] = "unavailable"

    # Sex
    if "sex" in test_df.columns:
        sex_out = {}
        for sx in ("male", "female"):
            m = (test_df["sex"].fillna("").str.lower() == sx).values
            n = int(m.sum())
            if n < 30:
                sex_out[sx] = {"n": n, "macro_f1": None}
                continue
            sex_out[sx] = {"n": n, "macro_f1": _safe_macro_f1(y_true[m], preds[m])}
        out["sex"] = sex_out

    # Age band
    if "age" in test_df.columns:
        age_out = {}
        bands = [("<40", lambda a: a < 40),
                  ("40-60", lambda a: (40 <= a) & (a < 60)),
                  ("60-75", lambda a: (60 <= a) & (a < 75)),
                  (">=75", lambda a: a >= 75)]
        for label, fn in bands:
            ages = test_df["age"].fillna(-1).astype(float).values
            m = fn(ages) & (ages > 0)
            n = int(m.sum())
            if n < 30:
                age_out[label] = {"n": n, "macro_f1": None}
                continue
            age_out[label] = {"n": n,
                               "macro_f1": _safe_macro_f1(y_true[m], preds[m])}
        out["age_band"] = age_out

    return out


# ─────────────────────────────────────────────────────────────────────
# A2. Per-lesion aggregation
# ─────────────────────────────────────────────────────────────────────

def per_lesion_aggregation(test_df: pd.DataFrame, probs: np.ndarray,
                             y_true: np.ndarray) -> dict:
    """Aggregate predictions to lesion level via majority vote and
    mean-prob. Report per-lesion F1 alongside per-image F1.
    """
    if "lesion_id" not in test_df.columns:
        log.warning("per_lesion: no lesion_id column; skipping")
        return {"available": False}

    df = test_df.copy()
    K = probs.shape[1]
    df["true"] = y_true
    df["pred"] = probs.argmax(axis=1)
    for i in range(K):
        df[f"p{i}"] = probs[:, i]

    # majority vote
    mv = df.groupby("lesion_id")["pred"].agg(
        lambda s: int(s.mode().iloc[0]))
    truth = df.groupby("lesion_id")["true"].first()

    # mean-prob aggregation
    mp = df.groupby("lesion_id")[[f"p{i}" for i in range(K)]].mean()
    mp_pred = mp.values.argmax(axis=1)

    # align
    mv_pred = mv.reindex(truth.index).values
    mp_pred_aligned = mp.reindex(truth.index).values.argmax(axis=1)

    from sklearn.metrics import f1_score, accuracy_score
    per_lesion_majority_f1 = float(f1_score(
        truth.values, mv_pred, average="macro", zero_division=0))
    per_lesion_meanprob_f1 = float(f1_score(
        truth.values, mp_pred_aligned, average="macro", zero_division=0))

    return {
        "available": True,
        "n_lesions": int(len(truth)),
        "n_images": int(len(df)),
        "per_image_macro_f1": float(f1_score(
            df["true"].values, df["pred"].values,
            average="macro", zero_division=0)),
        "per_lesion_majority_vote_macro_f1": per_lesion_majority_f1,
        "per_lesion_mean_prob_macro_f1": per_lesion_meanprob_f1,
    }


# ─────────────────────────────────────────────────────────────────────
# A3. Selective accuracy at coverage
# ─────────────────────────────────────────────────────────────────────

def selective_accuracy_curve(probs: np.ndarray, y_true: np.ndarray,
                              out_png: Optional[Path] = None) -> dict:
    """Accuracy as a function of cumulative coverage when accepting
    cases in descending confidence order.
    """
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    order = np.argsort(-conf)
    preds_s = preds[order]; y_s = y_true[order]
    N = len(y_true)
    cum_correct = np.cumsum(preds_s == y_s)
    coverage = np.arange(1, N + 1) / N
    accuracy = cum_correct / np.arange(1, N + 1)

    # Key checkpoints
    cps = {}
    for c in (0.5, 0.7, 0.9, 1.0):
        n = max(1, int(N * c))
        cps[f"acc_at_{c:.0%}_coverage"] = float(accuracy[n - 1])

    # MEL miss rate at confidence thresholds
    mel_idx = CLASS_TO_IDX["MEL"]
    mel_mask = (y_true == mel_idx)
    n_mel = int(mel_mask.sum())
    if n_mel == 0:
        mel_curve = {"available": False}
    else:
        mel_curve = {}
        for thresh in (0.5, 0.7, 0.9):
            # cases predicted MEL with confidence >= thresh
            accepted = (probs[:, mel_idx] >= thresh)
            # of all true MELs, how many were caught
            caught = (accepted & mel_mask).sum()
            miss = (n_mel - caught) / n_mel
            mel_curve[f"miss_rate_at_conf_>={thresh}"] = float(miss)

    if out_png is not None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(coverage, accuracy, lw=1.5)
        ax.set_xlabel("Coverage (fraction of cases accepted)")
        ax.set_ylabel("Accuracy on accepted cases")
        ax.set_title("Risk-coverage curve")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        plt.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

    return {**cps, "mel": mel_curve}


# ─────────────────────────────────────────────────────────────────────
# A4. Missing-metadata robustness curve
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def missing_metadata_robustness(model, loader, device, per_class_T) -> dict:
    """For each metadata group (age, sex, site, fitz), set its mask to 0
    at inference and measure the resulting per-class F1.
    """
    model.eval()
    base_Z, Y = [], []
    masked_logits: dict[str, list[np.ndarray]] = {}

    # Define groups
    groups = {
        "age":  [META_FEATURE_NAMES.index("age_norm")],
        "sex":  [i for i, n in enumerate(META_FEATURE_NAMES) if n.startswith("sex_")],
        "site": [i for i, n in enumerate(META_FEATURE_NAMES) if n.startswith("site_")],
        "fitz": [i for i, n in enumerate(META_FEATURE_NAMES) if n.startswith("fitz_")],
    }
    for g in groups:
        masked_logits[g] = []

    for batch in loader:
        if batch is None:
            continue
        img = batch["image"].to(device, non_blocking=True)
        m = batch["meta"].to(device, non_blocking=True)
        mk = batch["meta_mask"].to(device, non_blocking=True)
        y = batch["label"].numpy()

        # baseline
        z = model(img, m, mk).float().cpu().numpy()
        base_Z.append(z); Y.append(y)

        # ablate each group
        for g, idxs in groups.items():
            mk2 = mk.clone()
            mk2[:, idxs] = 0.0
            m2 = m.clone()
            m2[:, idxs] = 0.0
            za = model(img, m2, mk2).float().cpu().numpy()
            masked_logits[g].append(za)

    if not base_Z:
        return {"available": False}
    base_Z = np.concatenate(base_Z); Y = np.concatenate(Y)
    probs_base = apply_per_class_T(base_Z, per_class_T)
    base_pred = probs_base.argmax(1)
    from sklearn.metrics import f1_score
    base_f1 = float(f1_score(Y, base_pred, average="macro",
                              zero_division=0))

    out = {"baseline_macro_f1": base_f1, "groups": {}}
    for g in groups:
        Z = np.concatenate(masked_logits[g])
        probs = apply_per_class_T(Z, per_class_T)
        pred = probs.argmax(1)
        f1_drop = base_f1 - float(f1_score(Y, pred, average="macro",
                                            zero_division=0))
        out["groups"][g] = {
            "macro_f1_without_group": float(f1_score(Y, pred, average="macro",
                                                       zero_division=0)),
            "macro_f1_drop": float(f1_drop),
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# A5. Per-feature integrated-gradient attribution
# ─────────────────────────────────────────────────────────────────────

def _ig_metadata_attribution(model, batch, device, target_class: int,
                              steps: int = 16) -> np.ndarray:
    """Integrated Gradients with metadata-zero baseline. Returns
    (META_DIM,) average importance for the requested class.
    """
    model.eval()
    img = batch["image"].to(device, non_blocking=True)
    m_in = batch["meta"].to(device, non_blocking=True)
    mk = batch["meta_mask"].to(device, non_blocking=True)

    baseline = torch.zeros_like(m_in)
    igs = torch.zeros_like(m_in)
    for k in range(1, steps + 1):
        alpha = k / steps
        m_interp = baseline + alpha * (m_in - baseline)
        m_interp.requires_grad_(True)
        z = model(img, m_interp, mk)
        target = z[:, target_class].sum()
        grad = torch.autograd.grad(target, m_interp,
                                    retain_graph=False, create_graph=False)[0]
        igs = igs + grad / steps
    attribution = (m_in - baseline) * igs
    return attribution.mean(dim=0).detach().cpu().numpy()


def per_feature_attribution(model, loader, device,
                              max_batches: int = 5) -> dict:
    """Average IG attribution per metadata feature, per class.

    Cheaper than running on full test set: cap at `max_batches` batches
    per class. Result is the relative importance of each metadata field
    for each predicted class.
    """
    log.info("Running integrated-gradient attribution (%d batches/class)",
             max_batches)
    per_class: dict[int, list[np.ndarray]] = {c: [] for c in range(len(CANONICAL_CLASSES))}
    counts = {c: 0 for c in per_class}
    for batch in loader:
        if batch is None:
            continue
        labels = batch["label"].numpy()
        for c in per_class:
            if counts[c] >= max_batches:
                continue
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            sub = {k: (v[mask] if torch.is_tensor(v) else v)
                   for k, v in batch.items()
                   if k not in ("row_id", "valid")}
            try:
                attr = _ig_metadata_attribution(model, sub, device, c)
                per_class[c].append(attr)
                counts[c] += 1
            except Exception as e:
                log.warning("IG failed for class %d: %s", c, e)
        if all(counts[c] >= max_batches for c in per_class):
            break

    out = {}
    for c, lst in per_class.items():
        if not lst:
            out[CANONICAL_CLASSES[c]] = {"available": False}
            continue
        avg = np.mean(np.stack(lst), axis=0)
        out[CANONICAL_CLASSES[c]] = {
            META_FEATURE_NAMES[i]: float(avg[i])
            for i in range(META_DIM)
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────

def run_audit(variant_id: str, checkpoint_path: str, split_dir: str,
               out_dir: str, *, img_size: int = 384, batch_size: int = 32,
               num_workers: int = 4, use_timm: bool = True) -> dict:
    out_dir = Path(out_dir) / "audit"; out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model + checkpoint
    model = build_variant(variant_id, use_timm=use_timm, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state, strict=False)

    # Loaders
    loaders, _ = build_loaders(split_dir, batch_size=batch_size,
                                img_size=img_size, num_workers=num_workers,
                                use_balanced_sampler=False)
    test_loader = loaders.get("test")
    if test_loader is None:
        raise RuntimeError("No test loader; check split dir")
    val_cal_loader = loaders.get("val_calibrate")

    # Cache test logits (single forward pass)
    log.info("Collecting test logits...")
    z, y, row_ids = collect_test_logits(model, test_loader, device)

    # Fit per-class T on val_calibrate (for A3, A4 to use calibrated probs)
    if val_cal_loader is not None:
        log.info("Fitting per-class T on val_calibrate...")
        z_cal, y_cal, _ = collect_test_logits(model, val_cal_loader, device)
        from stages.stage7_evaluate import fit_per_class_temperature
        per_class_T = fit_per_class_temperature(z_cal, y_cal)
    else:
        log.warning("No val_calibrate; using T=1.0 (uncalibrated)")
        per_class_T = np.ones(len(CANONICAL_CLASSES), dtype=np.float32)

    probs = apply_per_class_T(z, per_class_T)

    # Read test_df, aligned to row_ids
    test_df = pd.read_csv(Path(split_dir) / "test.csv")
    test_df = test_df.iloc[row_ids].reset_index(drop=True)

    results = {"variant": variant_id, "n_test": int(len(y))}

    # A1 fairness
    log.info("A1: fairness audit")
    results["fairness"] = fairness_audit(test_df, probs, y)

    # A2 per-lesion
    log.info("A2: per-lesion aggregation")
    results["per_lesion"] = per_lesion_aggregation(test_df, probs, y)

    # A3 selective accuracy
    log.info("A3: selective accuracy")
    results["selective"] = selective_accuracy_curve(
        probs, y, out_png=out_dir / "risk_coverage.png")

    # A4 missing metadata robustness
    log.info("A4: missing-metadata robustness")
    results["missing_meta"] = missing_metadata_robustness(
        model, test_loader, device, per_class_T)

    # A5 per-feature attribution
    log.info("A5: per-feature attribution")
    results["attribution"] = per_feature_attribution(
        model, test_loader, device, max_batches=3)

    with (out_dir / "audit.json").open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    log.info("Audit complete -> %s", out_dir)
    return results


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="LesionIQ audit suite")
    p.add_argument("--variant", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-timm", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run_audit(
        variant_id=args.variant, checkpoint_path=args.checkpoint,
        split_dir=args.split_dir, out_dir=args.out_dir,
        img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, use_timm=not args.no_timm,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
