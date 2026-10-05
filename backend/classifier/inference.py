"""
LesionIQ -- Inference + Explainability Pipeline
==================================================
5-stage pipeline: Input → Preprocess → Classify → Explain → SLM Output

Each image produces a diagnostic bundle:
  output/<image_name>/
    ├── final_preprocessed.png  # Model input (384×384)
    ├── gradcam.png         # Grad-CAM++ heatmap overlay
    ├── attention.png       # Swin attention rollout overlay
    └── diagnosis.json      # Full diagnostic data for SLM

Usage:
  python backend/inference.py --image lesion.png
  python backend/inference.py --image img.png --age 65 --sex male --site "head/neck"
  python backend/inference.py --image img.png --age NA --sex NA --site NA
  python backend/inference.py --dir path/to/images/ --output-dir ./results
"""

import os, sys, json, argparse
from functools import lru_cache
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------------
#  Resolve paths — add REPO_ROOT so `from preprocessing import ...` works
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
CKPT_DIR   = REPO_ROOT / "checkpoints"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from backend.classifier.config import AGE_NORM_FACTOR
except ImportError:
    AGE_NORM_FACTOR = 90.0

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_FULL  = {
    "MEL": "Melanoma", "NV": "Melanocytic Nevus", "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis", "BKL": "Benign Keratosis", "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion", "SCC": "Squamous Cell Carcinoma",
}
CLASS_THRESHOLDS = {
    "MEL": 0.57, "NV": 0.48, "BCC": 0.42, "BKL": 0.39,
    "AK": 0.31, "SCC": 0.35, "VASC": 0.22, "DF": 0.24,
}
MALIGNANT   = {"MEL", "BCC", "AK", "SCC"}
IMG_SIZE    = 384
NUM_CLASSES = 8
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

SEX_COLS  = ["sex_female", "sex_male", "sex_unknown"]
SITE_COLS = [
    "site_anterior torso", "site_head/neck", "site_lateral torso",
    "site_lower extremity", "site_oral/genital", "site_palms/soles",
    "site_posterior torso", "site_unknown", "site_upper extremity",
]
ALL_SITES     = [s.replace("site_", "") for s in SITE_COLS]
META_FEATURES = ["age_approx"] + SEX_COLS + SITE_COLS  # 13 features


# ===================================================================
#  Stage 1 — Interactive Metadata Input
# ===================================================================

def _prompt_metadata_interactive():
    """Prompt user for each metadata field with NA option."""
    print("\n╔══════════════════════════════════════╗")
    print("║        Patient Metadata Input        ║")
    print("╚══════════════════════════════════════╝\n")

    # --- Age ---
    while True:
        raw = input("  Age (years, or NA): ").strip()
        if raw.upper() == "NA" or raw == "":
            age = None; break
        try:
            age = float(raw)
            if 0 <= age <= 120: break
            print("    → Please enter a value between 0-120.")
        except ValueError:
            print("    → Enter a number or NA.")

    # --- Sex ---
    while True:
        raw = input("  Sex (male / female / NA): ").strip().lower()
        if raw in ("na", ""):
            sex = None; break
        if raw in ("male", "female"):
            sex = raw; break
        print("    → Enter male, female, or NA.")

    # --- Site ---
    site_options = [
        "anterior torso", "head/neck", "lateral torso", "lower extremity",
        "oral/genital", "palms/soles", "posterior torso", "upper extremity",
    ]
    print("\n  Anatomical site:")
    for i, s in enumerate(site_options, 1):
        print(f"    {i}. {s}")
    print(f"    9. NA (unknown)")

    while True:
        raw = input("  Select [1-9]: ").strip()
        if raw.upper() == "NA" or raw == "9" or raw == "":
            site = None; break
        try:
            idx = int(raw)
            if 1 <= idx <= 8:
                site = site_options[idx - 1]; break
            print("    → Enter 1-9.")
        except ValueError:
            print("    → Enter a number 1-9 or NA.")

    return age, sex, site


def encode_metadata(age=None, sex=None, site=None):
    """Encode patient metadata into a 13-d tensor matching training format."""
    meta = np.zeros(13, dtype=np.float32)
    meta[0] = min(max(age, 0.0), AGE_NORM_FACTOR) / AGE_NORM_FACTOR if age is not None else 0.5
    sex = (sex or "unknown").lower()
    if sex == "female":   meta[1] = 1.0
    elif sex == "male":   meta[2] = 1.0
    else:                 meta[3] = 1.0
    site = (site or "unknown").lower()
    matched = False
    for i, s in enumerate(ALL_SITES):
        if s == site:
            meta[4 + i] = 1.0
            matched = True
            break
    if not matched:
        meta[4 + ALL_SITES.index("unknown")] = 1.0
    return torch.tensor(meta).unsqueeze(0)


# ===================================================================
#  Stage 2 — Preprocessing (delegates to preprocessing package)
# ===================================================================

from backend.preprocessing import run_pipeline as _run_preprocess_pipeline
from backend.preprocessing.dull_razor import dullrazor as _dullrazor
from backend.preprocessing.shades_of_grey import shades_of_gray as _shades_of_gray
from backend.preprocessing.apply_clahe import apply_clahe as _apply_clahe
from backend.preprocessing.remove_circular_border import has_circular_border as _has_circular_border
from backend.preprocessing.remove_circular_border import inscribed_square as _inscribed_square
from backend.classifier.models import LesionIQHybrid


ARTIFACT_FILENAMES = {
    "raw": "raw.png",
    "dullrazor": "01_dullrazor.png",
    "shadesofgrey": "02_shades_of_grey.png",
    "clahe": "03_clahe.png",
    "borderremoved": "04_border_removed.png",
    "final_preprocessed": "final_preprocessed.png",
    "original": "final_preprocessed.png",
    "gradcam": "gradcam.png",
    "attention": "attention.png",
    "diagnosis": "diagnosis.json",
}


class PreprocessingArtifactError(RuntimeError):
    """Raised when a required case artifact cannot be produced."""


def _save_bgr(path, img_bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img_bgr is None:
        raise PreprocessingArtifactError(f"Cannot save empty image artifact: {path.name}")
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise PreprocessingArtifactError(f"Failed to write image artifact: {path}")
    return str(path)


def _require_image(stage, img_bgr):
    if img_bgr is None:
        raise PreprocessingArtifactError(f"Preprocessing stage failed: {stage}")
    return img_bgr


def run_preprocessing_artifacts(image_path, output_dir, target_size=IMG_SIZE):
    """Run preprocessing and save every auditable layer used by the UI/API."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    artifacts = {
        "raw": _save_bgr(output_dir / ARTIFACT_FILENAMES["raw"], raw_bgr)
    }

    try:
        dull_bgr, _ = _dullrazor(str(image_path))
        dull_bgr = _require_image("DullRazor hair removal", dull_bgr)
        artifacts["dullrazor"] = _save_bgr(
            output_dir / ARTIFACT_FILENAMES["dullrazor"], dull_bgr)

        sog_bgr = _require_image(
            "Shades-of-Gray normalization", _shades_of_gray(dull_bgr, power=4))
        artifacts["shadesofgrey"] = _save_bgr(
            output_dir / ARTIFACT_FILENAMES["shadesofgrey"], sog_bgr)

        clahe_input = output_dir / "_clahe_input.png"
        _save_bgr(clahe_input, sog_bgr)
        try:
            clahe_bgr = _require_image(
                "LAB CLAHE enhancement", _apply_clahe(str(clahe_input)))
        finally:
            clahe_input.unlink(missing_ok=True)
        artifacts["clahe"] = _save_bgr(
            output_dir / ARTIFACT_FILENAMES["clahe"], clahe_bgr)

        border_bgr = clahe_bgr
        circle = _has_circular_border(border_bgr)
        if circle is not None:
            cx, cy, radius = circle
            h, w = border_bgr.shape[:2]
            x1, y1, x2, y2 = _inscribed_square(cx, cy, radius, h, w)
            border_bgr = border_bgr[y1:y2, x1:x2]

        border_bgr = _require_image("Circular border removal", border_bgr)
        border_bgr = cv2.resize(
            border_bgr, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        artifacts["borderremoved"] = _save_bgr(
            output_dir / ARTIFACT_FILENAMES["borderremoved"], border_bgr)
        final_path = _save_bgr(
            output_dir / ARTIFACT_FILENAMES["final_preprocessed"], border_bgr)
    except Exception as exc:
        if isinstance(exc, PreprocessingArtifactError):
            raise
        raise PreprocessingArtifactError(str(exc)) from exc

    artifacts["final_preprocessed"] = final_path
    artifacts["original"] = final_path

    return border_bgr, artifacts


def preprocess_image(image_path):
    """Full Layer 0 pipeline + model normalization → tensor [1,3,384,384]."""
    img_bgr = _run_preprocess_pipeline(str(image_path), target_size=IMG_SIZE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transform(image=img_rgb)["image"].unsqueeze(0)


def get_display_image(image_path):
    """Preprocessed image as [0,1] RGB numpy for visualization overlays."""
    img_bgr = _run_preprocess_pipeline(str(image_path), target_size=IMG_SIZE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0


# ===================================================================
#  Stage 3a — Model (uses canonical LesionIQHybrid from models.py)
# ===================================================================

def build_model(mode="full", checkpoint_path=None):
    model = LesionIQHybrid(mode=mode, pretrained=False).to(DEVICE)
    if checkpoint_path is None:
        checkpoint_path = str(CKPT_DIR / f"best_{mode}.pt")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state_dict = {
        k.replace("module.", "", 1): v
        for k, v in state_dict.items()
        if hasattr(v, "shape")
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[WARN] Loaded checkpoint with relaxed key matching "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    # fp16 inference: halves VRAM (~6GB → ~3GB) with no accuracy loss at eval time
    if DEVICE == "cuda" and os.getenv("LESIONIQ_FP16", "1") != "0":
        model.half()
        print(f"[OK] Model loaded in fp16: {mode} from {checkpoint_path}")
    else:
        print(f"[OK] Model loaded: {mode} from {checkpoint_path}")
    return model


MODE_ALIASES = {
    "full": "full",
    "full hybrid": "full",
    "image_only": "image_only",
    "image only": "image_only",
    "effnet_only": "effnet_only",
    "effnet only": "effnet_only",
    "swin_only": "swin_only",
    "swin only": "swin_only",
}


def normalize_mode(mode="full"):
    """Normalize CLI/frontend mode labels into checkpoint mode names."""
    key = str(mode or "full").strip().lower().replace("-", "_")
    normalized = MODE_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unsupported inference mode: {mode}")
    return normalized


@lru_cache(maxsize=8)
def _load_runtime(mode="full", checkpoint_path=None,
                  use_scales=True, use_temperature=True, use_mel_safety=True):
    """Load model and calibration assets once for API/server reuse.

    Returns a 5-tuple:
        (model, scales, temperature, mel_threshold, per_class_temperatures)

    ``per_class_temperatures`` is a float32 numpy array of shape (8,) when
    ``backend/checkpoints/per_class_temperatures.npy`` exists, otherwise None.
    When present it supersedes the scalar ``temperature`` inside ``predict()``.
    The scalar is kept as a fallback and for logging.
    """
    mode = normalize_mode(mode)
    model = build_model(mode, checkpoint_path)

    scales = None
    if use_scales:
        scales_path = CKPT_DIR / "optimal_scales.npy"
        if scales_path.exists():
            scales = np.load(str(scales_path))
            print("[OK] DiffEvo threshold scales loaded")

    temperature = 1.0
    if use_temperature:
        temp_path = CKPT_DIR / "optimal_temperature.npy"
        if temp_path.exists():
            temperature = float(np.load(str(temp_path)))
            print(f"[OK] Global temperature scaling: T={temperature:.4f}")

    mel_threshold = None
    if use_mel_safety:
        mel_path = CKPT_DIR / "mel_safety_threshold.npy"
        if mel_path.exists():
            mel_threshold = float(np.load(str(mel_path)))
            print(f"[OK] MEL safety threshold: {mel_threshold:.3f}")

    per_class_temperatures = None
    if use_temperature:
        pc_path = CKPT_DIR / "per_class_temperatures.npy"
        if pc_path.exists():
            per_class_temperatures = np.load(str(pc_path)).astype(np.float32)
            print(f"[OK] Per-class temperature scaling loaded "
                  f"(mean T={per_class_temperatures.mean():.4f})")

    # Prior-shift adaptation is OPT-IN. Activated by LESIONIQ_ADAPT_PRIOR=sld|oracle.
    # Loads:
    #   effective_train_prior.npy   (the prior the model actually learned)
    #   target_prior_{mode}.npy     (the target prior computed offline, by
    #       backend.classifier.compute_effective_train_prior for the train
    #       side and either an SLD batch run or oracle class-count for the
    #       test side -- see prior_adaptation.py)
    # At inference, probs are corrected via prior_adaptation.adjust_probs_for_prior.
    # Default mode "none" keeps existing behavior bit-identical.
    prior_adapt_mode = os.getenv("LESIONIQ_ADAPT_PRIOR", "none").lower()
    train_prior = None
    target_prior = None
    if prior_adapt_mode in ("sld", "oracle"):
        tp_path = CKPT_DIR / "effective_train_prior.npy"
        target_path = CKPT_DIR / f"target_prior_{prior_adapt_mode}.npy"
        if tp_path.exists() and target_path.exists():
            train_prior  = np.load(str(tp_path)).astype(np.float32)
            target_prior = np.load(str(target_path)).astype(np.float32)
            print(f"[OK] Prior-shift adaptation active "
                  f"(mode={prior_adapt_mode}, target mean={target_prior.mean():.3f})")
        else:
            missing = []
            if not tp_path.exists():    missing.append(str(tp_path))
            if not target_path.exists(): missing.append(str(target_path))
            print(f"[WARN] LESIONIQ_ADAPT_PRIOR={prior_adapt_mode} but missing: "
                  f"{missing}. Falling back to no prior adjustment.")
            prior_adapt_mode = "none"

    return (model, scales, temperature, mel_threshold,
            per_class_temperatures, train_prior, target_prior, prior_adapt_mode)


# ===================================================================
#  Stage 3b — 2-way TTA prediction with temperature scaling
# ===================================================================

@torch.no_grad()
def predict(model, image_tensor, meta_tensor=None, temperature=1.0,
            scales=None, per_class_temperatures=None):
    """2-way TTA prediction with temperature calibration + DiffEvo scaling.

    Calibration precedence (highest to lowest):
        1. per_class_temperatures (8-d, one scalar per class) — preferred
        2. temperature (global scalar) — fallback when (1) is unavailable
    DiffEvo ``scales`` are applied after softmax in both cases.

    Uses horizontal flip only. autocast on CUDA for reduced activation memory.
    """
    image_tensor = image_tensor.to(DEVICE)
    if meta_tensor is not None:
        meta_tensor = meta_tensor.to(DEVICE)

    # Cast input to fp16 if model was loaded in fp16
    if DEVICE == "cuda" and next(model.parameters()).dtype == torch.float16:
        image_tensor = image_tensor.half()
        if meta_tensor is not None:
            meta_tensor = meta_tensor.half()

    def _fwd(x):
        out = model(x, meta_tensor)
        return out[0] if isinstance(out, tuple) else out

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if DEVICE == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with autocast_ctx:
        logits = (
            _fwd(image_tensor)
            + _fwd(torch.flip(image_tensor, dims=[3]))   # horizontal flip only
        ) / 2.0

    # Temperature scaling (applied before softmax)
    # Per-class temperatures take precedence over the global scalar.
    logits = logits.float()
    if per_class_temperatures is not None:
        temps_t = torch.from_numpy(per_class_temperatures).to(logits.device)
        logits  = logits / temps_t          # broadcast (1, K) / (K,)
    else:
        logits  = logits / temperature      # global scalar fallback

    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # DiffEvo threshold scaling (applied after softmax)
    if scales is not None:
        probs = probs * scales
        probs = probs / probs.sum()

    return probs


# ===================================================================
#  Stage 4a — Grad-CAM++ (EfficientNet-B4 branch)
# ===================================================================

class GradCAMPP:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, image, meta, class_idx=None):
        self.model.eval()
        image = image.clone()
        if next(self.model.parameters()).dtype == torch.float16:
            image = image.half()
            if meta is not None:
                meta = meta.half()
        image = image.requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(image, meta)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            score = logits[0, class_idx]
            self.model.zero_grad()
            score.backward()

        # Cast to float32 for numerically stable Grad-CAM++ math.
        # fp16 underflows on pow(2)/pow(3) leaving only padding-edge artifacts.
        grads = self.gradients[0].float()
        acts  = self.activations[0].float()
        alpha_num   = grads.pow(2)
        alpha_denom = 2.0 * grads.pow(2) + (acts * grads.pow(3)).sum(
            dim=(1, 2), keepdim=True)
        alpha_denom = torch.where(
            alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        alpha   = alpha_num / alpha_denom
        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def _get_effnet_target_layer(model):
    if hasattr(model, 'effnet'):
        return model.effnet.blocks[-1]
    return None


def _make_heatmap_overlay(image_np, cam, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """Overlay heatmap on [H,W,3] image in [0,1]. Returns [0,1] RGB."""
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = heatmap[:, :, ::-1].astype(np.float32) / 255.0
    return np.clip(alpha * heatmap + (1 - alpha) * image_np, 0, 1)


def _region_from_point(x_norm, y_norm):
    if 0.34 <= x_norm <= 0.66 and 0.34 <= y_norm <= 0.66:
        return "central lesion body"
    vertical = "upper" if y_norm < 0.34 else "lower" if y_norm > 0.66 else "mid"
    horizontal = "left" if x_norm < 0.34 else "right" if x_norm > 0.66 else "central"
    if vertical == "mid":
        return f"{horizontal} peripheral lesion border"
    if horizontal == "central":
        return f"{vertical} lesion border"
    return f"{vertical}-{horizontal} lesion border"


def _zone_family(region):
    if not region:
        return None
    if "central" in region or "body" in region:
        return "central"
    if "upper" in region:
        return "upper"
    if "lower" in region:
        return "lower"
    if "left" in region:
        return "left"
    if "right" in region:
        return "right"
    if "border" in region or "peripheral" in region:
        return "peripheral"
    return "non_specific"


def _summarize_heatmap(heatmap, label):
    """Produce conservative spatial evidence for the SLM prompt."""
    if heatmap is None:
        return {
            "available": False,
            "region": "not available",
            "description": f"{label} artifact was not generated.",
            "area_pct": None,
            "peak_pixel": None,
            "zone_family": None,
        }

    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.size == 0 or not np.isfinite(arr).any():
        return {
            "available": False,
            "region": "not available",
            "description": f"{label} artifact was not interpretable.",
            "area_pct": None,
            "peak_pixel": None,
            "zone_family": None,
        }

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - float(arr.min())
    max_val = float(arr.max())
    if max_val > 0:
        arr = arr / max_val

    h, w = arr.shape[:2]
    peak_y, peak_x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    x_norm = peak_x / max(w - 1, 1)
    y_norm = peak_y / max(h - 1, 1)
    region = _region_from_point(x_norm, y_norm)
    active = arr >= max(0.55, float(np.percentile(arr, 85)))
    area_pct = round(float(active.mean() * 100.0), 1)
    peak_pixel = [
        int(round(x_norm * (IMG_SIZE - 1))),
        int(round(y_norm * (IMG_SIZE - 1))),
    ]
    prefix = "Grad-CAM++" if label == "gradcam" else "SwinV2 attention"

    return {
        "available": True,
        "region": region,
        "description": f"{prefix} concentrates on the {region}.",
        "area_pct": area_pct,
        "peak_pixel": peak_pixel,
        "zone_family": _zone_family(region),
    }


def _confidence_level(confidence):
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.5:
        return "moderate"
    return "low"


def build_feature_hints(prediction, gradcam_summary, attention_summary, metadata):
    """Convert raw model evidence into conservative clinical feature hints."""
    prediction = prediction or {}
    gradcam_summary = gradcam_summary or {}
    attention_summary = attention_summary or {}
    metadata = metadata or {}

    class_code = str(prediction.get("class", ""))
    confidence = float(prediction.get("confidence") or 0.0)
    threshold = CLASS_THRESHOLDS.get(class_code, 0.5)
    grad_zone = gradcam_summary.get("zone_family")
    attn_zone = attention_summary.get("zone_family")
    grad_region = gradcam_summary.get("region")

    if grad_zone and attn_zone and grad_zone == attn_zone:
        evidence_alignment = "aligned"
    elif grad_zone and attn_zone and (
        "central" in {grad_zone, attn_zone} or "peripheral" in {grad_zone, attn_zone}
    ):
        evidence_alignment = "partially_aligned"
    elif grad_zone and attn_zone:
        evidence_alignment = "discordant"
    else:
        evidence_alignment = "partially_aligned"

    primary_feature = "non-specific localisation"
    secondary_features = []
    if class_code == "VASC" and (gradcam_summary.get("available") or attention_summary.get("available")):
        primary_feature = "vascular emphasis"
    elif grad_region and "border" in grad_region:
        primary_feature = "border irregularity"
        secondary_features.append("asymmetry")
    elif grad_region and "central" in grad_region and class_code in {"BCC", "SCC"}:
        primary_feature = "central nodule"
    elif grad_region and "central" in grad_region:
        primary_feature = "lesion body"

    if evidence_alignment == "aligned":
        secondary_features.append("convergent image attribution")
    elif evidence_alignment == "discordant":
        secondary_features.append("discordant image attribution")

    age = metadata.get("age") or metadata.get("ageYears")
    sex = str(metadata.get("sex") or "").lower()
    site = str(metadata.get("site") or metadata.get("anatomicalSite") or "").lower()
    clinical_context = "metadata non-contributory"
    try:
        age_value = float(age) if age is not None else None
    except (TypeError, ValueError):
        age_value = None
    if site in {"head/neck", "upper extremity"} and class_code in MALIGNANT:
        clinical_context = "sun-exposed site increases suspicion"
    elif age_value is not None and age_value < 35 and sex == "female" and "torso" in site:
        clinical_context = "torso site in young female slightly lowers concern"

    uncertainty_flags = []
    if confidence < 0.5:
        uncertainty_flags.append("low_confidence")
    if evidence_alignment == "discordant":
        uncertainty_flags.append("discordant_attention")
    if abs(confidence - threshold) <= 0.08:
        uncertainty_flags.append("borderline_threshold")

    return {
        "primary_feature": primary_feature,
        "secondary_features": sorted(set(secondary_features)),
        "evidence_alignment": evidence_alignment,
        "clinical_context": clinical_context,
        "uncertainty_flags": uncertainty_flags,
        "area_pct": gradcam_summary.get("area_pct"),
        "peak_pixel": gradcam_summary.get("peak_pixel"),
    }


def build_slm_payload(diagnosis, gradcam_summary=None, attention_summary=None, metadata=None):
    """Build the strict evidence packet sent to the local SLM."""
    diagnosis = diagnosis or {}
    prediction = diagnosis.get("prediction", {})
    class_code = str(prediction.get("class", ""))
    confidence = float(prediction.get("confidence") or 0.0)
    threshold = CLASS_THRESHOLDS.get(class_code, 0.5)
    metadata = metadata or diagnosis.get("metadata_input", {})
    explainability = diagnosis.get("explainability", {})
    gradcam_summary = gradcam_summary or explainability.get("gradcam_summary") or {}
    attention_summary = attention_summary or explainability.get("attention_summary") or {}
    feature_hints = build_feature_hints(
        prediction, gradcam_summary, attention_summary, metadata)

    return {
        "prediction": {
            "class_code": class_code,
            "class_label": prediction.get("class_full") or CLASS_FULL.get(class_code, class_code),
            "confidence": round(confidence, 4),
            "confidence_pct": round(confidence * 100.0, 1),
            "confidence_level": _confidence_level(confidence),
            "threshold": threshold,
            "threshold_margin": round(confidence - threshold, 4),
            "is_malignant": bool(prediction.get("is_malignant")),
        },
        "probabilities": diagnosis.get("probabilities", {}),
        "top3": diagnosis.get("top3", []),
        "gradcam": gradcam_summary,
        "attention": attention_summary,
        "metadata": metadata,
        "feature_hints": feature_hints,
        "clinical_flags": diagnosis.get("clinical_flags", {}),
        "instructions": {
            "audience": "general physician or dermatologist",
            "scope": "preliminary decision support explanation only",
            "forbidden": [
                "independent diagnosis",
                "treatment directive",
                "biopsy directive",
                "reassurance claim",
                "unsupported dermoscopic features",
            ],
        },
    }


def build_slm_prompt(payload):
    """Strict report prompt for Gemma/Ollama."""
    pred = payload.get("prediction", {})
    top3 = payload.get("top3", [])
    hints = payload.get("feature_hints", {})
    gradcam = payload.get("gradcam", {})
    attention = payload.get("attention", {})
    metadata = payload.get("metadata", {})
    flags = payload.get("clinical_flags", {})

    # Pre-compute descriptive context so the LLM reasons over specifics
    class_label = pred.get("class_label", "Unknown")
    class_code = pred.get("class_code", "NA")
    conf = pred.get("confidence_pct", 0)
    conf_level = pred.get("confidence_level", "low")
    threshold = pred.get("threshold", 0.5)
    margin = pred.get("threshold_margin", 0)
    is_mal = pred.get("is_malignant", False)

    alternatives = ", ".join(
        f"{t.get('full_name', t.get('class', '?'))} ({round(t.get('probability', 0) * 100, 1)}%)"
        for t in top3[1:3]
    ) if len(top3) > 1 else "none above 5%"

    gc_region = gradcam.get("region", "not available")
    gc_area = gradcam.get("area_pct")
    at_region = attention.get("region", "not available")
    alignment = hints.get("evidence_alignment", "unknown")
    primary_feat = hints.get("primary_feature", "non-specific localisation")
    uncertainty = hints.get("uncertainty_flags") or []
    clinical_ctx = hints.get("clinical_context", "metadata non-contributory")

    age = metadata.get("age") or metadata.get("ageYears") or "unknown"
    sex = metadata.get("sex") or "unknown"
    site = metadata.get("site") or metadata.get("anatomicalSite") or "unknown"
    mal_total = flags.get("malignant_total_prob")

    return (
        "You are an expert clinical dermatology reviewer inside LesionIQ. "
        "Write a preliminary physician-facing reasoning report. "
        "Your audience is a physician who can ALREADY see the Grad-CAM overlay, "
        "the attention heatmap, and the predicted class. Do NOT restate what they see. "
        "Instead, EXPLAIN the reasoning: why the evidence supports (or challenges) the "
        "predicted diagnosis, what the spatial patterns suggest dermoscopically, and "
        "where the model's confidence may be unreliable.\n\n"

        "SPECIFIC EVIDENCE TO REASON OVER:\n"
        f"• Prediction: {class_label} ({class_code}) at {conf}% ({conf_level} confidence)\n"
        f"• Threshold: {threshold} — margin {'+' if margin >= 0 else ''}{round(margin * 100, 1)} pts\n"
        f"• Malignant flag: {'yes' if is_mal else 'no'}"
        f"{f' (combined malignant probability {round(mal_total * 100, 1)}%)' if mal_total else ''}\n"
        f"• Closest differentials: {alternatives}\n"
        f"• Grad-CAM++ peak region: {gc_region}"
        f"{f' covering {gc_area}% of image' if gc_area else ''}\n"
        f"• SwinV2 attention peak: {at_region}\n"
        f"• Evidence alignment between branches: {alignment}\n"
        f"• Inferred primary feature: {primary_feat}\n"
        f"• Patient: age {age}, sex {sex}, site {site}\n"
        f"• Clinical context: {clinical_ctx}\n"
        f"• Uncertainty flags: {', '.join(uncertainty) if uncertainty else 'none'}\n\n"

        "REASONING REQUIREMENTS:\n"
        "1. Start with the key question: WHY does this pattern look like the predicted class? "
        "Reference spatial evidence (where the heatmap focuses and what that could mean dermoscopically).\n"
        "2. If the two branches (Grad-CAM, SwinV2) disagree, explain what that implies for confidence.\n"
        "3. Discuss the differential: why the predicted class wins over the closest alternative, "
        "or why the margin is uncomfortably narrow.\n"
        "4. Use age/sex/site ONLY if they materially affect the reasoning (e.g., sun-exposed site "
        "for actinic keratosis, young female torso for benign nevi).\n"
        "5. End with an honest uncertainty note: what the physician should look for on dermoscopy "
        "that the model cannot assess.\n\n"

        "FORBIDDEN:\n"
        "- Do not recommend biopsy, treatment, discharge, or reassurance as a directive.\n"
        "- Do not fabricate dermoscopic features not supported by the supplied evidence.\n"
        "- Do not restate the prediction, confidence, or threshold — the physician already has those.\n\n"

        "Return exactly this schema:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "LESIONIQ CLINICAL EXPLAINABILITY REPORT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "PREDICTION\n"
        f"  Diagnosis   : {class_label} ({class_code})\n"
        f"  Confidence  : {conf}% ({conf_level} confidence)\n"
        f"  Threshold   : {threshold} (tuned — default 0.50)\n\n"
        "EVIDENCE\n"
        "  <five to seven sentences of clinical reasoning as described above>\n\n"
        f"Evidence packet JSON:\n{json.dumps(payload, sort_keys=True)}"
    )


def _fallback_slm_report(payload):
    pred = payload.get("prediction", {})
    hints = payload.get("feature_hints", {})
    gradcam = payload.get("gradcam", {})
    attention = payload.get("attention", {})
    top3 = payload.get("top3", [])
    flags = payload.get("clinical_flags", {})
    metadata = payload.get("metadata", {})
    alignment = hints.get("evidence_alignment", "partially_aligned")
    uncertainty = hints.get("uncertainty_flags") or []

    class_label = pred.get("class_label", "the top class")
    class_code = pred.get("class_code", "NA")
    conf_pct = pred.get("confidence_pct", 0)
    conf_level = pred.get("confidence_level", "low")
    threshold = pred.get("threshold", 0.5)
    is_mal = pred.get("is_malignant", False)

    gc_region = gradcam.get("region", "not available")
    gc_area = gradcam.get("area_pct")
    at_region = attention.get("region", "not available")
    primary_feat = hints.get("primary_feature", "non-specific localisation")
    clinical_ctx = hints.get("clinical_context", "metadata non-contributory")

    # Build reasoning paragraphs
    evidence = []

    # 1. WHY this class — spatial reasoning
    spatial = f"The model favours {class_label} because "
    if gradcam.get("available") and gc_region != "not available":
        spatial += f"Grad-CAM++ activation concentrates on the {gc_region}"
        if gc_area:
            spatial += f" ({gc_area}% coverage)"
        spatial += f", suggesting the key discriminative signal is {primary_feat}."
    else:
        spatial += f"the calibrated probability of {conf_pct}% exceeds the class threshold, though spatial evidence was not generated for this mode."
    evidence.append(spatial)

    # 2. Branch agreement
    if gradcam.get("available") and attention.get("available"):
        if alignment == "aligned":
            evidence.append(
                f"Both EfficientNet (Grad-CAM++) and SwinV2 (attention) branches converge on the {gc_region}, "
                "providing corroborative spatial support for the predicted class."
            )
        elif alignment == "discordant":
            evidence.append(
                f"The EfficientNet branch highlights the {gc_region} while SwinV2 attention peaks at the "
                f"{at_region} — this discordance reduces diagnostic certainty and warrants closer dermoscopic inspection."
            )
        else:
            evidence.append(
                f"The two branches show partial overlap: Grad-CAM++ peaks at the {gc_region} and "
                f"SwinV2 at the {at_region}; the evidence is not fully convergent."
            )

    # 3. Differential context
    if len(top3) > 1:
        alt = top3[1]
        alt_name = alt.get("full_name", alt.get("class", "?"))
        alt_prob = round(alt.get("probability", 0) * 100, 1)
        margin = conf_pct - alt_prob
        if margin < 15:
            evidence.append(
                f"The closest differential is {alt_name} at {alt_prob}% — only {round(margin, 1)} pts below the "
                f"primary prediction. This narrow separation means clinical correlation is especially important."
            )
        else:
            evidence.append(
                f"The nearest alternative, {alt_name} ({alt_prob}%), trails by {round(margin, 1)} pts, "
                "providing reasonable separation from the primary prediction."
            )

    # 4. Malignancy flag + metadata
    if is_mal:
        mal_note = f"The lesion classifies as malignant"
        mal_total = flags.get("malignant_total_prob")
        if mal_total:
            mal_note += f" with a combined malignant probability of {round(mal_total * 100, 1)}%"
        mal_note += "."
        if clinical_ctx != "metadata non-contributory":
            mal_note += f" Metadata context: {clinical_ctx}."
        evidence.append(mal_note)
    elif clinical_ctx != "metadata non-contributory":
        evidence.append(f"Clinical context: {clinical_ctx}.")

    # 5. Uncertainty + verification
    if uncertainty:
        evidence.append(
            f"Uncertainty flags raised: {', '.join(f.replace('_', ' ') for f in uncertainty)}. "
            "Clinician should verify with direct dermoscopic examination of structural features "
            "(pigment network, vascular patterns, symmetry) that the model cannot reliably assess."
        )
    else:
        evidence.append(
            "No specific uncertainty flags were raised, but clinician verification remains required — "
            "the model does not assess dermoscopic structures such as pigment network regularity, "
            "vascular patterns, or ulceration directly."
        )

    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "DERMALENS CLINICAL EXPLAINABILITY REPORT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "PREDICTION\n"
        f"  Diagnosis   : {class_label} ({class_code})\n"
        f"  Confidence  : {conf_pct}% ({conf_level} confidence)\n"
        f"  Threshold   : {threshold} (tuned — default 0.50)\n\n"
        "EVIDENCE\n"
        f"  {' '.join(evidence)}"
    )


def validate_and_repair_slm_output(text, payload):
    """Ensure the SLM report is usable before exposing it as slmSummary."""
    text = (text or "").strip()
    required = [
        "CLINICAL EXPLAINABILITY REPORT",
        "PREDICTION",
        "Diagnosis",
        "Confidence",
        "Threshold",
        "EVIDENCE",
    ]
    forbidden = ["[", "]", "<", ">"]
    unsafe_directives = [
        "you should biopsy",
        "biopsy is required",
        "no follow-up needed",
        "definitively benign",
    ]
    lower = text.lower()
    if (
        not text or
        any(item not in text for item in required) or
        any(token in text for token in forbidden) or
        any(phrase in lower for phrase in unsafe_directives)
    ):
        return _fallback_slm_report(payload)

    return text


# ===================================================================
#  Stage 4b — Swin Attention Rollout
# ===================================================================

def _extract_swin_attention(model, image):
    """Gradient-based spatial attribution for the Swin branch.

    SwinV2 uses windowed attention, which makes global attention rollout
    infeasible.  Instead we hook the last feature map before pooling and
    compute gradient × activation (Grad-AM) with respect to the predicted
    class.  This produces a clean spatial heatmap that highlights the
    regions the Swin branch relies on most.
    """
    if not hasattr(model, 'swin'):
        return None

    feature_map = {}

    def _fwd_hook(module, inp, out):
        out.retain_grad()
        feature_map['feat'] = out

    # Hook the norm layer right before the global average pool
    hook = model.swin.norm.register_forward_hook(_fwd_hook)

    # Need gradients for this pass
    img = image.clone()
    if next(model.swin.parameters()).dtype == torch.float16:
        img = img.half()
    img = img.requires_grad_(True)
    with torch.enable_grad():
        logits = model.swin(img)
        pred_idx = logits.argmax(dim=1).item()
        score = logits[0, pred_idx]
        model.swin.zero_grad()
        score.backward()

    hook.remove()

    feat = feature_map.get('feat')
    if feat is None:
        return None

    # feat shape: [1, H, W, C] for SwinV2 (spatial, channels last)
    grad = feat.grad[0]      # [H, W, C] or [N, C]
    act  = feat.detach()[0]

    # Grad-AM: element-wise gradient × activation, summed over channels
    spatial = (grad * act).sum(dim=-1)    # [H, W] or [N]
    spatial = F.relu(spatial)             # keep positive attributions
    spatial = spatial.cpu().numpy()

    # Handle both [H, W] (4D norm output) and [N] (3D flattened) cases
    if spatial.ndim == 1:
        side = int(np.sqrt(spatial.shape[0]))
        spatial = spatial[:side * side].reshape(side, side)

    spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8)
    return spatial


# ===================================================================
#  Stage 4c — SHAP (perturbation-based metadata attribution)
# ===================================================================

def _compute_shap_values(model, image_tensor, meta_tensor):
    """Perturbation-based feature attribution for metadata features.

    All 13 perturbed meta tensors are batched into a single forward pass
    (down from 13 sequential passes) for ~10x speedup on this stage.
    """
    if meta_tensor is None:
        return None

    model.eval()
    image_t = image_tensor.to(DEVICE)
    meta_t  = meta_tensor.to(DEVICE)
    if next(model.parameters()).dtype == torch.float16:
        image_t = image_t.half()
        meta_t = meta_t.half()
    n_feats = meta_t.shape[1]

    # Build batch: [base + N perturbations, 13] meta, [base + N, 3, H, W] image
    perturbed_metas = [meta_t.clone()]  # index 0 = baseline (unperturbed)
    for i in range(n_feats):
        p = meta_t.clone()
        p[0, i] = 0.0
        perturbed_metas.append(p)

    batched_meta  = torch.cat(perturbed_metas, dim=0)                       # [14, 13]
    batched_image = image_t.expand(len(perturbed_metas), -1, -1, -1)        # [14, 3, H, W]

    with torch.no_grad():
        logits = model(batched_image, batched_meta)
        probs  = torch.softmax(logits.float(), dim=1).cpu().numpy()          # [14, 8]

    base_probs = probs[0]
    pred_class = base_probs.argmax()

    shap_values = {}
    for i, feat_name in enumerate(META_FEATURES):
        delta = float(base_probs[pred_class] - probs[i + 1][pred_class])
        shap_values[feat_name] = round(delta, 6)

    return shap_values


# ===================================================================
#  Stage 5 — Full diagnostic pipeline (per image)
# ===================================================================



def diagnose_image(model, image_path, meta_tensor, scales, temperature,
                   output_dir, raw_meta, mel_threshold=None,
                   per_class_temperatures=None,
                   train_prior=None, target_prior=None, prior_adapt_mode="none"):
    """Run the full 5-stage pipeline on a single image.

    Produces a case folder with raw, preprocessing, explainability, and
    diagnosis.json artifacts. final_preprocessed.png is the model input.
    """
    stem       = Path(image_path).stem
    img_out_dir = Path(output_dir) / stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # -- Preprocess --
    final_bgr, artifact_paths = run_preprocessing_artifacts(
        image_path, img_out_dir, IMG_SIZE)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    img_display = final_rgb.astype(np.float32) / 255.0
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    img_tensor = transform(image=final_rgb)["image"].unsqueeze(0)

    # -- Classify (TTA + temperature calibration + DiffEvo) --
    probs = predict(model, img_tensor, meta_tensor, temperature, scales,
                    per_class_temperatures)

    # Prior-shift adaptation (opt-in). Applied AFTER calibration + DiffEvo
    # but BEFORE MEL safety, so MEL safety still triggers on raw model
    # confidence rather than adjusted confidence.
    prior_adapt_applied = False
    if (prior_adapt_mode in ("sld", "oracle")
            and train_prior is not None and target_prior is not None):
        from backend.classifier.prior_adaptation import adjust_probs_for_prior
        probs = adjust_probs_for_prior(probs[np.newaxis, :], train_prior, target_prior)[0]
        prior_adapt_applied = True

    # MEL safety override: if raw MEL prob exceeds threshold, force MEL
    mel_safety_applied = False
    if mel_threshold is not None:
        raw_mel_prob = float(probs[CLASS_NAMES.index('MEL')])
        if raw_mel_prob >= mel_threshold:
            mel_safety_applied = True

    ranked  = np.argsort(probs)[::-1]
    if mel_safety_applied:
        pred_cls = 'MEL'
        pred_conf = float(probs[CLASS_NAMES.index('MEL')])
    else:
        pred_cls  = CLASS_NAMES[ranked[0]]
        pred_conf = float(probs[ranked[0]])
    is_mal    = pred_cls in MALIGNANT

    # -- Grad-CAM++ --
    gradcam_file = None
    gradcam_summary = _summarize_heatmap(None, "gradcam")
    if hasattr(model, 'effnet'):
        target_layer = _get_effnet_target_layer(model)
        if target_layer is not None:
            gcpp = GradCAMPP(model, target_layer)
            cam  = gcpp.generate(
                img_tensor.to(DEVICE),
                meta_tensor.to(DEVICE) if meta_tensor is not None else None,
                class_idx=int(ranked[0]))
            gradcam_summary = _summarize_heatmap(cam, "gradcam")
            overlay = _make_heatmap_overlay(img_display, cam, cv2.COLORMAP_JET)
            gradcam_file = ARTIFACT_FILENAMES["gradcam"]
            artifact_paths["gradcam"] = str(img_out_dir / gradcam_file)
            cv2.imwrite(str(img_out_dir / gradcam_file),
                        cv2.cvtColor((overlay * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))

    # -- Swin Attention --
    attention_file = None
    attention_summary = _summarize_heatmap(None, "attention")
    if hasattr(model, 'swin'):
        attn_map = _extract_swin_attention(model, img_tensor.to(DEVICE))
        if attn_map is not None:
            attention_summary = _summarize_heatmap(attn_map, "attention")
            overlay = _make_heatmap_overlay(img_display, attn_map,
                                           cv2.COLORMAP_INFERNO)
            attention_file = ARTIFACT_FILENAMES["attention"]
            artifact_paths["attention"] = str(img_out_dir / attention_file)
            cv2.imwrite(str(img_out_dir / attention_file),
                        cv2.cvtColor((overlay * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))

    # -- SHAP (metadata) --
    shap_vals = None
    if meta_tensor is not None:
        shap_vals = _compute_shap_values(model, img_tensor, meta_tensor)

    # -- Build diagnosis.json (SLM input) --
    diagnosis = {
        "version": "1.0",
        "image": Path(image_path).name,
        "model": {
            "mode": model.mode,
            # "temperature" always a scalar for frontend compat; when per-class
            # temperatures are active we report their mean here.
            "temperature": round(
                float(np.mean(per_class_temperatures)) if per_class_temperatures is not None
                else temperature, 4),
            "calibration_mode": (
                "per_class" if per_class_temperatures is not None else "global"),
            "thresholds_applied": scales is not None,
            "mel_safety_threshold": round(mel_threshold, 3) if mel_threshold else None,
            "mel_safety_triggered": mel_safety_applied,
            "prior_adapt_mode": prior_adapt_mode,
            "prior_adapt_applied": prior_adapt_applied,
        },
        "prediction": {
            "class": pred_cls,
            "class_full": CLASS_FULL[pred_cls],
            "confidence": round(pred_conf, 4),
            "is_malignant": is_mal,
        },
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]), 4) for i in ranked
        },
        "top3": [
            {"class": CLASS_NAMES[ranked[i]],
             "full_name": CLASS_FULL[CLASS_NAMES[ranked[i]]],
             "probability": round(float(probs[ranked[i]]), 4),
             "is_malignant": CLASS_NAMES[ranked[i]] in MALIGNANT}
            for i in range(3)
        ],
        "clinical_flags": {
            "malignant_total_prob": round(float(
                sum(probs[CLASS_NAMES.index(c)] for c in MALIGNANT)), 4),
            "requires_biopsy": is_mal and pred_conf >= 0.5,
            "low_confidence": pred_conf < 0.5,
            "differential_diagnosis": pred_conf < 0.7,
        },
        "explainability": {
            "gradcam": gradcam_file,
            "gradcam_summary": gradcam_summary,
            "attention": attention_file,
            "attention_summary": attention_summary,
            "shap_metadata": shap_vals,
        },
        "artifacts": {
            "raw": ARTIFACT_FILENAMES["raw"],
            "dullrazor": ARTIFACT_FILENAMES["dullrazor"],
            "shadesofgrey": ARTIFACT_FILENAMES["shadesofgrey"],
            "clahe": ARTIFACT_FILENAMES["clahe"],
            "borderremoved": ARTIFACT_FILENAMES["borderremoved"],
            "final_preprocessed": ARTIFACT_FILENAMES["final_preprocessed"],
            "original": ARTIFACT_FILENAMES["original"],
            "gradcam": gradcam_file,
            "attention": attention_file,
            "diagnosis": ARTIFACT_FILENAMES["diagnosis"],
        },
        "metadata_input": {
            "age": raw_meta.get("age"),
            "sex": raw_meta.get("sex"),
            "site": raw_meta.get("site"),
        },
    }
    diagnosis["feature_hints"] = build_feature_hints(
        diagnosis["prediction"],
        gradcam_summary,
        attention_summary,
        diagnosis["metadata_input"],
    )
    diagnosis["slm_payload"] = build_slm_payload(
        diagnosis,
        gradcam_summary=gradcam_summary,
        attention_summary=attention_summary,
        metadata=diagnosis["metadata_input"],
    )

    with open(str(img_out_dir / ARTIFACT_FILENAMES["diagnosis"]), "w") as f:
        json.dump(diagnosis, f, indent=2)
    artifact_paths["diagnosis"] = str(img_out_dir / ARTIFACT_FILENAMES["diagnosis"])

    # Console summary
    tag = "** MALIGNANT **" if is_mal else "benign"
    print(f"\n  {stem}: {CLASS_FULL[pred_cls]} ({pred_cls}) "
          f"[{pred_conf:.1%}] {tag}")
    print(f"    -> {img_out_dir}")

    return {
        "diagnosis": diagnosis,
        "artifact_paths": artifact_paths,
        "output_dir": str(img_out_dir),
    }


def _parse_age(age):
    if age is None:
        return None
    if isinstance(age, str) and age.strip().upper() in {"", "NA", "UNKNOWN"}:
        return None
    try:
        return float(age)
    except (TypeError, ValueError):
        return None


def run_inference_pipeline(image_path, age, sex, site, mode="full",
                           output_dir=None, checkpoint_path=None):
    """Importable backend entrypoint for one live LesionIQ analysis.

    Returns a dict with diagnosis.json-compatible data, absolute artifact paths,
    and the output directory so the API can expose the bundle statically.
    """
    mode = normalize_mode(mode)
    if output_dir is None:
        output_dir = REPO_ROOT / "output" / "inference"

    age_value = _parse_age(age)
    sex_value = None if sex is None else str(sex).strip().lower()
    if sex_value in {"", "na", "unknown"}:
        sex_value = None
    site_value = None if site is None else str(site).strip().lower()
    if site_value in {"", "na", "unknown"}:
        site_value = None

    (model, scales, temperature, mel_threshold, per_class_temperatures,
     train_prior, target_prior, prior_adapt_mode) = _load_runtime(mode, checkpoint_path)
    meta_tensor = encode_metadata(age_value, sex_value, site_value) \
        if mode == "full" else None
    raw_meta = {
        "age": int(age_value) if age_value is not None else None,
        "sex": sex_value,
        "site": site_value,
    }

    return diagnose_image(
        model, image_path, meta_tensor, scales, temperature,
        output_dir, raw_meta, mel_threshold, per_class_temperatures,
        train_prior, target_prior, prior_adapt_mode)


# ===================================================================
#  CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LesionIQ — Inference + Explainability Pipeline")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--dir",   type=str, help="Path to directory of images")

    parser.add_argument("--mode", type=str, default="full",
                        choices=["effnet_only", "swin_only",
                                 "image_only", "full"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "output" / "inference"))

    # Metadata — pass "NA" or omit for unknown/population defaults
    parser.add_argument("--age",  type=str, default=None,
                        help="Patient age (number or NA)")
    parser.add_argument("--sex",  type=str, default=None,
                        choices=["male", "female", "unknown", "NA"])
    parser.add_argument("--site", type=str, default=None,
                        help="Anatomical site (e.g. head/neck) or NA")

    parser.add_argument("--no-scales",      action="store_true",
                        help="Disable DiffEvo threshold scaling")
    parser.add_argument("--no-temperature", action="store_true",
                        help="Disable temperature scaling")
    parser.add_argument("--no-mel-safety",  action="store_true",
                        help="Disable MEL recall safety threshold")
    parser.add_argument("--adapt-prior",    default="none",
                        choices=["none", "sld", "oracle"],
                        help="Apply post-hoc prior-shift correction using a "
                             "precomputed target prior. Requires "
                             "effective_train_prior.npy and target_prior_<mode>.npy "
                             "in backend/checkpoints/. Default 'none' keeps "
                             "the existing behavior.")
    parser.add_argument("--interactive",    action="store_true",
                        help="Force interactive metadata prompts")

    args = parser.parse_args()

    # ── Load model ──
    model = build_model(args.mode, args.checkpoint)

    # ── Load DiffEvo scales ──
    scales = None
    if not args.no_scales:
        scales_path = CKPT_DIR / "optimal_scales.npy"
        if scales_path.exists():
            scales = np.load(str(scales_path))
            print("[OK] DiffEvo threshold scales loaded")

    # ── Load temperature (global scalar fallback) ──
    temperature = 1.0
    if not args.no_temperature:
        temp_path = CKPT_DIR / "optimal_temperature.npy"
        if temp_path.exists():
            temperature = float(np.load(str(temp_path)))
            print(f"[OK] Global temperature scaling: T={temperature:.4f}")

    # ── Load per-class temperatures (preferred; overrides global T) ──
    per_class_temperatures = None
    if not args.no_temperature:
        pc_path = CKPT_DIR / "per_class_temperatures.npy"
        if pc_path.exists():
            per_class_temperatures = np.load(str(pc_path)).astype(np.float32)
            print(f"[OK] Per-class temperature scaling loaded "
                  f"(mean T={per_class_temperatures.mean():.4f})")

    # ── Load MEL safety threshold ──
    mel_threshold = None
    if not args.no_mel_safety:
        mel_path = CKPT_DIR / "mel_safety_threshold.npy"
        if mel_path.exists():
            mel_threshold = float(np.load(str(mel_path)))
            print(f"[OK] MEL safety threshold: {mel_threshold:.3f}")

    # ── Load prior-shift adaptation assets (opt-in) ──
    train_prior  = None
    target_prior = None
    prior_adapt_mode = args.adapt_prior
    if prior_adapt_mode in ("sld", "oracle"):
        tp_path     = CKPT_DIR / "effective_train_prior.npy"
        target_path = CKPT_DIR / f"target_prior_{prior_adapt_mode}.npy"
        if tp_path.exists() and target_path.exists():
            train_prior  = np.load(str(tp_path)).astype(np.float32)
            target_prior = np.load(str(target_path)).astype(np.float32)
            print(f"[OK] Prior-shift adaptation active "
                  f"(mode={prior_adapt_mode}, target mean={target_prior.mean():.3f})")
        else:
            missing = [str(p) for p in (tp_path, target_path) if not p.exists()]
            print(f"[WARN] --adapt-prior {prior_adapt_mode} but missing files: "
                  f"{missing}. Falling back to no prior adjustment.")
            prior_adapt_mode = "none"

    # ── Metadata ──
    # Determine if we need interactive prompts:
    #   - Single image mode + no metadata CLI args + not piped stdin
    has_cli_meta = any([args.age, args.sex, args.site])
    want_interactive = (args.interactive
                        or (args.image and not has_cli_meta
                            and sys.stdin.isatty()))

    if args.mode == "full":
        if want_interactive and not args.dir:
            age, sex, site = _prompt_metadata_interactive()
        else:
            # Parse CLI args — treat "NA" as None
            age  = None
            if args.age and args.age.upper() != "NA":
                try:    age = float(args.age)
                except: age = None
            sex  = args.sex if args.sex and args.sex.upper() != "NA" else None
            site = args.site if args.site and args.site.upper() != "NA" else None

        meta_tensor = encode_metadata(age, sex, site)
        raw_meta = {
            "age":  int(age) if age is not None else None,
            "sex":  sex,
            "site": site,
        }

        parts = []
        if age is not None: parts.append(f"age={int(age)}")
        if sex:             parts.append(f"sex={sex}")
        if site:            parts.append(f"site={site}")
        print(f"[OK] Metadata: {', '.join(parts) if parts else 'NA (population defaults)'}")
    else:
        meta_tensor = None
        raw_meta = {"age": None, "sex": None, "site": None}

    # ── Collect images ──
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted([
            str(f) for f in Path(args.dir).iterdir()
            if f.suffix.lower() in extensions
        ])
        print(f"[OK] Found {len(image_paths)} images in {args.dir}")

    # ── Run pipeline ──
    print(f"\n{'='*55}")
    print(f"  LesionIQ Diagnostic Pipeline")
    calib_label = (f"per-class T (mean={per_class_temperatures.mean():.3f})"
                   if per_class_temperatures is not None
                   else f"global T={temperature:.4f}")
    print(f"  Mode: {args.mode} | Calibration: {calib_label} | "
          f"Scales: {'on' if scales is not None else 'off'}")
    print(f"{'='*55}")

    all_results = []
    for img_path in image_paths:
        try:
            result = diagnose_image(
                model, img_path, meta_tensor, scales, temperature,
                args.output_dir, raw_meta, mel_threshold, per_class_temperatures,
                train_prior, target_prior, prior_adapt_mode)
            all_results.append(result["diagnosis"])
        except Exception as e:
            print(f"  [ERROR] {Path(img_path).name}: {e}")
            import traceback; traceback.print_exc()

    # ── Batch summary ──
    if len(all_results) > 1:
        print(f"\n{'='*55}")
        print(f"  BATCH SUMMARY ({len(all_results)} images)")
        print(f"{'='*55}")
        from collections import Counter
        counts = Counter(r["prediction"]["class"] for r in all_results)
        for cls in CLASS_NAMES:
            if counts[cls] > 0:
                mal = " [!]" if cls in MALIGNANT else ""
                print(f"    {cls:>5s}: {counts[cls]:3d}{mal}")
        mal_count = sum(1 for r in all_results
                        if r["prediction"]["is_malignant"])
        print(f"\n  Malignant: {mal_count}/{len(all_results)}")

    print(f"\n  Output: {args.output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
