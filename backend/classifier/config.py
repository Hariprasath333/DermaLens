"""
LesionIQ Hybrid Classifier — Shared Configuration
===================================================
Paths are loaded from backend/config.yaml.  Environment variables
(LESIONIQ_TRAIN_CSV, LESIONIQ_VAL_CSV, etc.) override the YAML values.
Hyperparameters below are edit-in-place.
"""

import os
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]

# ── Load config.yaml ─────────────────────────────────────────
_CONFIG_PATH = BACKEND_ROOT / "config.yaml"
_yaml_cfg: dict = {}
try:
    import yaml
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as _f:
            _yaml_cfg = yaml.safe_load(_f) or {}
except ImportError:
    pass


def _cfg(key: str, env_key: str, default: str) -> str:
    """Resolve a config value: env var > config.yaml > default."""
    return os.environ.get(env_key, _yaml_cfg.get(key, default))


# ── Data paths (config.yaml → env override) ──────────────────
TRAIN_CSV     = _cfg("train_csv",     "LESIONIQ_TRAIN_CSV",     "path/to/LesionIQ/layer0_train.csv")
VAL_CSV       = _cfg("val_csv",       "LESIONIQ_VAL_CSV",       "path/to/LesionIQ/layer0_val.csv")
TEST_CSV      = _cfg("test_csv",      "LESIONIQ_TEST_CSV",      "path/to/LesionIQ/test set/final/layer0_test.csv")
TRAIN_IMG_DIR = _cfg("train_img_dir", "LESIONIQ_TRAIN_IMG_DIR", "path/to/LesionIQ/Segregated")
TEST_IMG_DIR  = _cfg("test_img_dir",  "LESIONIQ_TEST_IMG_DIR",  "path/to/LesionIQ/Test")
METADATA_CSV  = _cfg("metadata_csv",  "LESIONIQ_METADATA_CSV",  "path/to/LesionIQ/metadata.csv")
OUTPUT_DIR    = _cfg("output_dir",    "LESIONIQ_OUTPUT_DIR",    "") or str(BACKEND_ROOT / "output")

# ── Metadata column names ────────────────────────────────────
META_ID_COL     = "isic_id"
META_AGE_COL    = "age_approx"
META_SEX_COL    = "sex"
META_REGION_COL = "region"
META_LABEL_COL  = "disease-class"

# ── Class configuration ──────────────────────────────────────
NUM_CLASSES = 8
CLASS_NAMES = None  # auto-detected from subfolder names when None

# ── Training hyper-parameters ────────────────────────────────
EPOCHS       = 50
BATCH_SIZE   = 16
GRAD_ACCUM_STEPS = 3
LR           = 1e-4
WEIGHT_DECAY = 0.01
IMG_SIZE     = 384
VAL_SPLIT    = 0.2
SEED         = 42
NUM_WORKERS  = 4

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = [1.0] * 8  # Uniform alpha — class balance is handled by WeightedRandomSampler
                         # Using both sampler + alpha = double compensation that hurts common classes

# ── Cost-sensitive clinical penalty ──────────────────────────
# Prioritize Melanoma (MEL) recall: false negatives carry severe clinical risk
MEL_COST_FACTOR = float(os.getenv("MEL_COST_FACTOR", "2.5"))
SCC_COST_FACTOR = float(os.getenv("SCC_COST_FACTOR", "1.8"))
BCC_COST_FACTOR = float(os.getenv("BCC_COST_FACTOR", "1.5"))

# ── Early stopping ───────────────────────────────────────────
PATIENCE = 10

# ── Cosine-annealing ─────────────────────────────────────────
# Using plain CosineAnnealingLR (no restarts) to avoid timing conflict:
# WarmRestarts T_0=10 + patience=10 causes early stopping right after a
# LR restart before the model has time to recover in the new cycle.
COSINE_T_MAX = 50  # Decay LR smoothly over all epochs

# ── Swin layer-wise LR decay ────────────────────────────────
SWIN_LR_DECAY = 0.65

# ── Label smoothing (reduces overconfidence on rare classes) ─
LABEL_SMOOTHING = 0.1

# ── Metadata branch ─────────────────────────────────────────
META_EMBED_DIM   = 64   # dimensionality kept high enough to stay relevant
META_AUX_WEIGHT  = 0.3  # auxiliary-loss weight so the metadata branch never goes dormant
META_LR_SCALE    = 3.0  # metadata branch LR = LR * META_LR_SCALE (trains from scratch)
AGE_NORM_FACTOR  = float(os.getenv("AGE_NORM_FACTOR", "90.0"))  # Consistent train/inference age scaling

# ── Explainability ───────────────────────────────────────────
CONFIDENCE_THRESHOLD        = 0.7
NUM_EXPLAINABILITY_SAMPLES  = 20
SHAP_BACKGROUND_SAMPLES     = 100

# ── Mixed precision ──────────────────────────────────────────
USE_AMP = True

# ── Device ───────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
