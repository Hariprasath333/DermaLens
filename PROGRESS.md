# LesionIQ — Project Progress & Context Handoff

> Last updated: 2026-06-09. This document is a cold-start summary of
> everything done across the LesionIQ project so work can resume after a
> context reset. Two parallel tracks: the **hackathon app** (frozen,
> deployed) and the **research pipeline** (active).

---

## 0. TL;DR

- **Hackathon track** (`backend/`, `frontend/`): a working dermoscopy
  classifier + FastAPI + React app, deployed via ngrok → Vercel. **Frozen**
  — do not modify. Recent work added a calibration suite (per-class
  temperature, Dirichlet, SLD prior adaptation) and honest test-set eval.
- **Research track** (`research/`): a semester-long study of
  metadata-fusion mechanisms under domain shift. A self-contained,
  portable training pipeline lives in `research/pipeline/` (8 stages, 12
  model variants, packaged as a single zip + bootstrap script).
- **Git identities used** (this repo has had several committers):
  `Ranjith <ranjith070327@gmail.com>` and
  `Astroshreyas1 <astroshreyas495@gmail.com>`. GitHub remote:
  `https://github.com/Astroshreyas1/LesionIQ.git`.

---

## 1. Hackathon track (FROZEN — do not touch)

### What it is
8-class ISIC 2019 dermoscopy classifier: dual backbone
(EfficientNet-B4 + SwinV2-Base) + 13-d metadata MLP, late-concat fusion.
FastAPI bridge (`backend/api.py`) → React frontend (`frontend/`). SLM
(Gemma 3 4B via Ollama, local or remote) writes the clinical narrative.

### Deployment
- Frontend on **Vercel**; backend on a local GPU exposed via a tunnel.
- Switched from Cloudflare quick-tunnels (URL changes each restart) to an
  **ngrok static domain**: `overload-opacity-connector.ngrok-free.dev`.
- `frontend/vercel.json` rewrites `/api/*`, `/artifacts/*`, `/health` to
  that ngrok URL. Frontend uses **relative `/api` paths** (no build-time
  URL baked in).
- Start backend: `set PYTHONPATH=C:\LesionIQ` then
  `python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000`.
- Start tunnel: `ngrok http --url=overload-opacity-connector.ngrok-free.dev 8000`
  (**port 8000**, not 80).
- ngrok free tier shows a browser interstitial for image GETs; the
  backend SLM call uses a short connect-timeout so it fails fast.

### Performance work done (calibration suite, no retraining)
All in `backend/`, calibrated on the ISIC 2019 split:
- `backend/data/build_layer0_csvs.py` rebuilds `layer0_{train,val,test}.csv`
  from raw ISIC GroundTruth+Metadata. **Stratified 80/20**, `random_state=42`.
- Per-class temperature scaling (`post_training.py`,
  `PerClassTemperatureScaler`) → `per_class_temperatures.npy`. Val NLL
  0.82→0.70, val macro-F1 0.61→0.63.
- Dirichlet calibration (`experimental/calibrate_dirichlet.py`,
  gated by `LESIONIQ_USE_DIRICHLET=1`) → val NLL 0.81→0.60, ECE
  0.131→0.019, acc +2.8%.
- Prior-shift adaptation (SLD + oracle) via `--adapt-prior {none,sld,oracle}`,
  off by default. Files: `effective_train_prior.npy`,
  `target_prior_{sld,oracle}.npy`.
- ECE + reliability diagrams added to `backend/classifier/evaluate.py`.
- Logit-space ensemble averaging in `boost_f1*.py` + `post_training.py`.

### Held-out TEST evaluation (ISIC 2019 public test, 6,191 imgs)
- Exact numbers are intentionally NOT recorded here or in any committed
  file — they stay private (chat only). Do not write test-set metrics
  into the repo.
- Qualitative finding: there is a notable **val → test macro-F1 drop**.
  Two causes: (1) **prior shift** (test has higher MEL prevalence than
  val) and (2) **lesion leakage** in the old val split (val was an 80/20
  split of the *training* collection, not a true holdout — many lesions
  appear in both). AUC stays high because ranking is prior-invariant.
- **This is why the research track exists** (lesion-aware splits +
  prior-shift adaptation).

### Known hackathon-model defects (documented, NOT fixed — frozen)
From an architectural audit; left as-is in `backend/` on purpose:
1. Aux metadata head half-wired (`META_AUX_WEIGHT` does nothing;
   `meta_aux_head` undefined → would `AttributeError`).
2. Age norm train/serve skew (train `/90`, inference `/100`, NA=0.5).
3. Fusion magnitude swamp (32-d meta vs 2816-d image).
4. `WeightedRandomSampler` pure `1/count` (over-aggressive).
5. Tuned-on-val, never reported clean test.
6. Dual backbone marginal gain over single.
7. CutMix label noise on centered lesions.
8. Four imbalance mechanisms stacked.
These are **fixed or addressed by-design in the research pipeline** (see §3).

---

## 2. Research track overview (`research/`)

- `research/PLAN.md` — the semester thesis plan. Thesis: *metadata-fusion
  mechanism × injection-depth under domain shift*. 12 variants V0–V11
  cherry-picked from an 8-mechanism × 4-depth grid. Heterogeneous-metadata
  SchemaAligner to train across datasets with different schemas.
- `research/notes/DECISIONS.md` — running decision log. Most recent entry
  (2026-06-09) records the pipeline correctness audit (§4 below).
- `research/README.md` — folder map + isolation guarantee (research never
  imports backwards into the hackathon; hackathon stays frozen).
- `research/pipeline/` — the portable training pipeline (§3).

---

## 3. The research pipeline (`research/pipeline/`)

Self-contained. Packaged via `python package.py` →
`lesioniq-pipeline-vX.Y.Z.zip` (currently **v1.2.0**; zip is gitignored,
regenerate as needed). Bootstrap: `lesioniq.bat` (Windows) /
`lesioniq.sh` (Linux) create a `.venv`, install `requirements.txt`, then
forward to `run.py`.

### Stages (`stages/`)
1. `stage1_datasets.py` — `DATASET_REGISTRY` + `verify_datasets`. Lists 7
   curated datasets, prints download instructions, **downloads nothing**.
   Tier-1: ISIC2019, HAM10000, ISIC2020, PAD-UFES-20; +Fitzpatrick17k
   (fairness), Derm7pt (multi-task), PH2 (external test).
2. `stage2_preprocess.py` — 4-step (DullRazor → Shades-of-Gray → CLAHE →
   vignette crop → resize+pad), multiprocess, quarantines corrupt files,
   writes manifest. Worker never raises.
3. `stage3_split.py` — **lesion-aware** GroupShuffleSplit /
   StratifiedGroupKFold. `val_select ⊥ val_calibrate ⊥ test`. Cross-dataset
   SHA1 dedup. **Aborts if any lesion leaks across splits.** Per-dataset
   adapters in `DATASET_LOADERS`. Canonical 8 classes + 9 sites.
4. `stage4_dataloader.py` — single `encode_row_metadata` (19-d canonical:
   age + sex×3 + site×9 + fitz×6) with a **presence mask** (missing →
   mask 0 → learned absent embedding). `1/sqrt(count)` balanced sampler.
   Safe-collate drops corrupt rows. Zero-deprecation augs.
5. `stage5_ensemble.py` — batched multi-model forward, **logit-space**
   learned weights (LBFGS simplex), calibration stack in correct order,
   `evaluate_ensemble()` orchestrator entry.
6. `stage6_train.py` — 4 losses (focal/cb_focal/soft_f1/ldam),
   MixUp+CutMix (**image-only**, metadata NOT blended), EMA with
   store/restore, AMP, cosine+warmup, grad accum, NaN/OOM guards.
7. `stage7_evaluate.py` — global T + per-class T + Dirichlet + SLD,
   ECE/per-class ECE, reliability diagrams, confusion matrix.
8. `stage8_audit.py` — fairness (skin-tone/sex/age), per-lesion
   aggregation, selective accuracy/risk-coverage, missing-metadata
   robustness, integrated-gradient per-feature attribution.

### Models (`models/`)
- `schema_aligner.py` — `SchemaAligner` maps 19-d meta+mask → 64-d patient
  embedding; learned "absent" embedding per feature; optional per-feature
  dropout (V11). Ends in LayerNorm.
- `injectors.py` — M0 `LateConcat`, M1 `FiLM`, M2 `CrossAttention`,
  M3 `TokenFusion` (post-encoder), M4 `Hypernetwork`, M5 `ConditionalBN`,
  M6 `GatedFusion`.
- `backbones.py` — `DualBackbone` (timm EfficientNet-B4 + SwinV2-Base,
  with tiny CPU fallbacks for `use_timm=False` smoke tests). Robust
  `swin_feature_tokens` returns `(B,L,C)` post-encoder tokens.
- `variants.py` — V0–V11 registry (`build_variant`, `ALL_VARIANT_IDS`).
  V4/V8/V11 use `TokenFusion`. V9 is the deliberate useless stem-concat
  control. V11 = V8 + per-feature dropout.

### Run it
```
python run.py selftest          # synthetic E2E + 2 regression guards (~30s)
python run.py verify   --data-root <dir>
python run.py preprocess --data-root <raw> --out-root <pre> --datasets isic2019 ...
python run.py split    --pre-root <pre> --raw-root <raw> --out <splits> --datasets ...
python run.py train    --variant V1 --split-dir <splits/run> --out-dir <runs/V1>
python run.py evaluate --variant V1 --checkpoint <runs/V1/best.pt> --split-dir <..> --out-dir <..>
python run.py audit    --variant V1 --checkpoint <..> --split-dir <..> --out-dir <..>
python run.py ensemble --variants V1 V4 --checkpoints a.pt b.pt --split-dir <..> --out-dir <..>
python run.py full     --config pipeline.yaml
```

---

## 4. Pipeline bugs found & FIXED (audit on 2026-06-09)

Two were the same "reads as active but isn't" class as the old model's
aux-head bug:

- **Bug A — EMA never restored** (`stage6_train.py`). EMA eval overwrote
  live weights and never restored them → training continued from lagged
  EMA + best.pt save/select mismatch. **Fixed**: `EMA.store()/restore()`;
  loop now eval-live → store → copy-EMA → eval-EMA → restore (finally),
  selects max(live,ema), saves the scored weights + `weights_source` tag.
- **Bug B — token injection silently dead on real timm** (`variants.py`,
  `backbones.py`). V4 (and token-half of V8/V11) appended tokens
  *pre-encoder* and relied on an `.encode()` that only the tiny fallback
  has → under timm the tokens were discarded, V4 trained image-only with
  **zero gradient to metadata**. **Fixed**: new `TokenFusion` operates on
  the post-encoder token sequence (self-attention over
  `[patches; meta_tokens]`, pool patches). Gradient-carrying on timm.
- **Latent reshape bug** (found while fixing B): old `swin_features`
  mis-reshaped timm SwinV2's channels-last `(B,H,W,C)` → would corrupt
  every variant under real timm. **Fixed** by robust `swin_feature_tokens`.
- **Bug C — CutMix blended metadata across patients**. **Fixed**:
  `mix_batch` mixes image only; primary sample's metadata passes through.

### Regression guards added (in `run.py selftest`)
- **Metadata gradient-flow guard** — every variant must route nonzero
  grad to `meta` (with a few bootstrap steps to clear FiLM's identity
  init). Proven to catch a meta-ignoring model. *This is the test that
  would have caught Bug B.*
- **EMA store/restore invariant** — proves copy_to swaps shadow in and
  restore returns exact live weights.

### Cross-check: 8 audit points vs the new pipeline
| # | Status in `research/pipeline/` |
|---|---|
| 1 (aux head) | No aux head; same pathology recurred as Bug B → fixed |
| 2 (norm skew) | Eliminated — single encode fn, learned absent embedding |
| 3 (magnitude swamp) | By design — V1–V11 are the fixes; V0 swamped baseline |
| 4 (sampler) | Already `1/sqrt(count)` |
| 5 (test contamination) | Fixed — val_select ⊥ val_calibrate ⊥ test, lesion-grouped |
| 6 (dual backbone) | Deferred by design (held constant for fair comparison) |
| 7 (CutMix noise) | Meta-blend facet fixed (Bug C); bbox risk noted |
| 8 (stacked regularizers) | Loss is an ablation axis (L0–L3); default stacks, measurable |

---

## 5. Key file locations

```
backend/                                  hackathon (frozen)
  classifier/{inference,post_training,evaluate,explainability,boost_f1*}.py
  data/build_layer0_csvs.py
  checkpoints/*.npy *.npz                  calibration artifacts
experimental/calibrate_dirichlet.py        Dirichlet (gated)
frontend/vercel.json                       ngrok rewrites
research/PLAN.md                            thesis plan
research/notes/DECISIONS.md                 decision log (audit entry at top)
research/pipeline/                          portable training pipeline
  run.py                                    CLI entry (+ selftest guards)
  package.py                                builds the zip
  pipeline.yaml                             full-run config
  stages/ models/ utils/ docs/DATASETS.md
PROGRESS.md                                 this file
```

---

## 6. Datasets (none committed; user downloads separately)

- Recommended training mix: **ISIC2019 + HAM10000 + ISIC2020 +
  PAD-UFES-20** (PAD has 22 metadata fields incl. Fitzpatrick skin tone).
- Fairness external: **Fitzpatrick17k** (skin tone), never trained on.
- Cross-dataset **dedup is mandatory** before splitting (HAM10000 ⊂
  ISIC2018 ⊂ ISIC2019; BCN20000 ⊂ ISIC2019).
- User's local raw ISIC 2019: `C:\LesionIQ\dataset\ISIC_2019_*_Input\...`
  (nested one level). Validate images for PPT:
  `C:\Users\Shreyas\Desktop\validate_images_zip\validate_images\`.

---

## 7. What's next (see `research/pipeline/README.md` → "Future Testing")

1. Download Tier-1 datasets, run `verify` → `preprocess` → `split`.
2. Train V0 baseline + V1–V11 on the lesion-aware split; report **test**
   macro-F1 + ECE per variant (the core ablation).
3. Loss-function ablation (L0–L3) on top-3 variants.
4. Fairness audit (needs PAD-UFES-20 + Fitzpatrick17k).
5. (Hackathon, optional) the single biggest lever for the deployed model
   would be a lesion-aware re-split + retrain — but that's a research
   activity, and the hackathon stays frozen.

---

## 8. Gotchas / conventions

- **Never modify `backend/` or `frontend/`** for research work — frozen.
- Research code may import *from* `backend/` but never vice-versa.
- Windows console is cp1252 → avoid Unicode (→, box chars, ▓) in `print`;
  use ascii or set `PYTHONIOENCODING=utf-8`. Several past crashes were
  exactly this.
- `*.zip`, `__pycache__/`, `.venv/`, and pipeline `runs/ splits/
  preprocessed/ datasets/` are gitignored.
- Commits in this session were requested under
  **Astroshreyas1 <astroshreyas495@gmail.com>** (the repo owner).
