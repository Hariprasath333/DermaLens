# LesionIQ Research Pipeline

A self-contained, portable training pipeline for the LesionIQ
metadata-fusion thesis. Drop the zip on any machine with Python 3.10+
and an NVIDIA GPU, run the bootstrap script, and follow the prompts.

## Quick start

### Windows
```cmd
lesioniq.bat verify   --data-root C:\path\to\datasets
lesioniq.bat full     --config pipeline.yaml
```

### Linux / macOS
```bash
chmod +x lesioniq.sh
./lesioniq.sh verify   --data-root /path/to/datasets
./lesioniq.sh full     --config pipeline.yaml
```

The first invocation creates a local `.venv/` and installs requirements.
Subsequent runs reuse it.

---

## Pipeline stages

1. **Verify datasets** — checks which datasets you've downloaded; prints
   instructions for missing ones. **Does not download anything.**
2. **Preprocess** — applies the 4-step pipeline (DullRazor →
   Shades-of-Gray → CLAHE → vignette removal → resize+pad) with
   multi-process workers, quarantines corrupt files, and writes a
   manifest.
3. **Split** — lesion-aware GroupShuffleSplit (or StratifiedGroupKFold
   for the final ablation). `val_select` and `val_calibrate` are
   strictly disjoint. Cross-dataset dedup via SHA1. Aborts on any
   lesion leak between splits.
4. **Dataloader** — canonical 19-d metadata + presence mask, safe-collate
   that drops corrupt rows, zero-deprecation augmentation pipelines.
5. **Ensemble** — single batched forward across N submodels, learned
   logit-space weights (LBFGS, simplex via softmax param), fp16 backbone
   + fp32 reduction. Applies the calibration stack (T → Dirichlet →
   prior shift → scales) in mathematically correct order.
6. **Train** — pick one of the 12 variants V0..V11, one of the 4 losses
   (focal, cb_focal, soft_f1, ldam), train with MixUp+CutMix, EMA, AMP,
   cosine schedule + warmup.
7. **Evaluate** — collects test logits, fits global T + per-class T +
   Dirichlet on `val_calibrate`, estimates target prior via SLD,
   computes ECE / per-class ECE, saves reliability diagrams + confusion
   matrix.
8. **Audit** — fairness (per-skin-tone, per-sex, per-age band),
   per-lesion vs per-image aggregation, selective accuracy at coverage,
   missing-metadata robustness, per-feature integrated-gradient
   attribution. Required deliverables for the thesis.

Run `./lesioniq.sh selftest` to verify all 8 stages on tiny synthetic
data in under 30 seconds — no real datasets needed.

---

## Datasets

See `docs/DATASETS.md` for the full curated catalogue. Tier-1
essentials:

| Key | Download |
|---|---|
| `isic2019` | https://challenge.isic-archive.com/data/#2019 |
| `ham10000` | https://doi.org/10.7910/DVN/DBW86T |
| `isic2020` | https://challenge.isic-archive.com/data/#2020 |
| `pad_ufes_20` | https://data.mendeley.com/datasets/zr7vgbcyr2/1 |
| `fitzpatrick17k` (fairness audit only) | https://github.com/mattgroh/fitzpatrick17k |

Run `./lesioniq.sh verify --data-root <dir>` and it prints the exact
file/folder layout each dataset expects, with disk paths.

---

## Variants

| ID  | Mechanism      | Depth        | Hypothesis |
|-----|----------------|--------------|------------|
| V0  | late concat    | classifier   | baseline (hackathon) |
| V1  | FiLM           | feature      | cheapest meaningful gain |
| V2  | FiLM           | CNN block    | deep CNN modulation |
| V3  | cross-attn     | feature      | spatial attn via meta |
| V4  | token inject   | Swin pre-enc | transformer-native |
| V5  | hypernetwork   | classifier   | interpretable per-patient |
| V6  | conditional BN | CNN block    | per-block BN modulation |
| V7  | gated fusion   | feature      | soft mask over channels |
| V8  | hybrid M1+M3   | block        | all branches see meta |
| V9  | concat         | stem         | useless control |
| V10 | FiLM           | stem         | stem-level FiLM |
| V11 | V8 + dropout   | block        | missing-meta robustness |

---

## Commands

```bash
# Verify what's downloaded
lesioniq verify --data-root <dir> [--only-tier 1]

# Preprocess (multi-process, skip-if-exists)
lesioniq preprocess --data-root <raw> --out-root <pre> \
                     --datasets isic2019 ham10000 \
                     --workers 4 --resize 384

# Split (lesion-aware, val_select ⊥ val_calibrate)
lesioniq split --pre-root <pre> --raw-root <raw> --out <splits> \
                --datasets isic2019 ham10000 --mode single --seed 42

# Train one variant
lesioniq train --variant V1 --split-dir <splits/run_id> \
                --out-dir <runs/V1> --epochs 30 --batch-size 32 \
                --loss focal

# Evaluate one variant
lesioniq evaluate --variant V1 --checkpoint <runs/V1/best.pt> \
                   --split-dir <splits/run_id> --out-dir <runs/V1>

# End-to-end from YAML
lesioniq full --config pipeline.yaml
```

---

## Configuration

`pipeline.yaml` is the single config file. Defaults reproduce the
hackathon ablation. Edit:

- `datasets`: which Tier-1 datasets to include
- `variants`: which variant IDs to sweep
- `epochs`, `batch_size`, `lr`, `loss`: standard training knobs
- `pretrained`: whether to load ImageNet weights (large download)
- `use_timm`: set false for tiny-backbone smoke tests on CPU only

---

## Folder layout after a full run

```
<runs_root>/
├── V0/
│   ├── config.json
│   ├── train_log.jsonl
│   ├── best.pt
│   └── eval/
│       ├── metrics.json
│       ├── calibration.json
│       ├── reliability_{raw,global_T,per_class_T,dirichlet}.png
│       └── confusion_matrix.png
├── V1/
│   └── ...
└── SUMMARY.json
```

---

## Smoke testing without a real dataset

For CI / development, every module supports a `use_timm=False` fallback
that swaps the real backbones for tiny CPU-friendly stand-ins. Variants
still build, forward-pass, train, and evaluate end-to-end. Output
quality is meaningless (the fallback backbone is too small to learn
anything) — this is purely a code-path test.

Run `python run.py selftest` to exercise the whole stack on synthetic
data in ~30 s, including two regression guards:

- **Metadata gradient-flow guard** — asserts every variant V0–V11 routes
  nonzero gradient to its metadata input (catches any silently-dead
  fusion mechanism).
- **EMA store/restore invariant** — asserts EMA evaluation never corrupts
  the live training weights.

---

## Future Testing

The items below are empirical experiments to run once the real datasets
are in place. They come from an architectural audit of the original
hackathon model; each entry notes whether the new pipeline already
addresses it structurally or whether it is an open experiment to run.

### Already guarded in code (verify on first real run)
- **Metadata is actually used.** The gradient-flow guard proves every
  variant *can* learn from metadata. On the first real training run,
  confirm `audit` → `missing_meta` shows a non-trivial macro-F1 drop when
  metadata groups are ablated (if dropping age/sex/site/skin-tone does
  nothing, the fusion is not contributing and the variant needs review).
- **Honest validation protocol.** `val_select` (model selection),
  `val_calibrate` (temperature/threshold fitting), and `test` (report
  once) are strictly disjoint and lesion-grouped. Confirm the headline
  number is read **only** from `test`, never from any val split.
- **Train/serve normalization parity.** A single `encode_row_metadata`
  encodes metadata identically at train and inference; missing fields
  use a learned "absent" embedding (not an arbitrary constant). No skew
  to test for — but assert it stays single-source if a separate
  inference path is ever added.

### Open experiments (the core ablation grid)
- **Fusion mechanism × injection depth.** Train V0 (late-concat baseline)
  against V1–V11 on the lesion-aware split and report the per-variant
  test macro-F1 + ECE. The hypothesis is that FiLM / token-fusion /
  gating beat the magnitude-swamped concat baseline. This is the thesis.
- **Loss-function axis (L0–L3).** On the top-3 fusion variants, ablate
  Focal vs Class-Balanced Focal vs soft-macro-F1 vs LDAM. Report a 3×4
  mean-F1 matrix with bootstrap CIs.
- **Sampler strength.** The dataloader uses `1/sqrt(count)` sampling.
  Test it against plain `1/count`, "effective number" reweighting
  (Cui et al. 2019), and uniform — macro-F1 on the held-out test set.
- **Regularizer stacking.** Ablate the four imbalance mechanisms
  (focal + label smoothing + balanced sampler + heavy aug) one at a
  time; a square-root-balanced sampler with plain cross-entropy is a
  strong baseline worth measuring against the full stack.
- **MixUp/CutMix on centered lesions.** Metadata is no longer blended,
  but the spatial bbox can still grab peripheral skin. Test
  center-biased bbox and per-class disabling for DF/VASC.
- **Single strong backbone vs dual.** The dual EfficientNet-B4 + SwinV2
  trunk is held constant for a fair fusion comparison, but a follow-up
  should ablate a single modern backbone (ConvNeXt-V2-L / EVA-02 /
  EfficientNetV2-L at 384) against the dual setup on the test set.

### Fairness & deployment (run before any external claim)
- **Per-skin-tone audit.** `audit` → `fairness` reports macro-F1 and MEL
  recall stratified by Fitzpatrick I–II / III–IV / V–VI (needs
  PAD-UFES-20 + Fitzpatrick17k). Any stratum > 0.10 macro-F1 below the
  others is a blocker, regardless of headline F1.
- **Per-lesion aggregation.** Report per-lesion (majority vote and
  mean-prob) alongside per-image F1 for multi-shot lesions.
- **Selective accuracy / risk-coverage.** Report accuracy at 50/70/90%
  coverage and the MEL miss-rate at confidence thresholds — the real
  human-in-the-loop deployment metric.
- **Prior-shift robustness.** Re-fit SLD / oracle target priors on each
  external test set and confirm the calibration stack recovers
  performance under the val→test prior shift.
