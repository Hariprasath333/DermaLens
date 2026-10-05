# LesionIQ Research — Semester Plan

> **Working title:** *Metadata-Aware Dermoscopy Classification under Domain Shift:
> A Systematic Study of Patient-Context Fusion Mechanisms.*

> **Scope discipline.** This is a semester-long, depth-first research project.
> It is **not** a continuation of the hackathon. The hackathon code remains
> untouched in `backend/`. All research lives in `research/` and must not
> introduce dependencies on hackathon-specific calibration files or routing.

---

## 1. Thesis statement (one sentence)

The dominant failure mode of current dermoscopy classifiers under deployment
is not feature extraction but **how patient metadata (age, sex, anatomical
site, skin tone) is fused with image features**, and a principled study
of fusion-mechanism *× injection-depth* against heterogeneous metadata
schemas across multiple dermoscopy datasets yields a calibrated,
domain-shift-robust pipeline that closes most of the val→test
generalization gap without resorting to architectural exotica.

---

## 2. Motivation: what the hackathon revealed

| Observation from the hackathon | Implication |
|---|---|
| Val F1 = 0.61, test F1 = 0.50 | The reported gap is mostly **patient leakage** (66% of val lesions seen in train), not architecture limitations. |
| AUC stays at 0.88 on test | The backbones rank lesions correctly; the threshold/calibration layer is brittle under prior shift. |
| Late-fused metadata (concat at classifier) | Image backbones (105M params) never see patient context during feature extraction. A 70yo head-and-neck lesion and a 20yo torso lesion get identical features. |
| Per-class temperature, Dirichlet, SLD all help calibration but not the underlying F1 ceiling | Post-hoc fixes cannot recover what the architecture did not learn. |
| Multi-dataset training is impractical with current code | Each dataset has a different metadata schema — no clean way to merge ISIC + HAM10000 + PAD-UFES-20 (skin tone) + Fitzpatrick17k. |

The architectural fix and the schema-heterogeneity fix are the same problem
in different disguises: **how is patient context injected into a vision
backbone, and how does that injection generalize when the metadata schema
itself shifts?** This is the thesis.

---

## 3. Research questions

**RQ1.** Among standard metadata-fusion mechanisms (late concat, FiLM,
cross-attention, token injection, hypernetworks, conditional norm),
which yields the best val-→test generalization on dermoscopy
classification, and does the answer depend on *where in the network*
the injection happens (input, stem, block, feature, classifier)?

**RQ2.** Does fusion-mechanism choice interact with **metadata schema
heterogeneity** across datasets? Specifically, when a model is trained
on a dataset with rich metadata (PAD-UFES-20, 20+ features including
skin tone and lesion history) and evaluated on a dataset with sparse
metadata (ISIC 2019, 3 features), which mechanism degrades gracefully?

**RQ3.** Is the prior-shift adaptation pipeline (SLD + Dirichlet) we
developed in the hackathon **complementary or substitutable** with
better metadata fusion? That is, does a well-fused model still benefit
from post-hoc adaptation, or does it already capture the deployment
distribution implicitly?

**RQ4.** Do per-feature contributions to per-class accuracy follow
**clinically expected priors** (e.g. site=head/neck should help AK/SCC;
fitzpatrick=I-II should help MEL detection)? A negative answer would
imply the model treats metadata as nuisance features.

**RQ5.** Can a single trained model handle **missing metadata at
deployment** (a clinic that records sex but not skin tone) without
catastrophic per-class degradation?

---

## 4. Proposed contributions

1. **Comparative study of 8 metadata-fusion variants** along two
   orthogonal axes: *mechanism* (concat, FiLM, cross-attention, token
   injection, hypernetwork, conditional norm, gated fusion, hybrid) ×
   *injection depth* (stem, block, feature, classifier).

2. **A heterogeneous-metadata training framework** that handles 5+
   dermoscopy datasets with different metadata schemas via a learnable
   schema-aligner. Each dataset contributes the features it has;
   missing features are imputed by a learnable per-feature default
   embedding.

3. **Per-feature, per-class causal attribution** of metadata
   contribution, distinguishing features the model uses correctly
   (clinically expected) from features it overfits to (spurious).

4. **A robust calibration pipeline** combining per-class temperature
   scaling, Dirichlet calibration, and SLD prior adaptation, evaluated
   under both architectural and dataset shifts. Builds on the hackathon
   pipeline but with a clean re-derivation under the new architecture.

5. **A reproducibility artifact:** all splits patient-aware, all
   experiments seeded, all configurations as YAML, results in
   structured JSON, plots auto-generated.

---

## 5. Datasets

All datasets must be downloaded separately by the user. Storage at
`research/datasets/<name>/` (not tracked in git).

| Dataset | Size | Lesion-level | Image-level metadata | Skin tone |
|---|---|---|---|---|
| ISIC 2019 (train + test) | ~33K | partial | age, sex, anatom_site_general | ✗ |
| ISIC 2018 task 3 (HAM10000) | 10K | yes | age, sex, localization, dx_type | ✗ |
| ISIC 2020 | 33K | yes | age, sex, anatom_site_general, diagnosis | ✗ |
| BCN20000 | 19K | yes | age_approx, sex, body location | ✗ |
| PAD-UFES-20 | 2K | yes | **20+ features:** age, sex, region, skin tone (Fitzpatrick), diameter_1, diameter_2, family cancer history, smoke, drink, pesticide exposure, itch, grew, hurt, bleed, elevation | **✓** |
| Fitzpatrick 17k | 17K | no | skin condition, **Fitzpatrick skin type I-VI** | **✓** |
| Derm7pt | 2K | no | 7-point checklist attributes | ✗ |
| PH² | 200 | yes | clinical diagnosis, dermoscopic structures | ✗ |

**Total unique images target:** ~100K after de-duplication. **Cross-dataset
patient overlap** must be checked and removed (especially HAM10000 ↔ ISIC 2019,
which share images).

### Schema-aligner

A learnable layer that maps the union of all metadata features (≈40
fields) into a canonical 64-d patient embedding. Per-feature missingness
is signalled by a learnable mask token concatenated to that feature's
input. Each dataset contributes only the features it has.

```
canonical_features = [
    age, sex_enc(3), site_enc(9), skin_tone(6),
    diameter_mm, family_cancer_history(bool), smoke(bool), drink(bool),
    itch(bool), grew(bool), hurt(bool), bleed(bool), changed(bool),
    elevation(bool), pesticide_exposure(bool),
    dataset_source(8)  # learned source embedding (debiased downstream)
]
patient_embedding = SchemaAligner(canonical_features, mask_token_per_feature)
```

`dataset_source` is supplied during training to let the model identify
collection-specific biases; at inference it is set to a learned
"deployment" token that averages the trained sources.

---

## 6. The metadata-fusion variant grid (the centerpiece)

Eight architectures, all sharing the same dual-backbone (EfficientNet-B4
+ SwinV2-Base, frozen until last stage). They differ only in **where**
and **how** the patient embedding meets the image features.

### Mechanism axis

| ID | Mechanism | Formal definition |
|---|---|---|
| M0 | Late concat (baseline) | `[image_feat ; meta_feat] → FC` |
| M1 | FiLM | `γ(meta) ⊙ image_feat + β(meta)` |
| M2 | Cross-attention | `Attn(Q=meta, K=image_patches, V=image_patches)` |
| M3 | Token injection | metadata becomes additional input tokens to SwinV2 self-attention at every layer |
| M4 | Hypernetwork classifier | classifier weights `W = HyperNet(meta)`; logits = `W · image_feat` |
| M5 | Conditional BatchNorm | BN affine params `γ_l, β_l` for layer l are generated from `meta` |
| M6 | Gated fusion | `g(meta) = σ(MLP(meta)); fused = g ⊙ image + (1-g) ⊙ meta_proj` |
| M7 | Hybrid (M1 + M3) | FiLM in CNN branch + token injection in transformer branch |

### Injection-depth axis

| ID | Depth | Where in the backbone |
|---|---|---|
| D0 | Classifier-only | after pooling, before final FC (current pipeline) |
| D1 | Feature-level | post-backbone, pre-pooling (1792×12×12 for EffNet, 1024×12×12 for Swin) |
| D2 | Block-level | injected at the start of every backbone block / Swin layer |
| D3 | Stem-level | injected after the very first conv/patch-embed |

Not every (mechanism × depth) combination is meaningful. The actual
study evaluates a **defensible subset of 12 combinations**, not all 32:

| Variant ID | Mechanism | Depth | Hypothesis |
|---|---|---|---|
| V0 | M0 concat | D0 | Baseline; current behaviour |
| V1 | M1 FiLM | D1 (feature) | Cheapest meaningful improvement |
| V2 | M1 FiLM | D2 (block) | Deep modulation in CNN branch |
| V3 | M2 cross-attn | D1 (feature) | Spatial attention via patient context |
| V4 | M3 token | D2 (block) | Native to transformer branch |
| V5 | M4 hypernet | D0 | Patient generates classifier (interpretable) |
| V6 | M5 CBN | D2 | Per-block BN modulation |
| V7 | M6 gated | D1 | Soft mask over image channels |
| V8 | M7 hybrid | D2 | Best-case scenario, all branches see metadata |
| V9 | M0 concat | D3 | Useless control: meta repeated at every stage as concat baseline |
| V10 | M1 FiLM | D3 (stem) | Test: does stem-level injection lose information by abstraction? |
| V11 | M7 hybrid | D2 + per-feature dropout | Robustness to missing metadata |

All variants trained for **same number of optimizer steps**, **same
augmentation suite**, **same loss** (see §6.5), **same lesion-aware
split**, **same seeds**. The only changing variable is the fusion
variant.

### 6.5 Secondary ablation axis — loss function

The fusion-variant grid above isolates *where and how metadata enters
the network*. A separate, narrower ablation isolates the loss function,
because the metric we optimize directly affects rare-class behaviour and
calibration:

| Loss ID | Loss | Rationale |
|---|---|---|
| L0 | Focal (γ=2.0, ε=0.1) | Current hackathon default; reference |
| L1 | Class-Balanced Focal (Cui et al., 2019) | Effective-sample reweighting; principled for long tail |
| L2 | Macro-F1 surrogate (soft-F1) | Directly optimizes the headline metric; reported to gain +0.1–0.15 macro-F1 on imbalanced medical tasks |
| L3 | LDAM (Label-Distribution-Aware Margin, Cao et al., 2019) | Margin-based, theoretically motivated for imbalance |

Loss-function ablation runs on the **top-3 fusion variants** chosen
from §6 (typically V1, V4, V8). Total = 3 fusions × 4 losses × 5
seeds = 60 runs. Reported as a 3 × 4 mean-F1 matrix with paired
bootstrap CIs.

This is intentionally a *separate* axis, not multiplied into the main
12-variant grid (which would explode to 48 × 5 = 240 runs).

---

## 7. Lesion-aware, source-aware splitting protocol

```
For each dataset:
  1. If lesion_id available: GroupShuffleSplit on lesion_id
       → train (70%) / val_select (10%) / val_calibrate (5%) / test (15%)
  2. If only patient_id: GroupShuffleSplit on patient_id (same fractions)
  3. If neither (Fitzpatrick17k, Derm7pt): random stratified by class

For cross-dataset evaluation:
  - Leave-one-dataset-out (LODO) on the dominant 4 datasets
  - Reports: (train: N-1 datasets, val: internal subset of training, test: held-out dataset Y)
```

**Two distinct validation subsets** (critical for honest calibration):

| Subset | Purpose | Used by |
|---|---|---|
| `val_select` (10%) | Hyperparameter tuning, early stopping, model selection | training loop |
| `val_calibrate` (5%) | Temperature/Dirichlet fit, threshold tuning, SLD effective prior | post-training |

These never overlap. Otherwise calibration is double-dipping on the same
val set that selected the model, producing overconfident calibration
artefacts.

**Statistical rigor for the top-3 fusion variants** (final ablation
only): re-run with `StratifiedGroupKFold(n_splits=5)` on `lesion_id` to
report 5-fold mean ± std for the headline table. The main 12-variant
sweep uses a single split for compute feasibility.

The hackathon split error (label-stratified only) is the **first thing
to never repeat**.

---

## 8. Heterogeneous-metadata training framework

```
def get_metadata_vector(row, dataset_id):
    """Return a canonical (D,) tensor + (D,) mask of presence."""
    canon = torch.zeros(CANONICAL_DIM)
    mask  = torch.zeros(CANONICAL_DIM, dtype=torch.bool)
    for field, value in row.items():
        canon_idx = SCHEMA_MAP[dataset_id][field]   # may be None
        if canon_idx is not None and value is not None:
            canon[canon_idx] = encode(field, value)
            mask[canon_idx]  = True
    return canon, mask
```

In the model:
```python
patient_embedding = SchemaAligner(canon, mask)
# SchemaAligner replaces each missing field with a learnable "absent" embedding,
# then runs an MLP on the resulting full-length vector.
```

This lets a single model train on ISIC 2019 (3 features), HAM10000 (3
features), PAD-UFES-20 (20 features) simultaneously without per-dataset
heads.

---

## 9. Calibration & adaptation layer (carried over, re-derived)

| Component | Mechanism | Fit on |
|---|---|---|
| Per-class temperature scaling | LBFGS-fit, one scalar per class | `val_calibrate` only |
| Dirichlet calibration | LBFGS-fit `W` (K×K) + `b` (K) | `val_calibrate` only |
| SLD prior adaptation | EM-estimate test prior from unlabelled test predictions | unlabelled test logits |
| Effective train prior | Mean softmax over training set | training set |
| 8-way Test-Time Augmentation | Average logits across original + H-flip + V-flip + HV-flip + 4 colour jitters | inference time |

Importantly:
- These are re-fit *per fusion variant* and *per dataset combination*.
- The hackathon's checkpoints under `backend/checkpoints/` are **not**
  reused — those were calibrated against a leaky split.
- The calibration fit set (`val_calibrate`) is strictly disjoint from
  the model-selection set (`val_select`); double-dipping produces
  optimistically-low ECE estimates that don't transfer.

### Ensemble averaging (when ensembling V1..V8 final predictions)

- Logit-space averaging (mathematically sound; mean-of-logits ≠ mean-of-
  probs because softmax is non-linear)
- Weights `w_i` per variant are **learned on `val_calibrate`** by LBFGS
  to maximize macro-F1, subject to `w_i >= 0` and `Σ w_i = 1`
- Reduces to uniform averaging if no fusion variant dominates

---

## 10. Evaluation protocol

### Primary metrics
- Macro-F1 (val and test)
- AUC-ROC (val and test)
- Per-class F1, sensitivity, specificity
- Expected Calibration Error (overall + per-class)
- **val→test macro-F1 gap** (the key generalization metric)

### Secondary metrics
- Per-feature, per-class **integrated gradient attribution** of metadata
  contribution (RQ4)
- **Missing-feature robustness curve**: drop each metadata field at
  inference, measure per-class F1 drop (RQ5)
- **Cross-dataset transfer matrix**: 4×4 LODO matrix of test F1 when
  training excludes dataset *i* and testing on dataset *i*
- **Calibration vs prior-shift attribution**: ECE before/after each
  component of the post-processing pipeline
- **Per-lesion vs per-image aggregation**: for test cases with multiple
  images of the same lesion, report both per-image F1 and per-lesion F1
  (majority vote and mean-prob aggregation). Per-lesion is the clinically
  meaningful metric

### Fairness audit (required for the thesis to be defensible)
- **Per-skin-tone F1**: stratified Macro-F1 and per-class MEL recall on
  Fitzpatrick I–II vs III–IV vs V–VI (using Fitzpatrick17k + PAD-UFES-20
  test subsets). A model that excels on F-I and fails on F-V is not
  publishable, regardless of overall F1
- **Per-sex / per-age-band** stratified metrics; flag any group with
  > 0.10 macro-F1 deficit
- **Equalized odds gap** for the malignant/benign binary head between
  skin-tone strata

### Clinical-decision metrics (deferral / abstention)
- **Selective accuracy at coverage**: accuracy curve as a function of
  fraction of cases accepted (by confidence). At 70% coverage, what
  accuracy is achievable? At 50%?
- **MEL miss rate at confidence ≥ X**: clinically critical
- **Risk-coverage curve** for human-in-the-loop deployment

### Statistical reporting
- Each variant trained with **5 seeds**
- Mean ± std reported for every metric
- Paired bootstrap CI for variant-vs-baseline differences
- McNemar's test for argmax differences

### Ablation tables (mandatory)
- V0 vs V1..V8 (which mechanism wins)
- D0 vs D1 vs D2 vs D3 (which depth wins)
- with/without external datasets
- with/without prior adaptation
- with/without per-feature dropout

---

## 11. Out-of-scope (explicit, to stay focused)

The following are real but **excluded from this thesis** to keep the
study tractable and the contribution sharp:

| Excluded | Reason |
|---|---|
| Knowledge distillation, self-distillation | Independent contribution; muddies metadata-fusion narrative |
| Optimizer experiments (Lion, SAM, Lookahead) | Confound the fusion-mechanism comparison |
| Self-supervised pretraining (DINO, MAE) — main scope | Multi-month side project; would confound the fusion-mechanism comparison by changing the initialization budget |
| Synthetic data generation (StyleGAN3, ControlNet, diffusion) | Already covered by a parallel ISIC research line; we use existing synthetic samples |
| Architecture search (EVA-02, ConvNeXt-V2, multi-scale, B5/B6 upgrade, third backbone) | **Backbones held constant at EfficientNet-B4 + SwinV2-Base for fair fusion comparison.** Varying backbone confounds the mechanism comparison. Reserved for a follow-up paper. |
| Different augmentation per ensemble member | Useful for ensemble *diversity*, but our ensemble varies *fusion mechanism* (V1..V8) — augmentation is held constant for clean attribution |
| Capsule networks, exotic pooling | Marginal expected gain, high engineering risk |
| Segmentation auxiliary head | Real signal but segmentation masks are not available across all 5+ datasets |
| Adversarial training | Robustness study, not fusion study |
| Active learning, pseudo-labelling | Different research direction |
| Open-set / novelty detection for the UNK class | Standalone problem; the 8-class closed-set is hard enough for this thesis |
| Knowledge distillation, self-distillation, SAM, Lookahead, Lion optimizer | Optimizer / regularization study; orthogonal to fusion |

### 11.1 Optional appendix-only experiments (only if main results land by week 12)

| Experiment | Purpose |
|---|---|
| SSL pretraining with MAE on 200K unlabelled dermoscopy images, fine-tune top-2 fusion variants | Demonstrate that fusion gains are *additive* with backbone improvements; not replicate Tier-A SSL papers |
| Test-Time Training (per-image SGD update on confidence-based pseudo-label) | Final-mile domain adaptation; reported as ablation only |

These are **appendix candidates**, not core deliverables. Run only if
weeks 14–15 of the timeline have buffer.

---

## 12. Reproducibility infrastructure

| Folder | Contents |
|---|---|
| `research/configs/` | YAML config per variant + seed |
| `research/datasets/` | Per-dataset download and indexing scripts (no images committed) |
| `research/splits/` | Patient-aware splits as CSVs, deterministically generated |
| `research/models/` | One Python module per fusion variant (V0–V11) |
| `research/training/` | Single training loop, fully configurable from YAML |
| `research/evaluation/` | ECE, attribution, cross-dataset transfer, statistical tests |
| `research/experiments/` | Per-experiment results: metrics JSON + reliability PNGs |
| `research/notebooks/` | Analysis notebooks, deferred (Jupyter) |
| `research/references/` | Cited papers, summaries |
| `research/thesis/` | LaTeX manuscript drafts |

Every experiment writes a `manifest.yaml` containing dataset hash, code
git SHA, env hash, seed, and exact config. Re-running the same manifest
should reproduce results bit-for-bit on the same GPU.

---

## 13. Timeline (16-week semester)

| Weeks | Phase | Output |
|---|---|---|
| 1–2 | Lit review + dataset download + cleaning | `references/` summary doc; `datasets/` indexed |
| 3 | Lesion-aware split protocol + schema-aligner | Splits CSV; `SchemaAligner` module unit-tested |
| 4–5 | Implement variants V0–V11 | All 12 models load and forward-pass; smoke tests pass |
| 6–8 | First training sweep on ISIC 2019 only | Per-variant baseline numbers; pick top 4 for full sweep |
| 9–10 | Cross-dataset training of top 4 variants | LODO transfer matrix populated |
| 11 | Calibration + adaptation layer | ECE comparison table |
| 12 | Attribution analysis (RQ4) | Per-feature, per-class plots |
| 13 | Missing-metadata robustness (RQ5) | Robustness curves |
| 14 | Statistical testing + tables | Final results pinned |
| 15 | Thesis writing | Draft submitted to advisor |
| 16 | Revision + presentation | Final submission |

---

## 14. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| ISIC 2019 ↔ HAM10000 image overlap inflates "external" datasets | High | De-duplicate via perceptual hash before training |
| Bigger models overfit small rare-class subsets | High | Per-class effective-sample reweighting; rare-class oversampling |
| `dataset_source` token gets used as a shortcut by the model | Medium | Gradient reversal on source classifier head (DANN-lite) |
| Single-seed reporting → reviewer rejection | High | Run 5 seeds *from week 6*, not after the fact |
| Cross-dataset patient leakage missed | Medium | Cross-reference all available patient IDs across datasets first |
| PAD-UFES-20 distribution very different from ISIC | High | This is *the point*; report performance as transfer score, not absolute |
| Compute exhaustion | Medium | A6000 + careful batching; fall back to top-4 variants if needed |

---

## 15. Connection to the hackathon code

| Hackathon artifact | Status in research |
|---|---|
| `backend/classifier/inference.py` | **Not used.** Inference is rewritten in `research/inference/` to support all 12 variants without the calibration shortcuts. |
| `backend/checkpoints/*.npy` | **Not used.** Calibration is re-fit on the new splits. |
| `backend/data/layer0_*.csv` | **Reference only.** New splits in `research/splits/`. |
| `experimental/calibrate_dirichlet.py` | **Imported and reused as-is** (clean utility). |
| `backend/classifier/prior_adaptation.py` (SLD module) | **Imported and reused as-is.** |
| Frontend, FastAPI, ngrok, etc. | **Out of scope.** Research deliverable is the model + paper, not a deployment. |

This isolation lets the hackathon stay frozen-in-time as a working demo
while the research project builds independently in `research/`.

---

## 16. Open questions to settle before week 3

1. **Dataset access:** Confirm we can legally redistribute splits from
   PAD-UFES-20 + Fitzpatrick17k. If not, document the download procedure.
2. **Image deduplication:** Implement perceptual-hash-based dedup across
   all datasets before any split. Estimate inter-dataset overlap.
3. **Schema canonicalization:** Settle the canonical feature list once.
   Adding a feature later requires re-running every experiment.
4. **Compute budget:** Total expected GPU-hours = 12 variants × 5 seeds
   × 4 dataset configurations × 14 hr/run ≈ **3,360 GPU-hr**. Plan A6000
   availability accordingly. If insufficient, drop to top 4 variants × 3
   seeds × 3 dataset configs ≈ 504 GPU-hr.

---

## 17. Success criteria

The thesis is successful if **any one** of the following is achieved on
the LODO test protocol (training on N-1 datasets, testing on the held-
out one):

1. The best fusion variant beats the late-concat baseline by **≥ 0.04
   macro-F1** with p < 0.05.
2. The best variant **closes the val→test gap to ≤ 0.03** macro-F1
   while maintaining baseline AUC.
3. The schema-aligner enables training on 5+ datasets simultaneously
   and outperforms any single-dataset model on its respective test set.

Failing all three is also a publishable negative result, but framed as
*"metadata fusion mechanism choice does not significantly affect
generalization; the bottleneck lies elsewhere."*

---

*Document version: v1 — initial plan. Update DECISIONS.md (in this
folder) whenever a section materially changes.*
