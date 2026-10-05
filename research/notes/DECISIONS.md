# Decision Log

A running log of key research-design decisions, the alternatives
considered, and the rationale. New decisions appended at the top.

---

## 2026-06-09 — Pipeline correctness audit (cross-check vs. old-model findings)

Audited `research/pipeline/` against an 8-point analysis of the old
hackathon model. Two of the new bugs were the same *class* of defect the
analysis flagged ("reads as active but isn't"). Fixed in pipeline
**v1.2.0**.

### Bugs fixed

- **Bug A — EMA never restored after eval** (`stages/stage6_train.py`).
  The per-epoch EMA evaluation overwrote the live training weights and
  never restored them, so training silently continued from the lagged
  EMA snapshot and `best.pt` saved EMA weights while selecting on live
  metrics. Fix: added `EMA.store()` / `EMA.restore()`; the epoch loop now
  does eval-live → store → copy-EMA → eval-EMA → **restore**, selects the
  better of live/EMA, and saves *the weights that were actually scored*
  with a `weights_source` tag. Mirrors old-finding #1.

- **Bug B — token injection silently disabled on real timm Swin**
  (`models/variants.py` V4/V8/V11, `models/backbones.py`). The old code
  appended metadata tokens *before* the Swin encoder and relied on an
  `.encode()` method that exists only on the tiny CPU fallback — so under
  real timm the meta tokens were discarded and V4 trained image-only
  (zero gradient to metadata). This silently invalidated the variants
  most central to the thesis. Fix: new `TokenFusion` injector operates on
  the **post-encoder** flat token sequence `(B,L,C)` (a self-attention
  block over `[patch_tokens ; meta_tokens]`, pool patches) — works on
  timm and carries gradient. Exact recurrence of old-finding #1.

- **Latent reshape bug (found while fixing B)** — `swin_features` assumed
  channels-first and mis-reshaped timm SwinV2's channels-last
  `(B,H,W,C)` output, which would corrupt/crash *every* variant under
  real timm. Replaced with a single robust `swin_feature_tokens` that
  handles channels-last / channels-first / already-flat.

- **Bug C — CutMix blended metadata across patients**
  (`stages/stage6_train.py::mix_batch`). Linearly averaging two
  patients' age/sex/site/skin-tone describes no real person and injects
  noise into the studied pathway. Fix: `mix_batch` mixes the **image
  only**; the primary sample's metadata passes through unchanged (the
  soft label already accounts for the image mix). Addresses old-finding
  #7's metadata facet.

### Regression guards added (`run.py selftest`)

- **Metadata gradient-flow guard** — every variant V0..V11 must route
  nonzero gradient to its `meta` input after a few bootstrap steps
  (bootstrap needed because FiLM/CBN use identity init = zero gradient at
  step 0). A meta-ignoring variant now fails loudly. This is the test
  that would have caught Bug B. Verified to have teeth against a
  deliberately broken meta-ignoring model.
- **EMA store/restore invariant** — asserts copy_to swaps the shadow in
  and restore returns the exact live weights. Regression test for Bug A.

### Cross-check of the 8 old-model findings vs. the new pipeline

| # | Old finding | New-pipeline status |
|---|---|---|
| 1 | Aux head half-wired | Literal form absent; same pathology recurred as **Bug B** → fixed. |
| 2 | Age train/serve skew | Eliminated — single `encode_row_metadata`; NA → learned absent embedding. |
| 3 | Fusion magnitude swamp | By design — V1..V11 are the fixes; V0 stays swamped as the honest baseline. |
| 4 | `1/count` sampler | Already `1/sqrt(count)`. |
| 5 | Tuned-on-test contamination | Already fixed — `val_select ⊥ val_calibrate ⊥ test`, lesion-grouped. |
| 6 | Dual backbone marginal | Deferred by design (backbones held constant for fair fusion comparison). |
| 7 | CutMix label noise | Meta-blend facet fixed (**Bug C**); centered-bbox risk noted, accepted for now. |
| 8 | Stacked imbalance mechanisms | Loss is an ablation axis (L0–L3); default stacks but is measurable — no code change. |

### Decisions

- **Do not "fix" V0's concat swamp** — V0 is the deliberate baseline that
  demonstrates the swamp the other variants solve; normalizing it would
  move the reference the whole study is measured against.
- **Do not mix metadata in MixUp/CutMix** — image-only mixing is the
  honest conditioning signal.
- **No changes to `backend/` or `frontend/`** — the hackathon stays frozen.

---

## 2026-06-08 — Integration of external "Model B optimization plan"

A second optimization plan was provided after PLAN.md v1 was written.
Each technique was evaluated against the existing fusion-thesis scope.
Below is the disposition.

### Integrated into PLAN.md

| Technique from Plan B | Where integrated | Notes |
|---|---|---|
| Group-aware splitting via `StratifiedGroupKFold(5)` | §7 (final-ablation statistical rigor for top-3 variants) | Main 12-variant sweep keeps single `GroupShuffleSplit` for compute feasibility; 5-fold reserved for the headline table only |
| Separate calibration set from model-selection set | §7 (val split now: `val_select` 10% + `val_calibrate` 5%); §9 reaffirms | Closes double-dipping bug that produces optimistically-low ECE estimates |
| F1-Loss / MCC-Loss / soft-F1 vs Focal | §6.5 (new secondary ablation axis) | Run on top-3 fusion variants × {Focal, CB-Focal, soft-F1, LDAM} × 5 seeds = 60 runs. Reported as 3×4 matrix |
| Learned ensemble weights via val optimization | §9 (Ensemble averaging subsection) | Logit-space, not soft-voting (mathematically cleaner). Weights LBFGS-fit on `val_calibrate` |
| 8-way TTA at inference | §9 (calibration table row) | Original + H-flip + V-flip + HV-flip + 4 colour jitters |
| SSL pretraining (SimCLR / MAE) | §11.1 appendix-only | Demoted from "core" to "optional weeks 14–15" — would confound the fusion-mechanism comparison if treated as primary |

### Explicitly rejected

| Plan B technique | Reason for exclusion |
|---|---|
| Add third backbone (ConvNeXt-B / MobileViT / EffNetV2-B3) | Backbones are held **constant** at EffNet-B4 + SwinV2-Base for the fusion comparison. Varying backbone confounds attribution of gains. Reserved as follow-up paper. |
| EfficientNet-B4 → B5/B6 upgrade | Same reason; backbone capacity is held constant |
| Different augmentation per ensemble member | Useful for diversity but our ensemble *varies fusion mechanism* (V1..V8). Augmentation must be held constant for clean attribution. Worth a footnote in the paper, not a deliverable. |
| TTA expectation of "uncertainty 69.6% → 30–40%" | The 69.6% comes from a confidence threshold of 0.70 against a model with strong prior shift; the cure is calibration + prior adaptation, not TTA. TTA helps but the framing is misleading. We report selective accuracy at coverage instead (§10 deferral metrics) |
| Final claim "works across any test composition (no re-tuning needed)" | False without prior adaptation. SLD or oracle prior is needed. Plan B understates this; the thesis is explicit about it |

### Additions I made beyond both plans (final scan)

These came out of a full re-scan after seeing Plan B:

1. **Per-skin-tone fairness audit** (§10) — required for any medical paper today; Fitzpatrick17k + PAD-UFES-20 enable it
2. **Per-lesion vs per-image aggregation metric** (§10) — multi-shot lesions in test must be reported correctly; majority vote and mean-prob both
3. **Selective accuracy at coverage** + risk-coverage curves (§10) — clinical deferral is the real deployment metric
4. **`val_calibrate` strictly disjoint from `val_select`** (§7) — closes a methodological hole present in both plans
5. **Plan B Open-set / UNK handling** explicitly noted as out-of-scope (§11) — to avoid scope creep mid-semester

---

## 2026-06-08 — Initial plan (v1)

PLAN.md v1 written. Thesis statement: metadata-fusion mechanism × injection
depth as the primary axis of study, with a heterogeneous-metadata
schema-aligner as the cross-dataset enabler. 12 fusion variants
cherry-picked from the 8 × 4 grid. 5 datasets in scope.

No prior decisions to record.

---

## Convention

When making a non-trivial choice, append an entry at the top with:
- Date
- The decision (one sentence)
- Alternatives considered
- Rationale (cite §X of PLAN.md if relevant)

Goal: future-Shreyas can answer "why did we do X and not Y?" without
re-deriving from scratch.
