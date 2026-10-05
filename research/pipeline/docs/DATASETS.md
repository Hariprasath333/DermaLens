# Dermoscopy Dataset Catalogue

> **Curated for the LesionIQ metadata-fusion thesis.** Datasets are tiered by
> their value to the specific research questions in `PLAN.md`.
>
> **None of these are downloaded automatically.** Each is licensed
> independently; the user must accept the source's terms.

---

## Tier 1 — Essential (must obtain)

These four datasets together span the metadata schema diversity, skin
tone diversity, and acquisition diversity needed for the thesis.

### 1.1 HAM10000 — *Tschandl, Rosendahl, Kittler 2018*

| Property | Value |
|---|---|
| Images | 10,015 |
| Classes | 7 (akiec, bcc, bkl, df, mel, nv, vasc) |
| Lesion-level IDs | ✓ (`lesion_id`) |
| Multi-image lesions | ✓ (essential for lesion-aware split validation) |
| Metadata | age, sex, localization, dx_type |
| Resolution | 600×450 (standardized) |
| Source | Mostly Vienna + Queensland; clinical practice |
| License | CC BY-NC 4.0 |
| Download | https://doi.org/10.7910/DVN/DBW86T (Harvard Dataverse) |
| Use in this thesis | Primary single-source baseline; multi-image lesion validation |

### 1.2 ISIC 2020 — *Rotemberg et al. 2021*

| Property | Value |
|---|---|
| Images | 33,126 |
| Classes | 9 (binary mel + multi-class) |
| Lesion-level IDs | ✓ (`isic_id` + `lesion_id`) |
| Multi-image lesions | ✓ (often 5-20 images per lesion via total body photography) |
| Metadata | age_approx, sex, anatom_site_general, diagnosis, benign_malignant |
| Resolution | variable, 1024×1024 standard |
| Source | International (multiple clinics, 2020 collection) |
| License | CC BY-NC 4.0 |
| Download | https://challenge.isic-archive.com/data/#2020 |
| Use in this thesis | Largest dermoscopy dataset; temporal-shift study (2019 vs 2020 acquisition); lesion-level aggregation experiments |

### 1.3 PAD-UFES-20 — *Pacheco et al. 2020*

| Property | Value |
|---|---|
| Images | 2,298 |
| Classes | 6 (akiec, bcc, mel, nv, sek, scc) |
| Acquisition | **Smartphone clinical photos** (NOT dermoscopy) |
| Lesion-level IDs | ✓ (`lesion_id` + `patient_id`) |
| Metadata | **22 fields**: age, gender, region, **Fitzpatrick skin type**, diameter_1, diameter_2, family_cancer_history, smoke, drink, pesticide, has_piped_water, has_sewage_system, itch, grew, hurt, changed, bleed, elevation, biopsied |
| Resolution | smartphone, variable |
| Source | Brazilian rural population (Espírito Santo); Federal University clinic |
| License | CC BY 4.0 (truly open) |
| Download | https://data.mendeley.com/datasets/zr7vgbcyr2/1 |
| Use in this thesis | **Richest metadata** in the catalogue; **non-dermoscopic** modality stress-test for SchemaAligner; underrepresented Brazilian patient distribution; Fitzpatrick labels |

### 1.4 Fitzpatrick17k — *Groh, Harris, Soenksen, Lau et al. 2021*

| Property | Value |
|---|---|
| Images | 16,577 |
| Classes | 114 skin conditions (not just lesion classification — broader dermatology) |
| Acquisition | Clinical photographs (textbook/atlas sources) |
| Lesion-level IDs | partial |
| Metadata | **Fitzpatrick skin type I–VI** (the only thing that matters for us) |
| Resolution | variable |
| Source | Web-scraped from dermatology textbooks (Atlas Dermatologico + DermNet NZ) |
| License | open (researcher-curated) |
| Download | https://github.com/mattgroh/fitzpatrick17k |
| Use in this thesis | **Fairness audit only** — used as external validation, not training. Skin-tone stratified per-class F1 |

---

## Tier 2 — High value (recommended)

### 2.1 Derm7pt — *Kawahara, Daneshvar, Argenziano, Hamarneh 2018*

| Property | Value |
|---|---|
| Images | 2,045 (image pairs: clinical + dermoscopic) |
| Classes | 5 + auxiliary 7-point checklist scores |
| Metadata | **7-point checklist:** Pigment Network, Streaks, Pigmentation, Regression Structures, Dots/Globules, Blue-Whitish Veil, Vascular Structures |
| Source | Argenziano interactive atlas |
| License | research-only (registration required) |
| Download | https://derm.cs.sfu.ca/ |
| Use in this thesis | **Multi-task auxiliary signals**: 7-point structures are clinically interpretable; can be predicted as side task |

### 2.2 BCN20000 — *Combalia et al. 2019*

| Property | Value |
|---|---|
| Images | 19,424 |
| Note | **Already included in ISIC 2019 training set** |
| Use in this thesis | Already covered; documented for completeness only |

### 2.3 ISIC 2024 — *3D Total-Body Photography crops*

| Property | Value |
|---|---|
| Images | 400K (large) |
| Source | Different acquisition: lesions cropped from 3D total-body scans, not handheld dermoscopy |
| Use in this thesis | **Acquisition-shift test**: model trained on handheld dermoscopy applied to TBP crops would stress the cross-clinic generalization claim |
| Risk | Lower per-image resolution; sparse metadata |
| Status | **Optional**; include only if compute budget allows |

---

## Tier 3 — External validation only (do not train on)

### 3.1 PH² — *Mendonça et al. 2013*

| Property | Value |
|---|---|
| Images | 200 |
| Use | Tiny external test set for sanity check; well-curated |
| License | research-only |

### 3.2 MED-NODE — too small (170 images). **Skip.**

### 3.3 DermNet / DermNet NZ web-scraped — quality issues, licensing unclear. **Skip.**

### 3.4 Argenziano Atlas — covered by Derm7pt in practice. **Skip.**

---

## Recommended training mix

| Configuration | Datasets | Rationale |
|---|---|---|
| **Baseline single-source** | ISIC 2019 only | Reproduces the hackathon for direct comparison |
| **Recommended for thesis** | ISIC 2019 + HAM10000 (full) + ISIC 2020 + PAD-UFES-20 | Schema-aligner stress-tested; 50K+ training images |
| **Fairness extension** | + Fitzpatrick17k (external test only) | Required for skin-tone audit |
| **Multi-task extension** | + Derm7pt | Adds 7-point auxiliary head |

## Cross-dataset deduplication required

| Overlap | Why it matters | Solution |
|---|---|---|
| HAM10000 ⊂ ISIC 2018 train ⊂ ISIC 2019 train | Same images counted multiple times | Hash-based dedup before split (script in stage1) |
| ISIC 2019 ⊃ HAM10000 | Some HAM samples are already in 2019 | Hash dedup |
| BCN20000 ⊂ ISIC 2019 | Same | Hash dedup |
| ISIC 2020 ↔ ISIC 2019 | Possible patient-level overlap | Patient-ID cross-reference after lesion_id check |

The dedup is done **before** any split. Otherwise a single image can
appear in train of one source and val of another, re-introducing the
leakage we are trying to eliminate.

---

## Storage estimate

| Dataset | Size on disk |
|---|---|
| HAM10000 | ~3.5 GB |
| ISIC 2020 | ~110 GB (1024² JPGs) |
| ISIC 2020 256-resized | ~14 GB |
| PAD-UFES-20 | ~3.2 GB |
| Fitzpatrick17k | ~5.5 GB |
| Derm7pt | ~1.0 GB |
| ISIC 2019 (already have) | ~25 GB |
| **Total recommended mix** | **~50 GB** (with ISIC 2020 resized) |

Use ISIC 2020 at 256² or 384² resolution to keep storage tractable. The
challenge organisers provide a 256² version officially.

---

## License footprint

All recommended training datasets allow **non-commercial research use**.
Plan B-style downstream commercial deployment would require re-licensing.
This is acceptable for the thesis.

---

## What is downloaded by this pipeline

**Nothing automatically.** `stages/stage1_datasets.py` only:

1. Lists datasets with metadata
2. Verifies whether each is already present on disk
3. Prints exact download instructions per dataset
4. Builds a `DATASET_REGISTRY` for the rest of the pipeline to discover
5. Computes hashes / counts to confirm correct download after the user has fetched files

The user runs the pipeline; the pipeline tells the user what to download
and where to place it. The user accepts each dataset's terms manually.
