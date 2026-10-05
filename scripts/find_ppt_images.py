"""Find ONE clean example each of AK, SCC, DF, VASC from validate_images
for the final ppt. Uses the ISIC GroundTruth CSVs for labels and the
inference pipeline to verify the model agrees confidently.

Output: prints top candidates per class with predicted probabilities so
the user can pick the visually best one.
"""
from pathlib import Path
import os, re, sys
import pandas as pd
import numpy as np

REPO = Path(r"C:\LesionIQ")
sys.path.insert(0, str(REPO))

FOLDER = Path(r"C:\Users\Shreyas\Desktop\validate_images_zip\validate_images")
TRAIN_GT = REPO / "backend" / "data" / "ISIC_2019_Training_GroundTruth.csv"
TEST_GT  = REPO / "backend" / "data" / "ISIC_2019_Test_GroundTruth.csv"
TRAIN_META = REPO / "backend" / "data" / "ISIC_2019_Training_Metadata.csv"
TEST_META  = REPO / "backend" / "data" / "ISIC_2019_Test_Metadata.csv"

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
TARGETS = ["AK", "SCC", "DF", "VASC"]

# ─── 1. Build {isic_id -> (label, downsampled_flag, file_path)} ────
print("[STEP] Scanning validate_images folder...")
files = list(FOLDER.glob("*.jpg"))
print(f"[OK]   Found {len(files)} files in {FOLDER}")

def norm_id(fname: str) -> str:
    """ISIC_0000043_downsampled.jpg -> ISIC_0000043"""
    return re.sub(r"_downsampled$", "", Path(fname).stem)

id_to_file = {norm_id(f.name): f for f in files}
print(f"[OK]   Unique ISIC IDs: {len(id_to_file)}")

# ─── 2. Merge GroundTruth from both train and test ────────────────
print("\n[STEP] Loading GroundTruth + Metadata...")
gt_train = pd.read_csv(TRAIN_GT)
gt_test  = pd.read_csv(TEST_GT)
# Keep only image + 8 label cols + UNK
label_cols = CLASS_NAMES + ["UNK"]
gt = pd.concat([
    gt_train[["image"] + label_cols].assign(source="train"),
    gt_test[["image"] + label_cols].assign(source="test"),
], ignore_index=True)

meta_train = pd.read_csv(TRAIN_META)[["image", "age_approx", "anatom_site_general", "sex"]]
meta_test  = pd.read_csv(TEST_META)[["image", "age_approx", "anatom_site_general", "sex"]]
meta = pd.concat([meta_train, meta_test], ignore_index=True).drop_duplicates("image")

# Filter to images we actually have
gt = gt[gt["image"].isin(id_to_file.keys())].copy()
gt = gt.merge(meta, on="image", how="left")
gt = gt[gt["UNK"] != 1.0]  # 8-class only
gt["label"] = gt[CLASS_NAMES].idxmax(axis=1)
print(f"[OK]   Matched + labelled rows: {len(gt)}")

# ─── 3. Counts per target class ───────────────────────────────────
print("\n[STEP] Available images per target class:")
for cls in TARGETS:
    n = (gt["label"] == cls).sum()
    print(f"   {cls:<5}  {n:>4}  (source split: "
          f"{(gt[(gt['label']==cls) & (gt['source']=='train')]).shape[0]} train / "
          f"{(gt[(gt['label']==cls) & (gt['source']=='test')]).shape[0]} test)")

# ─── 4. Score with model to find high-confidence examples ─────────
print("\n[STEP] Loading model for confidence ranking...")
from backend.classifier.inference import (
    _load_runtime, predict, preprocess_image, encode_metadata,
)
runtime = _load_runtime(mode="full", checkpoint_path=None)
model, scales, T, mel_thr, pcT = runtime[:5]


def score_image(img_path, age, sex, site):
    img = preprocess_image(str(img_path))
    age_v = age if isinstance(age, (int, float)) and not np.isnan(age) else None
    sex_v = sex if isinstance(sex, str) and sex.strip() else None
    site_v = site if isinstance(site, str) and site.strip() else None
    meta_t = encode_metadata(age_v, sex_v, site_v)
    return predict(model, img, meta_t, T, scales, pcT)


print("\n" + "=" * 72)
print(" TOP CANDIDATES PER CLASS  (sorted by model confidence on true class)")
print("=" * 72)
for cls in TARGETS:
    cls_idx = CLASS_NAMES.index(cls)
    pool = gt[gt["label"] == cls].copy()
    # Prefer non-downsampled originals when available
    pool["downsampled"] = pool["image"].apply(
        lambda x: "_downsampled" in id_to_file[x].name)
    pool = pool.sort_values("downsampled")  # False first
    pool = pool.head(50)  # limit scoring

    scored = []
    for _, row in pool.iterrows():
        path = id_to_file[row["image"]]
        try:
            probs = score_image(path, row["age_approx"], row["sex"],
                                row["anatom_site_general"])
            scored.append({
                "image": row["image"],
                "path": str(path),
                "downsampled": "_downsampled" in path.name,
                "true_prob": float(probs[cls_idx]),
                "pred": CLASS_NAMES[probs.argmax()],
                "site": row["anatom_site_general"],
                "age": row["age_approx"],
                "sex": row["sex"],
            })
        except Exception as e:
            continue

    scored.sort(key=lambda x: x["true_prob"], reverse=True)
    print(f"\n--- {cls}  (top 5 of {len(scored)} scored) ---")
    print(f"{'image':<22}  {'P(true)':>7}  {'pred':<5}  "
          f"{'ds':<3}  {'site':<18}  {'sex':<7}  age")
    for c in scored[:5]:
        ds = "Y" if c["downsampled"] else "N"
        site = str(c["site"])[:18]
        sex = str(c["sex"])[:7]
        print(f"{c['image']:<22}  {c['true_prob']:>7.3f}  "
              f"{c['pred']:<5}  {ds:<3}  {site:<18}  {sex:<7}  {c['age']}")

print("\n" + "=" * 72)
print(" Done. Pick the visually best per class from the top candidates above.")
print("=" * 72)
