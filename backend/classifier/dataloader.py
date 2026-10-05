# ============================================================
#  LesionIQ — DataLoader v2.0
# ============================================================

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from backend.classifier.config import NUM_WORKERS, TRAIN_CSV, VAL_CSV, TEST_CSV, AGE_NORM_FACTOR


# ─────────────────────────────────────────────────────────────
# Paths are loaded from backend/config.yaml via config.py.
# Override with env vars: LESIONIQ_TRAIN_CSV, LESIONIQ_VAL_CSV, LESIONIQ_TEST_CSV
IMAGE_DIR = ""  # Unused — image_path column in CSVs is absolute


# ─────────────────────────────────────────────────────────────
# SECTION 2: DON'T TOUCH — EXACT COLUMN NAMES FROM LAYER 0
LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
META_COLS = [
    "age_approx",
    "sex_female", "sex_male", "sex_unknown",
    "site_anterior torso",
    "site_head/neck",
    "site_lateral torso",
    "site_lower extremity",
    "site_oral/genital",
    "site_palms/soles",
    "site_posterior torso",
    "site_upper extremity",
    "site_unknown",
]


# ─────────────────────────────────────────────────────────────
# SECTION 3: AUGMENTATIONS — AGGRESSIVE VERSION FOR HIGH F1
TRAIN_TRANSFORMS = A.Compose([
    A.Resize(384, 384),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120*0.05, p=1.0),
        A.GridDistortion(p=1.0),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
    ], p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GaussNoise(p=0.2),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.4),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

VAL_TRANSFORMS = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class LesionDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        print(f"[LOAD] Loading: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"[OK] Loaded {len(self.df)} images")
        
        self.image_dir = image_dir
        self.transform = transform
        
        self.df["age_approx"] = self.df["age_approx"].fillna(45.0).clip(0, AGE_NORM_FACTOR) / AGE_NORM_FACTOR
        
        for col in META_COLS:
            if col not in self.df.columns:
                self.df[col] = 0.0
                print(f"[WARN] Added missing: {col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = row["image_path"]
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        # Parse boolean values manually before conversion to handle "True"/"False" strings
        meta_vals = []
        for col in META_COLS:
            val = row[col]
            if pd.isna(val) or val == "":
                meta_vals.append(0.0)
            elif str(val).lower() in ["true", "false"]:
                meta_vals.append(1.0 if str(val).lower() == "true" else 0.0)
            else:
                meta_vals.append(float(val))
        meta = torch.tensor(meta_vals, dtype=torch.float32)
        
        # FocalLoss expects a class index (0-7), not one-hot encoding
        label = torch.tensor(int(row["class_encoded"])).long()
        
        return image, meta, label


def get_dataloaders(batch_size=32, num_workers=NUM_WORKERS):
    train_dataset = LesionDataset(TRAIN_CSV, IMAGE_DIR, transform=TRAIN_TRANSFORMS)
    val_dataset   = LesionDataset(VAL_CSV,   IMAGE_DIR, transform=VAL_TRANSFORMS)
    test_dataset  = LesionDataset(TEST_CSV,  IMAGE_DIR, transform=VAL_TRANSFORMS)

    # ── Class-balanced sampling (CRITICAL for Macro-F1) ──
    # This forces the model to see rare classes (DF, VASC) as often as NV
    labels = train_dataset.df['class_encoded'].values
    class_counts = np.bincount(labels, minlength=8)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    print(f'[SAMPLER] Class counts: {class_counts}')
    print(f'[SAMPLER] Effective class weights: {np.round(class_weights / class_weights.sum(), 3)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader