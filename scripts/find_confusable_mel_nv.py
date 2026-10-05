"""Find a visually confusable MEL/NV pair from images with circular vignette."""
import os
from pathlib import Path
import torch

from backend.classifier.inference import (
    _load_runtime, predict, preprocess_image, encode_metadata,
)

ROOT = Path(r'C:\LesionIQ\dataset\ISIC_2019_Test_Input\ISIC_2019_Test_Input')

mel_ids = ['ISIC_0053460', 'ISIC_0053481', 'ISIC_0053489',
           'ISIC_0053605', 'ISIC_0053632']
nv_ids  = ['ISIC_0053805', 'ISIC_0053902', 'ISIC_0053942',
           'ISIC_0054045', 'ISIC_0054095', 'ISIC_0054175']

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

runtime = _load_runtime(mode='full', checkpoint_path=None)
# Unpack the first 5 known fields; ignore the rest (prior-shift extras applied
# downstream, not inside predict()).
model, scales, T, mel_thr, pcT = runtime[:5]


def score(image_id):
    p = str(ROOT / f'{image_id}.jpg')
    img = preprocess_image(p)
    meta = encode_metadata(60, 'unknown', 'unknown')
    probs = predict(model, img, meta, T, scales, pcT)
    return probs


def show(label, ids):
    print(f"\nTRUE LABEL: {label}")
    print(f"{'image':<18}  {'MEL':>6}  {'NV':>6}  {'BCC':>6}  pred")
    print('-' * 55)
    for id_ in ids:
        pr = score(id_)
        pred = CLASS_NAMES[pr.argmax()]
        print(f'{id_:<18}  {pr[0]:6.3f}  {pr[1]:6.3f}  {pr[2]:6.3f}  -> {pred}')


show('MEL', mel_ids)
show('NV',  nv_ids)
