"""
LesionIQ Preprocessing Pipeline — Public API
==============================================
Exposes the 4-step Layer 0 preprocessing pipeline used during both
training and inference:

    1. DullRazor hair removal
    2. Shades of Gray color normalization (power=4)
    3. CLAHE contrast enhancement (LAB L-channel, clipLimit=2.0)
    4. Circular border removal + crop

Usage (inference):
    from preprocessing import run_pipeline
    img_bgr = run_pipeline("path/to/lesion.png", target_size=384)
"""

import cv2
import numpy as np

from backend.preprocessing.dull_razor import dullrazor
from backend.preprocessing.shades_of_grey import shades_of_gray
from backend.preprocessing.apply_clahe import apply_clahe
from backend.preprocessing.remove_circular_border import has_circular_border, inscribed_square


def _apply_clahe_to_array(img_bgr, clip_limit=2.0):
    """Apply CLAHE to a BGR numpy array (in-memory, no file I/O).

    The file-based ``apply_clahe`` reads from disk.  This variant
    operates on an already-loaded array so the full pipeline can run
    without intermediate file writes.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_cl = clahe.apply(l_ch)
    merged = cv2.merge((l_cl, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def run_pipeline(image_path, target_size=384):
    """Run the full Layer 0 preprocessing pipeline on a single image.

    Matches the exact sequence used to prepare the training set:
        1. DullRazor hair removal
        2. Shades of Gray color normalization (power=4)
        3. CLAHE contrast enhancement (LAB L-channel)
        4. Circular border removal + inscribed-square crop
        5. Resize to ``target_size × target_size``

    Parameters
    ----------
    image_path : str or Path
        Path to the input dermoscopy image.
    target_size : int
        Output resolution (square).  Default 384 to match model input.

    Returns
    -------
    img_bgr : np.ndarray
        Preprocessed BGR image (uint8), shape ``(target_size, target_size, 3)``.
    """
    # Step 1: DullRazor
    img_bgr, _ = dullrazor(str(image_path))

    # Step 2: Shades of Gray
    img_bgr = shades_of_gray(img_bgr, power=4)

    # Step 3: CLAHE
    img_bgr = _apply_clahe_to_array(img_bgr)

    # Step 4: Circular border removal
    circle = has_circular_border(img_bgr)
    if circle is not None:
        cx, cy, radius = circle
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = inscribed_square(cx, cy, radius, h, w)
        img_bgr = img_bgr[y1:y2, x1:x2]

    # Step 5: Resize
    img_bgr = cv2.resize(img_bgr, (target_size, target_size),
                         interpolation=cv2.INTER_LANCZOS4)

    return img_bgr
