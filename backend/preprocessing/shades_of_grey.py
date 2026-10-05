import cv2
import numpy as np
import os
from tqdm import tqdm

def shades_of_gray(img, power=4):
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)

    norm_r = np.power(np.mean(np.power(np.abs(r), power)), 1.0 / power)
    norm_g = np.power(np.mean(np.power(np.abs(g), power)), 1.0 / power)
    norm_b = np.power(np.mean(np.power(np.abs(b), power)), 1.0 / power)

    overall_norm = np.power((norm_r**power + norm_g**power + norm_b**power) / 3.0, 1.0 / power)

    eps = 1e-6
    r_corrected = r * (overall_norm / (norm_r + eps))
    g_corrected = g * (overall_norm / (norm_g + eps))
    b_corrected = b * (overall_norm / (norm_b + eps))

    corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return corrected


def process_sog_batch(input_folder, output_folder, power=4):
    os.makedirs(output_folder, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(valid_extensions)]

    print(f"Found {len(image_files)} images to process.")
    print(f"Input  : {input_folder}")
    print(f"Output : {output_folder}")
    print("-" * 50)

    failed_images = []
    skipped = 0

    for filename in tqdm(image_files, desc="Shades of Gray"):
        input_path  = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(output_path):
            skipped += 1
            continue

        try:
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError(f"Could not load: {input_path}")

            corrected = shades_of_gray(img, power=power)
            cv2.imwrite(output_path, corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])

        except Exception as e:
            failed_images.append((filename, str(e)))

    print("\n--- Processing Complete ---")
    print(f"Processed : {len(image_files) - len(failed_images) - skipped}")
    print(f"Skipped   : {skipped}  (already done)")
    print(f"Failed    : {len(failed_images)}")
    if failed_images:
        for name, err in failed_images:
            print(f"  ✗ {name}: {err}")


if __name__ == "__main__":
    # ── PATHS ─────────────────────────────────────────────────────────────────────
    INPUT_DIR  = r"path/to/processed"
    OUTPUT_DIR = r"path/to/shades_of_grey"
    # ─────────────────────────────────────────────────────────────────────────────
    process_sog_batch(INPUT_DIR, OUTPUT_DIR, power=4)