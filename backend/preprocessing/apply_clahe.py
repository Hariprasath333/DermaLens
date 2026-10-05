import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def apply_clahe(image_path, output_path=None):
    """
    Reads an image (already processed by DullRazor and Shades of Gray),
    applies CLAHE in the LAB color space, and saves/returns it.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # 1. Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 2. Apply CLAHE only to Lightness channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # 3. Merge back and convert to BGR
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    final_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    if output_path:
        cv2.imwrite(output_path, final_image)
        
    return final_image


def test_single_image(input_path):
    """Shows the Before CLAHE vs After CLAHE comparison for one image."""
    original = cv2.imread(input_path)
    if original is None:
        raise ValueError(f"Cannot load image: {input_path}")
        
    after_clahe = apply_clahe(input_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input (After Shades of Gray)", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(after_clahe, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Output (+ CLAHE)", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def batch_process_clahe(input_folder, output_folder):
    """Applies CLAHE to a whole folder of images."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    valid_ext = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]
    
    print(f"Found {len(image_files)} images to enhance with CLAHE.")
    
    failed = []
    for filename in tqdm(image_files, desc="Applying CLAHE"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        if os.path.exists(output_path):
            continue
            
        try:
            apply_clahe(input_path, output_path=output_path)
        except Exception as e:
            failed.append(filename)
            
    print(f"\nDone! Processed {len(image_files) - len(failed)}/{len(image_files)} images.")


if __name__ == "__main__":
    # ── DESKTOP BATCH RUN ──────────────────────────────────
    
    # Input: The folder where your Shades of Gray processed images currently are
    INPUT_DIR  = r"path/to/shades_of_grey"
    
    # Output: A new folder named 'clahe' right under the preprocesing directory
    OUTPUT_DIR = r"path/to/clahe"
    
    print(f"Reading from: {INPUT_DIR}")
    print(f"Saving to:    {OUTPUT_DIR}")
    print("-" * 50)
    
    # Run the batch process
    batch_process_clahe(INPUT_DIR, OUTPUT_DIR)