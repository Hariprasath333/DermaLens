import cv2
import numpy as np
import os
from tqdm import tqdm

def dullrazor(image_path, output_path=None):
    """
    Robust DullRazor using directional morphological line filters.
    Highly effective for both fine and thick hairs in dermoscopy.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Multi-scale Blackhat filtering in multiple directions (0, 45, 90, 135)
    kernel_length = 17  
    max_blackhat = np.zeros_like(gray)
    
    for angle in [0, 45, 90, 135]:
        kernel = np.zeros((kernel_length, kernel_length), dtype=np.uint8)
        center = kernel_length // 2
        
        if angle == 0:
            kernel[center, :] = 1
        elif angle == 90:
            kernel[:, center] = 1
        elif angle == 45:
            np.fill_diagonal(kernel, 1)
        elif angle == 135:
            np.fill_diagonal(np.fliplr(kernel), 1)
            
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        max_blackhat = cv2.max(max_blackhat, blackhat)
    
    _, hair_mask = cv2.threshold(max_blackhat, 10, 255, cv2.THRESH_BINARY)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_mask = cv2.dilate(hair_mask, kernel_dilate, iterations=1)
    
    cleaned = cv2.inpaint(img, hair_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    if output_path:
        cv2.imwrite(output_path, cleaned)
    
    return cleaned, hair_mask


def process_entire_dataset(input_folder, output_folder):
    """Loops through dataset and applies DullRazor."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_files = os.listdir(input_folder)
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]
    
    print(f"Found {len(image_files)} images to process.")
    
    failed_images = []

    for filename in tqdm(image_files, desc="Applying DullRazor"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Skip if already processed so you can pause/resume anytime
        if os.path.exists(output_path):
            continue
            
        try:
            dullrazor(input_path, output_path=output_path)
        except Exception as e:
            failed_images.append((filename, str(e)))

    print("\n--- Processing Complete ---")
    if failed_images:
        print(f"Failed to process {len(failed_images)} images. (Check paths/corrupt files)")


if __name__ == "__main__":
    # Your exact folder paths
    INPUT_DIR = r"path/to/ISIC_2019_Training_Input"
    OUTPUT_DIR = r"path/to/processed"
    
    print(f"Reading from: {INPUT_DIR}")
    print(f"Saving to: {OUTPUT_DIR}")
    print("-" * 50)
    
    process_entire_dataset(INPUT_DIR, OUTPUT_DIR)