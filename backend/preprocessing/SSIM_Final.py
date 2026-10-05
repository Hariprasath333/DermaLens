import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# ==========================================
# 1. PLACEHOLDERS - UPDATE THESE PATHS!
# ==========================================
DIR_RAW       = r"path/to/ISIC_2019_Training_Input"
DIR_PROCESSED = r"path/to/processed"
DIR_CLAHE     = r"path/to/clahe"
DIR_CROPPED   = r"path/to/cropped"
DIR_FINAL     = r"path/to/final"
OUTPUT_DIR    = r"path/to/SSIM_reports"  # Folder to save the 5 graphs

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. AUDIT FUNCTION
# ==========================================
def run_audit_step(step_name, dir_a, dir_b, output_chart_name, apply_crop_to_a=False):
    print(f"\n{'='*50}")
    print(f"Running Audit: {step_name}")
    print(f"{'='*50}")
    
    # Get all valid images
    valid_exts = ('.jpg', '.jpeg', '.png')
    files_a = set([f for f in os.listdir(dir_a) if f.lower().endswith(valid_exts)])
    files_b = set([f for f in os.listdir(dir_b) if f.lower().endswith(valid_exts)])
    
    # Only process files that exist in both folders to avoid crash
    common_files = list(files_a.intersection(files_b))
    total_files = len(common_files)
    
    if total_files == 0:
        print("❌ Error: No matching files found between the two folders.")
        return

    print(f"Found {total_files} matching images. This may take a while...")
    
    scores = []
    
    for i, filename in enumerate(common_files):
        # Print progress every 1000 images so you know it hasn't frozen
        if i % 1000 == 0 and i > 0:
            print(f"  ...processed {i}/{total_files} images")
            
        path_a = os.path.join(dir_a, filename)
        path_b = os.path.join(dir_b, filename)
        
        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        
        # If this step involves a cropped image, crop the reference image (A) 
        # by 6% to match the physical area of image B before comparing
        if apply_crop_to_a:
            h, w = img_a.shape[:2]
            crop_h, crop_w = int(h * 0.06), int(w * 0.06)
            img_a = img_a[crop_h:h-crop_h, crop_w:w-crop_w]
            
        # Resize both to 224x224 so SSIM can compare pixel-to-pixel perfectly
        img_a = cv2.resize(img_a, (224, 224))
        img_b = cv2.resize(img_b, (224, 224))
        
        # Calculate SSIM (channel_axis=2 handles RGB colors, data_range=255 for standard 8-bit images)
        score, _ = ssim(img_a, img_b, channel_axis=2, data_range=255, full=True)
        scores.append(score)
        
    # Stats
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    print("\n── Results ──────────────────────────────")
    print(f"Images checked : {total_files}")
    print(f"Average SSIM   : {avg_score:.4f}")
    print(f"Minimum SSIM   : {min_score:.4f}")
    print(f"Maximum SSIM   : {max_score:.4f}")
    
    # Generate and save chart
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=40, color='steelblue', edgecolor='white')
    plt.axvline(avg_score, color='green', linestyle='solid', linewidth=2, label=f'Mean ({avg_score:.3f})')
    plt.axvline(0.65, color='red', linestyle='dashed', label='Lower Limit (0.65)')
    plt.title(f"SSIM Score Distribution\n{step_name}")
    plt.xlabel("SSIM Score")
    plt.ylabel("Number of Images")
    plt.legend()
    
    chart_path = os.path.join(OUTPUT_DIR, output_chart_name)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close() # clear memory
    
    print(f"✅ Chart saved to: {chart_path}")

# ==========================================
# 3. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("Starting full dataset SSIM Audit...")
    
    # 1. Raw vs Processed (DullRazor + Shades of Gray)
    run_audit_step(
        step_name="Step 1: Original vs DullRazor+SoG",
        dir_a=DIR_RAW,
        dir_b=DIR_PROCESSED,
        output_chart_name="1_ssim_dullrazor.png",
        apply_crop_to_a=False
    )
    
    # 2. Processed vs CLAHE
    run_audit_step(
        step_name="Step 2: Processed vs CLAHE",
        dir_a=DIR_PROCESSED,
        dir_b=DIR_CLAHE,
        output_chart_name="2_ssim_clahe.png",
        apply_crop_to_a=False
    )
    
    # 3. CLAHE vs Cropped (Crop fix applied!)
    run_audit_step(
        step_name="Step 3: CLAHE vs Cropped",
        dir_a=DIR_CLAHE,
        dir_b=DIR_CROPPED,
        output_chart_name="3_ssim_cropped.png",
        apply_crop_to_a=True
    )
    
    # 4. Cropped vs Final (Resize)
    run_audit_step(
        step_name="Step 4: Cropped vs Final Resized",
        dir_a=DIR_CROPPED,
        dir_b=DIR_FINAL,
        output_chart_name="4_ssim_final_resize.png",
        apply_crop_to_a=False
    )
    
    # 5. The Grand Finale: Raw vs Final (Crop fix applied!)
    run_audit_step(
        step_name="Final Check: Raw Original vs Final Output",
        dir_a=DIR_RAW,
        dir_b=DIR_FINAL,
        output_chart_name="5_ssim_raw_vs_final.png",
        apply_crop_to_a=True
    )
    
    print("\n🎉 ALL AUDITS COMPLETE! Check your output folder for the 5 charts.")