import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image

def create_contact_sheet(image_paths, output_path, title, grid_size=(4, 4), img_size=(100, 100)):
    cols, rows = grid_size
    n_images = min(len(image_paths), cols * rows)
    
    if n_images == 0:
        return
        
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    fig.suptitle(title, fontsize=16)
    
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for ax in axes:
        ax.axis('off')
        
    for i, img_path in enumerate(image_paths[:n_images]):
        try:
            img = Image.open(img_path)
            img = img.resize(img_size)
            axes[i].imshow(img)
            # Add short name
            name = os.path.basename(img_path)[:15] + '...' if len(os.path.basename(img_path)) > 15 else os.path.basename(img_path)
            axes[i].set_title(name, fontsize=8)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    base_dir = r"c:\Users\syed mohammad anas\Desktop\sir_khurram_project\sem_defect_project\datasets\classifier_crops"
    manifest_path = os.path.join(base_dir, "classifier_crop_manifest.csv")
    
    df = pd.read_csv(manifest_path)
    
    print("==================================================")
    print("1. OVERALL DISTRIBUTION")
    print("==================================================")
    
    print("\nCrops per Split:")
    print(df['split'].value_counts().to_string())
    
    print("\nCrops per Class:")
    print(df['original_class'].value_counts().to_string())
    
    print("\nSplit x Class Crosstab:")
    print(pd.crosstab(df['split'], df['original_class']))
    
    print("\n==================================================")
    print("2. BASE IMAGE / AUGMENTATION / LEAKAGE ANALYSIS")
    print("==================================================")
    
    # Calculate base image counts per split
    base_img_by_split = df.groupby('split')['base_image'].nunique()
    print("\nUnique Base Images (Physical source images) per split:")
    print(base_img_by_split.to_string())
    
    # Check for leakage (Base image appearing in multiple splits)
    train_bases = set(df[df['split'] == 'train']['base_image'])
    valid_bases = set(df[df['split'] == 'valid']['base_image'])
    test_bases = set(df[df['split'] == 'test']['base_image'])
    
    val_leakage = train_bases.intersection(valid_bases)
    test_leakage_train = train_bases.intersection(test_bases)
    test_leakage_val = valid_bases.intersection(test_bases)
    
    print(f"\nLeakage Train <-> Valid: {len(val_leakage)} overlapping base images")
    print(f"Leakage Train <-> Test: {len(test_leakage_train)} overlapping base images")
    print(f"Leakage Valid <-> Test: {len(test_leakage_val)} overlapping base images")
    
    if len(val_leakage) > 0 or len(test_leakage_train) > 0 or len(test_leakage_val) > 0:
        print("WARNING: Data Leakage detected!")
    else:
        print("SUCCESS: Zero Data Leakage detected based on base image names.")
        
    print("\n==================================================")
    print("3. CROP SIZE STATISTICS")
    print("==================================================")
    
    # Calculate pixel sizes (roughly based on 512x512 original assuming 512x512)
    # The dataframe has original_width and original_height (normalized 0-1)
    df['est_width_px'] = df['original_width'] * 512 * 1.1 # +10% padding
    df['est_height_px'] = df['original_height'] * 512 * 1.1
    df['est_area_px'] = df['est_width_px'] * df['est_height_px']
    
    print("\nAverage Estimated Crop Area (Pixels squared) per Class:")
    print(df.groupby('original_class')['est_area_px'].mean().apply(lambda x: f"{x:.2f} px^2"))
    
    print("\n==================================================")
    print("4. GENERATING CONTACT SHEETS")
    print("==================================================")
    
    os.makedirs(os.path.join(base_dir, 'contact_sheets'), exist_ok=True)
    
    for cls_name in df['original_class'].unique():
        cls_df = df[df['original_class'] == cls_name]
        
        # Sample images (ensure we prioritize original unaugmented paths if possible by grouping by base_image)
        # Just grab random 16 images for the contact sheet
        sample_df = cls_df.sample(min(16, len(cls_df)), random_state=42)
        
        out_path = os.path.join(base_dir, 'contact_sheets', f"{cls_name}_contact_sheet.png")
        create_contact_sheet(
            sample_df['crop_path'].tolist(), 
            out_path, 
            f"Sample Classifier Crops: {cls_name}",
            grid_size=(4, 4)
        )
        print(f"Generated {out_path}")
        
    print("\nAnalysis Complete!")

if __name__ == "__main__":
    main()
