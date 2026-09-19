import os
import glob
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def main():
    original_dataset_path = r"c:\Users\syed mohammad anas\Desktop\sir_khurram_project\dataset_syedanas"
    output_base_path = r"c:\Users\syed mohammad anas\Desktop\sir_khurram_project\sem_defect_project\datasets\classifier_crops"
    
    classes = {
        0: 'Inclusion-Particle',
        1: 'Porosity',
        2: 'Tear-Delamination'
    }
    
    splits = ['train', 'valid', 'test']
    
    # Create directories
    for split in splits:
        for cls_name in classes.values():
            os.makedirs(os.path.join(output_base_path, split, cls_name), exist_ok=True)
            
    manifest_data = []
    
    for split in splits:
        print(f"Processing split: {split}")
        img_dir = os.path.join(original_dataset_path, split, "images")
        lbl_dir = os.path.join(original_dataset_path, split, "labels")
        
        if not os.path.exists(img_dir):
            continue
            
        img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        
        for img_path in tqdm(img_paths, desc=split):
            img_filename = os.path.basename(img_path)
            
            # Extract base image name (Roboflow format: base_name_jpg.rf.hash.jpg)
            base_image = img_filename.split('_jpg.rf.')[0]
            if base_image == img_filename:
                base_image = img_filename.split('.rf.')[0]
                
            lbl_path = os.path.join(lbl_dir, img_filename.replace('.jpg', '.txt'))
            
            if not os.path.exists(lbl_path):
                continue
                
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                cls_id = int(parts[0])
                if cls_id not in classes:
                    continue
                    
                x_center, y_center, w, h = map(float, parts[1:])
                
                # Convert to pixel coordinates
                w_px = w * img_w
                h_px = h * img_h
                x_c_px = x_center * img_w
                y_c_px = y_center * img_h
                
                # Calculate 10% padding (5% each side)
                pad_w = 0.05 * w_px
                pad_h = 0.05 * h_px
                
                x1 = int(x_c_px - (w_px / 2) - pad_w)
                y1 = int(y_c_px - (h_px / 2) - pad_h)
                x2 = int(x_c_px + (w_px / 2) + pad_w)
                y2 = int(y_c_px + (h_px / 2) + pad_h)
                
                # Clip to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                
                # Check for valid crop
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop = img[y1:y2, x1:x2]
                
                cls_name = classes[cls_id]
                crop_filename = f"{base_image}_crop{idx}.jpg"
                crop_path = os.path.join(output_base_path, split, cls_name, crop_filename)
                
                # To handle potential duplicate names if offline augmentation generated same base + same crop index
                # We will append a counter if file exists
                counter = 1
                while os.path.exists(crop_path):
                    crop_filename = f"{base_image}_crop{idx}_v{counter}.jpg"
                    crop_path = os.path.join(output_base_path, split, cls_name, crop_filename)
                    counter += 1
                
                cv2.imwrite(crop_path, crop)
                
                manifest_data.append({
                    'crop_id': crop_filename,
                    'source_image': img_filename,
                    'base_image': base_image,
                    'split': split,
                    'original_class': cls_name,
                    'original_class_id': cls_id,
                    'original_x_center': x_center,
                    'original_y_center': y_center,
                    'original_width': w,
                    'original_height': h,
                    'padding_ratio': 0.10,
                    'crop_path': crop_path
                })
                
    # Save manifest
    manifest_df = pd.DataFrame(manifest_data)
    manifest_path = os.path.join(output_base_path, 'classifier_crop_manifest.csv')
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"\\nCrop generation complete! Manifest saved to {manifest_path}")
    print(f"Total crops generated: {len(manifest_df)}")

if __name__ == "__main__":
    main()
