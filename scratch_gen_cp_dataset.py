import os
import cv2
import numpy as np
import random
import glob
import shutil

# Paths
ROOT_DIR = r"c:\Users\syed mohammad anas\Desktop\sir_khurram_project\sem_defect_project"
SRC_DATASET = os.path.join(ROOT_DIR, "datasets", "dataset_yolo_single_class")
DST_DATASET = os.path.join(ROOT_DIR, "datasets", "dataset_yolo_single_class_cp")

if os.path.exists(DST_DATASET):
    shutil.rmtree(DST_DATASET)
os.makedirs(DST_DATASET, exist_ok=True)

# 1. Copy Valid and Test sets verbatim
for split in ["valid", "test"]:
    src_split = os.path.join(SRC_DATASET, split)
    if os.path.exists(src_split):
        shutil.copytree(src_split, os.path.join(DST_DATASET, split))
        print(f"Copied {split} set.")

# 2. Setup Train sets
os.makedirs(os.path.join(DST_DATASET, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(DST_DATASET, "train", "labels"), exist_ok=True)

src_train_imgs = glob.glob(os.path.join(SRC_DATASET, "train", "images", "*.*"))
print(f"Found {len(src_train_imgs)} training images.")

# Copy all original train images and labels
for img_path in src_train_imgs:
    shutil.copy(img_path, os.path.join(DST_DATASET, "train", "images"))
    lbl_name = os.path.basename(img_path).rsplit('.', 1)[0] + '.txt'
    src_lbl = os.path.join(SRC_DATASET, "train", "labels", lbl_name)
    if os.path.exists(src_lbl):
        shutil.copy(src_lbl, os.path.join(DST_DATASET, "train", "labels"))

# 3. Collect Tiny Defects
def get_tiny_defects(limit_area=0.005):
    tiny_defects = []
    label_files = glob.glob(os.path.join(SRC_DATASET, "train", "labels", "*.txt"))
    for lbl_file in label_files:
        with open(lbl_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                c, x, y, w, h = map(float, parts[:5])
                area = w * h
                if area < limit_area:
                    img_file = os.path.join(SRC_DATASET, "train", "images", os.path.basename(lbl_file).replace('.txt', '.png'))
                    if not os.path.exists(img_file):
                        img_file = img_file.replace('.png', '.jpg')
                    if os.path.exists(img_file):
                        tiny_defects.append((img_file, x, y, w, h))
    return tiny_defects

def get_gaussian_alpha_mask(w, h):
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    cv2.ellipse(mask, center, (int(w*0.4), int(h*0.4)), 0, 0, 360, 1.0, -1)
    blur_kernel = (max(3, w//3 * 2 + 1), max(3, h//3 * 2 + 1))
    mask = cv2.GaussianBlur(mask, blur_kernel, 0)
    return np.stack([mask]*3, axis=-1)

def iou(box1, box2):
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    inter_x1, inter_y1 = max(x1_1, x1_2), max(y1_1, y1_2)
    inter_x2, inter_y2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    return inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0

tiny_defects = get_tiny_defects(0.005)
print(f"Collected {len(tiny_defects)} tiny defect templates.")

# 4. Generate Augmented Images
augmented_count = 0
total_pasted = 0

for img_path in src_train_imgs:
    if random.random() > 0.5: continue # 50% chance
    
    target_img = cv2.imread(img_path)
    if target_img is None: continue
    
    img_h, img_w = target_img.shape[:2]
    
    lbl_path = os.path.join(SRC_DATASET, "train", "labels", os.path.basename(img_path).rsplit('.', 1)[0] + '.txt')
    existing_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    existing_boxes.append(list(map(float, parts[1:5])))
                    
    original_boxes = list(existing_boxes)
    pasted_this_img = 0
    num_to_paste = random.randint(1, 3)
    
    blended_img = target_img.copy()
    
    for _ in range(num_to_paste):
        src_img_path, sx, sy, sw, sh = random.choice(tiny_defects)
        src_img = cv2.imread(src_img_path)
        if src_img is None: continue
        
        x_center, y_center = int(sx * img_w), int(sy * img_h)
        box_w, box_h = int(sw * img_w), int(sh * img_h)
        
        x1, y1 = max(0, x_center - box_w//2), max(0, y_center - box_h//2)
        x2, y2 = min(img_w, x_center + box_w//2), min(img_h, y_center + box_h//2)
        
        crop = src_img[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 5 or crop_w < 5: continue
        
        valid = False
        for _ in range(50):
            tx = random.randint(crop_w//2, img_w - crop_w//2)
            ty = random.randint(crop_h//2, img_h - crop_h//2)
            new_box = [tx / img_w, ty / img_h, sw, sh]
            if not any(iou(new_box, eb) > 0.05 for eb in existing_boxes):
                valid = True
                break
                
        if not valid: continue
        
        tx1, ty1 = tx - crop_w//2, ty - crop_h//2
        tx2, ty2 = tx1 + crop_w, ty1 + crop_h
        
        bg_crop = blended_img[ty1:ty2, tx1:tx2]
        if bg_crop.shape != crop.shape: continue
        
        alpha = get_gaussian_alpha_mask(crop_w, crop_h)
        blended = (crop * alpha + bg_crop * (1 - alpha)).astype(np.uint8)
        blended_img[ty1:ty2, tx1:tx2] = blended
        
        existing_boxes.append(new_box)
        pasted_this_img += 1
        total_pasted += 1
        
    if pasted_this_img > 0:
        base_name = os.path.basename(img_path).rsplit('.', 1)[0]
        ext = os.path.basename(img_path).rsplit('.', 1)[1]
        
        new_img_name = f"{base_name}_cp.{ext}"
        new_lbl_name = f"{base_name}_cp.txt"
        
        cv2.imwrite(os.path.join(DST_DATASET, "train", "images", new_img_name), blended_img)
        
        with open(os.path.join(DST_DATASET, "train", "labels", new_lbl_name), 'w') as f:
            for box in existing_boxes:
                f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                
        augmented_count += 1

print(f"Generated {augmented_count} augmented images with {total_pasted} pasted defects.")

# 5. Create data.yaml
yaml_content = f'''path: {DST_DATASET}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['Defect']
'''
with open(os.path.join(DST_DATASET, "data.yaml"), "w") as f:
    f.write(yaml_content)

# 6. Audit
def audit_dataset():
    errors = []
    stats = {'train': {'imgs': 0, 'lbls': 0, 'boxes': 0},
             'valid': {'imgs': 0, 'lbls': 0, 'boxes': 0},
             'test': {'imgs': 0, 'lbls': 0, 'boxes': 0}}
             
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(DST_DATASET, split, "images")
        lbl_dir = os.path.join(DST_DATASET, split, "labels")
        if not os.path.exists(img_dir): continue
        
        imgs = glob.glob(os.path.join(img_dir, "*.*"))
        lbls = glob.glob(os.path.join(lbl_dir, "*.txt"))
        stats[split]['imgs'] = len(imgs)
        stats[split]['lbls'] = len(lbls)
        
        for lbl_path in lbls:
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                stats[split]['boxes'] += len(lines)
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        errors.append(f"Malformed line in {lbl_path}")
                        continue
                    c, x, y, w, h = map(float, parts)
                    if int(c) != 0:
                        errors.append(f"Invalid class {c} in {lbl_path}")
                    if x <= 0 or x >= 1 or y <= 0 or y >= 1 or w <= 0 or w > 1 or h <= 0 or h > 1:
                        errors.append(f"OOB box in {lbl_path}")
                        
    return stats, errors

stats, errors = audit_dataset()
print("\\n--- SANITY AUDIT RESULTS ---")
print(f"Train: {stats['train']['imgs']} images, {stats['train']['lbls']} labels, {stats['train']['boxes']} total boxes")
print(f"Valid: {stats['valid']['imgs']} images, {stats['valid']['lbls']} labels, {stats['valid']['boxes']} total boxes")
print(f"Test:  {stats['test']['imgs']} images, {stats['test']['lbls']} labels, {stats['test']['boxes']} total boxes")
print(f"Errors found: {len(errors)}")
if errors:
    print(errors[:5])

with open("scratch_cp_audit_results.txt", "w") as f:
    f.write(f"Augmented Images Created: {augmented_count}\\n")
    f.write(f"Tiny Defects Pasted: {total_pasted}\\n")
    f.write(f"Train Images: {stats['train']['imgs']}\\n")
    f.write(f"Train Boxes: {stats['train']['boxes']}\\n")
    f.write(f"Valid Images: {stats['valid']['imgs']}\\n")
    f.write(f"Test Images: {stats['test']['imgs']}\\n")
    f.write(f"Errors: {len(errors)}\\n")
