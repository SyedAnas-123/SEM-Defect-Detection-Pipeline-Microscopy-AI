import os
import cv2
import numpy as np
import random
import glob

# Paths
DATASET_DIR = r"c:\Users\syed mohammad anas\Desktop\sir_khurram_project\sem_defect_project\datasets\dataset_yolo_single_class"
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_DIR, "train", "labels")

OUT_DIR = r"C:\Users\syed mohammad anas\.gemini\antigravity-ide\brain\45bf5d65-37fd-48bc-bf12-4ad0088701a9\cp_samples"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Collect Tiny Defects
def get_tiny_defects(limit_area=0.005):
    tiny_defects = []
    label_files = glob.glob(os.path.join(TRAIN_LABELS, "*.txt"))
    for lbl_file in label_files:
        with open(lbl_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                c, x, y, w, h = map(float, parts[:5])
                area = w * h
                if area < limit_area:
                    img_file = os.path.join(TRAIN_IMAGES, os.path.basename(lbl_file).replace('.txt', '.png'))
                    if not os.path.exists(img_file):
                        img_file = img_file.replace('.png', '.jpg')
                    if os.path.exists(img_file):
                        tiny_defects.append((img_file, x, y, w, h))
    return tiny_defects

def get_gaussian_alpha_mask(w, h):
    # Create an elliptical mask that fades to the edges
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (w // 2, h // 2)
    
    # Draw solid ellipse slightly smaller than the box to keep core intact
    cv2.ellipse(mask, center, (int(w*0.4), int(h*0.4)), 0, 0, 360, 1.0, -1)
    
    # Blur it to create smooth alpha transition
    blur_kernel = (max(3, w//3 * 2 + 1), max(3, h//3 * 2 + 1))
    mask = cv2.GaussianBlur(mask, blur_kernel, 0)
    
    # Expand dims for broadcasting
    return np.stack([mask]*3, axis=-1)

def iou(box1, box2):
    # box: (x_center, y_center, w, h)
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    
    return inter_area / (area1 + area2 - inter_area)

print("Scanning for tiny defects...")
tiny_defects = get_tiny_defects(0.005)
print(f"Found {len(tiny_defects)} tiny defects.")

if not tiny_defects:
    exit("No tiny defects found!")

image_files = glob.glob(os.path.join(TRAIN_IMAGES, "*.*"))

# Generate 20 samples
samples_generated = 0
while samples_generated < 20:
    target_img_path = random.choice(image_files)
    target_img = cv2.imread(target_img_path)
    if target_img is None: continue
    
    img_h, img_w = target_img.shape[:2]
    
    # Load existing boxes
    lbl_path = os.path.join(TRAIN_LABELS, os.path.basename(target_img_path).rsplit('.', 1)[0] + '.txt')
    existing_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    existing_boxes.append(list(map(float, parts[1:5])))
    
    # Choose a random tiny defect
    src_img_path, sx, sy, sw, sh = random.choice(tiny_defects)
    src_img = cv2.imread(src_img_path)
    if src_img is None: continue
    
    # Extract crop
    x_center, y_center = int(sx * img_w), int(sy * img_h)
    box_w, box_h = int(sw * img_w), int(sh * img_h)
    
    x1, y1 = max(0, x_center - box_w//2), max(0, y_center - box_h//2)
    x2, y2 = min(img_w, x_center + box_w//2), min(img_h, y_center + box_h//2)
    
    crop = src_img[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    
    if crop_h < 5 or crop_w < 5: continue
    
    # Find valid target location
    valid = False
    for _ in range(50):
        tx = random.randint(crop_w//2, img_w - crop_w//2)
        ty = random.randint(crop_h//2, img_h - crop_h//2)
        
        tx_norm, ty_norm = tx / img_w, ty / img_h
        new_box = [tx_norm, ty_norm, sw, sh]
        
        has_overlap = any(iou(new_box, eb) > 0.05 for eb in existing_boxes)
        if not has_overlap:
            valid = True
            break
            
    if not valid: continue
    
    # Apply Gaussian Alpha Blending
    tx1, ty1 = tx - crop_w//2, ty - crop_h//2
    tx2, ty2 = tx1 + crop_w, ty1 + crop_h
    
    bg_crop = target_img[ty1:ty2, tx1:tx2]
    if bg_crop.shape != crop.shape: continue
    
    alpha = get_gaussian_alpha_mask(crop_w, crop_h)
    blended = (crop * alpha + bg_crop * (1 - alpha)).astype(np.uint8)
    
    # Visualize: Create a side-by-side comparison
    # Draw green box on the blended image
    result_img = target_img.copy()
    result_img[ty1:ty2, tx1:tx2] = blended
    cv2.rectangle(result_img, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
    
    # Create zoom panel
    zoom_size = 150
    zx1, zy1 = max(0, tx - zoom_size//2), max(0, ty - zoom_size//2)
    zx2, zy2 = min(img_w, tx + zoom_size//2), min(img_h, ty + zoom_size//2)
    
    zoom_bg = target_img[zy1:zy2, zx1:zx2].copy()
    zoom_pasted = result_img[zy1:zy2, zx1:zx2].copy()
    
    # Pad if near edge to keep 150x150
    zoom_bg = cv2.resize(zoom_bg, (zoom_size, zoom_size))
    zoom_pasted = cv2.resize(zoom_pasted, (zoom_size, zoom_size))
    
    # Annotate texts
    cv2.putText(zoom_bg, "Original Empty", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(zoom_pasted, "Pasted Defect", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    panel = np.hstack((zoom_bg, zoom_pasted))
    
    out_file = os.path.join(OUT_DIR, f"sample_{samples_generated+1:02d}.jpg")
    cv2.imwrite(out_file, panel)
    
    samples_generated += 1

print(f"Successfully generated 20 validation samples in {OUT_DIR}")
