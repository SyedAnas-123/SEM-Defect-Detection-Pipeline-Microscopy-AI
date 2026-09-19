import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 10: False-Negative Error Analysis (Test Set)\n\nThis notebook evaluates the current best detector (`EXP-10 YOLO11m`) on the untouched test set to identify exactly WHICH ground-truth defects are missed, their sizes, and any nearby false positive predictions."),
    
    nbf.v4.new_code_cell("""\
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""),

    nbf.v4.new_code_cell("""\
# Install dependencies
!pip install ultralytics pandas matplotlib
"""),

    nbf.v4.new_code_cell("""\
import os
import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'
os.chdir(PROJECT_ROOT)
print(f"Current working directory: {os.getcwd()}")
"""),

    nbf.v4.new_code_cell("""\
# Define paths
weights_path = 'runs/detect/runs/detect/EXP-10-YOLOv12m-640-200/weights/best.pt'
img_dir = 'dataset_yolo_single_class/test/images'
lbl_dir = 'dataset_yolo_single_class/test/labels'
output_dir = 'runs/detect/FN_Analysis'

os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created at: {output_dir}")

# Load the best YOLO11m model
model = YOLO(weights_path)
"""),

    nbf.v4.new_markdown_cell("## Inference & IoU Matching Logic"),

    nbf.v4.new_code_cell("""\
def bbox_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Convert YOLO format (cx, cy, w, h) to (x1, y1, x2, y2)
def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]
"""),

    nbf.v4.new_code_cell("""\
fn_records = []
test_images = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))

for img_path in test_images:
    img_name = os.path.basename(img_path)
    lbl_name = os.path.splitext(img_name)[0] + '.txt'
    lbl_path = os.path.join(lbl_dir, lbl_name)
    
    # Read Image
    img = cv2.imread(img_path)
    if img is None: continue
    img_h, img_w = img.shape[:2]
    
    # Read GT
    gt_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cx, cy, w, h = map(float, parts[1:5])
                    gt_boxes.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))
    
    # Run YOLO Inference (locked protocol: conf=0.15, iou=0.50)
    results = model(img_path, conf=0.15, iou=0.50, verbose=False)
    pred_boxes = []
    pred_confs = []
    if len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            pred_boxes.append([x1, y1, x2, y2])
            pred_confs.append(conf)
            
    # Match GT to Preds
    for gt_idx, gt_box in enumerate(gt_boxes):
        best_iou = 0.0
        best_pred_idx = -1
        
        for p_idx, p_box in enumerate(pred_boxes):
            iou = bbox_iou(gt_box, p_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = p_idx
                
        # If best IoU < 0.5, it's a False Negative
        if best_iou < 0.50:
            gt_w = gt_box[2] - gt_box[0]
            gt_h = gt_box[3] - gt_box[1]
            gt_area = gt_w * gt_h
            img_area = img_w * img_h
            rel_area_pct = (gt_area / img_area) * 100.0
            
            # Size categorization
            if rel_area_pct < 0.5:
                size_cat = 'Small'
            elif rel_area_pct < 2.0:
                size_cat = 'Medium'
            else:
                size_cat = 'Large'
                
            fn_record = {
                'image_id': img_name,
                'gt_box': [round(x, 2) for x in gt_box],
                'width': round(gt_w, 2),
                'height': round(gt_h, 2),
                'area': round(gt_area, 2),
                'rel_area_pct': round(rel_area_pct, 4),
                'size_category': size_cat,
                'nearby_pred_iou': round(best_iou, 4) if best_pred_idx != -1 else 0.0,
                'nearby_pred_conf': round(pred_confs[best_pred_idx], 4) if best_pred_idx != -1 else 0.0,
                'nearby_pred_box': [round(x, 2) for x in pred_boxes[best_pred_idx]] if best_pred_idx != -1 else None
            }
            fn_records.append(fn_record)
            
            # Plot and save diagnostic image
            plt.figure(figsize=(8,8))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            ax = plt.gca()
            
            # Draw GT (Red)
            rect_gt = plt.Rectangle((gt_box[0], gt_box[1]), gt_w, gt_h, fill=False, edgecolor='red', linewidth=3)
            ax.add_patch(rect_gt)
            plt.text(gt_box[0], gt_box[1]-5, 'Missed GT', color='red', weight='bold')
            
            # Draw nearest pred if exists (Blue)
            if best_pred_idx != -1 and best_iou > 0:
                p_box = pred_boxes[best_pred_idx]
                p_w = p_box[2] - p_box[0]
                p_h = p_box[3] - p_box[1]
                rect_p = plt.Rectangle((p_box[0], p_box[1]), p_w, p_h, fill=False, edgecolor='blue', linewidth=2, linestyle='--')
                ax.add_patch(rect_p)
                plt.text(p_box[0], p_box[1]-5, f'Pred (IoU {best_iou:.2f})', color='blue', weight='bold')
                
            plt.axis('off')
            out_img_name = f"FN_{img_name}_gt{gt_idx}.jpg"
            plt.savefig(os.path.join(output_dir, out_img_name), bbox_inches='tight', pad_inches=0)
            plt.close()

df_fn = pd.DataFrame(fn_records)
csv_path = os.path.join(output_dir, 'fn_analysis.csv')
df_fn.to_csv(csv_path, index=False)
print(f"Found {len(df_fn)} False Negatives. Data saved to {csv_path}")
"""),

    nbf.v4.new_code_cell("""\
# Zip the results for easy download
import shutil
shutil.make_archive('FN_Analysis_Results', 'zip', output_dir)
print("Saved FN_Analysis_Results.zip!")
""")
]

nb['cells'] = cells
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '12_false_negative_error_analysis.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated at {notebook_path}")
