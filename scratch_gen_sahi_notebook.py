import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 11: SAHI / Tiled Inference Validation Sweep\n\nThis notebook evaluates different slicing configurations on the Validation set to see if Tiled Inference improves Recall for small defects."),
    
    nbf.v4.new_code_cell("""\
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""),

    nbf.v4.new_code_cell("""\
# Install dependencies
!pip install ultralytics sahi pandas
"""),

    nbf.v4.new_code_cell("""\
import os
import glob
import cv2
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'
os.chdir(PROJECT_ROOT)
print(f"Current working directory: {os.getcwd()}")
"""),

    nbf.v4.new_code_cell("""\
# Define paths
weights_path = 'runs/detect/runs/detect/EXP-10-YOLOv12m-640-200/weights/best.pt'
img_dir = 'dataset_yolo_single_class/valid/images'
lbl_dir = 'dataset_yolo_single_class/valid/labels'

# Load SAHI Model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=weights_path,
    confidence_threshold=0.15,
    device="cuda:0"
)
"""),

    nbf.v4.new_code_cell("""\
# Helper for IoU calculation
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

    return intersection_area / float(box1_area + box2_area - intersection_area)

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]
"""),

    nbf.v4.new_code_cell("""\
# Validation Evaluation Function
def evaluate_sahi(slice_size, overlap_ratio, img_dir, lbl_dir, nms_iou_thresh=0.50):
    test_images = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_time = 0
    total_preds = 0
    
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(lbl_dir, lbl_name)
        
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
                        
        # Run SAHI
        t0 = time.time()
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=nms_iou_thresh,
            verbose=False
        )
        t1 = time.time()
        total_time += (t1 - t0)
        
        # Parse Predictions
        pred_boxes = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox
            pred_boxes.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
            
        total_preds += len(pred_boxes)
        
        # Matching (TP, FP, FN)
        matched_gt = set()
        matched_pred = set()
        
        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt: continue
                iou = bbox_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    
            if best_iou >= 0.50:
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
                total_tp += 1
            else:
                total_fp += 1
                
        total_fn += (len(gt_boxes) - len(matched_gt))

    # Metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_time = total_time / len(test_images)
    
    return {
        'Config': f"SAHI {slice_size}x{slice_size} (ov={overlap_ratio})",
        'Precision': f"{precision*100:.2f}%",
        'Recall': f"{recall*100:.2f}%",
        'F1-Score': f"{f1*100:.2f}%",
        'Total Preds': total_preds,
        'Avg Time/Img': f"{avg_time:.3f}s"
    }
"""),

    nbf.v4.new_code_cell("""\
# Run Sweep
configs = [
    (256, 0.20), (256, 0.30), (256, 0.40),
    (320, 0.20), (320, 0.30), (320, 0.40)
]

results = []
print("Starting SAHI Validation Sweep... This will take a few minutes.")
for slice_size, overlap in configs:
    print(f"Evaluating {slice_size}x{slice_size} with overlap {overlap}...")
    res = evaluate_sahi(slice_size, overlap, img_dir, lbl_dir)
    results.append(res)
    print(res)

df_results = pd.DataFrame(results)
print("\\n--- FINAL SWEEP RESULTS ---")
print(df_results.to_string(index=False))

df_results.to_csv("sahi_validation_sweep_results.csv", index=False)
""")
]

nb['cells'] = cells
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '13_sahi_validation_sweep.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated at {notebook_path}")
