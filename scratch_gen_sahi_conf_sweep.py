import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 12: SAHI Confidence Optimization Sweep\n\nThis notebook evaluates the `320x320 (0.20 overlap)` SAHI configuration across multiple confidence thresholds on the **Validation Set** to find the optimal trade-off between Recall and Precision."),
    
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

# Helper functions
def bbox_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / float(box1_area + box2_area - intersection)

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]
"""),

    nbf.v4.new_code_cell("""\
def evaluate_sahi_confidence(conf_thresh, img_dir, lbl_dir):
    # Load SAHI Model with dynamic confidence
    detector = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=weights_path,
        confidence_threshold=conf_thresh,
        device="cuda:0"
    )
    
    test_images = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_time = 0
    total_preds = 0
    
    for img_path in test_images:
        img_name = os.path.basename(img_path)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + '.txt')
        
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
            detector,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.20,
            overlap_width_ratio=0.20,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.50,
            verbose=False
        )
        total_time += (time.time() - t0)
        
        pred_boxes = [[obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy] for obj in result.object_prediction_list]
        total_preds += len(pred_boxes)
        
        # Matching
        matched_gt = set()
        for p_box in pred_boxes:
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
                total_tp += 1
            else:
                total_fp += 1
                
        total_fn += (len(gt_boxes) - len(matched_gt))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_time = total_time / len(test_images)
    
    return {
        'Conf': conf_thresh,
        'Precision': f"{precision*100:.2f}%",
        'Recall': f"{recall*100:.2f}%",
        'F1-Score': f"{f1*100:.2f}%",
        'Total Preds': total_preds,
        'FP': total_fp,
        'FN': total_fn,
        'Avg Time/Img': f"{avg_time:.3f}s"
    }
"""),

    nbf.v4.new_code_cell("""\
# Run Sweep
conf_sweep = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

results = []
print("Starting SAHI Confidence Sweep on Validation Set...")
for conf in conf_sweep:
    print(f"Evaluating conf={conf}...")
    res = evaluate_sahi_confidence(conf, img_dir, lbl_dir)
    results.append(res)
    print(res)

df_results = pd.DataFrame(results)
print("\\n--- FINAL SWEEP RESULTS ---")
print(df_results.to_string(index=False))

df_results.to_csv("sahi_conf_sweep_results.csv", index=False)
""")
]

nb['cells'] = cells
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '15_sahi_confidence_sweep.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated at {notebook_path}")
