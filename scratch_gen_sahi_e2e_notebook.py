import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 11 Final: SAHI + ResNet End-to-End Evaluation\n\nThis notebook evaluates the final End-to-End pipeline on the untouched test set using SAHI (Tiled Inference) for the Stage-1 YOLO detector, followed by the Stage-2 ResNet-50 classifier."),
    
    nbf.v4.new_code_cell("""\
# Mount Google Drive
from google.colab import drive
import os
import zipfile

drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'
os.chdir(PROJECT_ROOT)
print(f"Current working directory: {os.getcwd()}")
"""),

    nbf.v4.new_code_cell("""\
# Install dependencies
!pip install ultralytics sahi torchvision torch numpy pandas matplotlib seaborn scikit-learn
"""),

    nbf.v4.new_code_cell("""\
# Extract the true 3-class test set to Colab's high-speed local storage
zip_path = 'test_3class.zip'
local_extract_dir = '/content/test_3class'

if os.path.exists(zip_path):
    print(f"Extracting {zip_path} to {local_extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(local_extract_dir)
    print("Extracted successfully.")
else:
    print("test_3class.zip not found! Please upload it.")

# Fix Windows backslash issue if extracted on Linux
import shutil
for root, dirs, files in os.walk(local_extract_dir):
    for file in files:
        if '\\\\' in file:
            new_path = file.replace('\\\\', '/')
            full_new_path = os.path.join(local_extract_dir, new_path)
            os.makedirs(os.path.dirname(full_new_path), exist_ok=True)
            shutil.move(os.path.join(root, file), full_new_path)
print("Zip extraction complete.")
"""),

    nbf.v4.new_markdown_cell("## PART A: SAHI Detector Test Evaluation (Single-Class)"),
    
    nbf.v4.new_code_cell("""\
import glob
import cv2
import time
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

yolo_weights = 'runs/detect/runs/detect/EXP-10-YOLOv12m-640-200/weights/best.pt'
detector = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=yolo_weights,
    confidence_threshold=0.15,
    device="cuda:0"
)

# Helper for IoU
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

test_img_dir = 'dataset_yolo_single_class/test/images'
test_lbl_dir = 'dataset_yolo_single_class/test/labels'
images = glob.glob(os.path.join(test_img_dir, '*.jpg')) + glob.glob(os.path.join(test_img_dir, '*.png'))

total_tp = 0
total_fp = 0
total_fn = 0
total_time = 0
total_gt_boxes = 0
total_pred_boxes = 0

print("Running SAHI Detector Evaluation on Test Set...")
for img_path in images:
    img_name = os.path.basename(img_path)
    lbl_path = os.path.join(test_lbl_dir, os.path.splitext(img_name)[0] + '.txt')
    
    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]
    
    gt_boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = map(float, parts[1:5])
                    gt_boxes.append(yolo_to_xyxy(cx, cy, bw, bh, w, h))
    
    t0 = time.time()
    result = get_sliced_prediction(
        img_path, detector,
        slice_height=320, slice_width=320,
        overlap_height_ratio=0.20, overlap_width_ratio=0.20,
        postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=0.50,
        verbose=False
    )
    total_time += (time.time() - t0)
    
    pred_boxes = [[obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy] for obj in result.object_prediction_list]
    
    total_gt_boxes += len(gt_boxes)
    total_pred_boxes += len(pred_boxes)
    
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

prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
avg_time = total_time / len(images)

print("\\n=== PART A: SAHI DETECTOR TEST SET RESULTS ===")
print(f"Total GT Boxes: {total_gt_boxes}")
print(f"Total Pred Boxes: {total_pred_boxes}")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall: {rec*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")
print(f"Avg Inference Time: {avg_time:.3f}s per image")
"""),

    nbf.v4.new_markdown_cell("## PART B: SAHI + ResNet End-to-End Test"),

    nbf.v4.new_code_cell("""\
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet_weights = 'runs/classify/EXP-11-RunB2/best.pt'

print("Loading ResNet-50...")
classifier = models.resnet50(weights=None)
num_ftrs = classifier.fc.in_features
classifier.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 3))
classifier.load_state_dict(torch.load(resnet_weights, map_location=device))
classifier.to(device)
classifier.eval()

class_names = ['Inclusion-Particle', 'Porosity', 'Tear-Delamination']
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_e2e_gt(label_path, img_w, img_h):
    gt = []
    if not os.path.exists(label_path): return gt
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                c_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                gt.append({'bbox': yolo_to_xyxy(cx, cy, w, h, img_w, img_h), 'class': class_names[c_id]})
    return gt

e2e_img_dir = '/content/test_3class/test/images'
e2e_lbl_dir = '/content/test_3class/test/labels'
e2e_images = glob.glob(os.path.join(e2e_img_dir, '*.jpg')) + glob.glob(os.path.join(e2e_img_dir, '*.png'))

e2e_gt_total = 0
e2e_pred_total = 0
e2e_tp = 0
padding_factor = 0.10

all_true_classes = []
all_pred_classes = []

print("Running SAHI + ResNet End-to-End on Test Set...")
for img_path in e2e_images:
    img_name = os.path.basename(img_path)
    lbl_path = os.path.join(e2e_lbl_dir, os.path.splitext(img_name)[0] + '.txt')
    
    pil_img = Image.open(img_path).convert("RGB")
    img_w, img_h = pil_img.size
    
    gt_boxes = load_e2e_gt(lbl_path, img_w, img_h)
    e2e_gt_total += len(gt_boxes)
    
    # 1. SAHI Detection
    result = get_sliced_prediction(
        img_path, detector,
        slice_height=320, slice_width=320,
        overlap_height_ratio=0.20, overlap_width_ratio=0.20,
        postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=0.50,
        verbose=False
    )
    
    preds = []
    for obj in result.object_prediction_list:
        x1, y1, x2, y2 = obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = w * padding_factor, h * padding_factor
        
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(img_w, x2 + pad_x)
        crop_y2 = min(img_h, y2 + pad_y)
        
        # 2. ResNet Classification
        crop_img = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        input_tensor = preprocess(crop_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = classifier(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, class_idx = torch.max(probs, 1)
        
        preds.append({'bbox': [x1, y1, x2, y2], 'pred_class': class_names[class_idx.item()]})
        
    e2e_pred_total += len(preds)
    
    # 3. E2E Matching
    matched_gt = set()
    for p in preds:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt: continue
            iou = bbox_iou(p['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                
        if best_iou >= 0.50:
            matched_gt.add(best_gt_idx)
            gt_class = gt_boxes[best_gt_idx]['class']
            pred_class = p['pred_class']
            
            all_true_classes.append(gt_class)
            all_pred_classes.append(pred_class)
            
            if gt_class == pred_class:
                e2e_tp += 1

e2e_fp = e2e_pred_total - e2e_tp
e2e_fn = e2e_gt_total - e2e_tp

e2e_prec = e2e_tp / e2e_pred_total if e2e_pred_total > 0 else 0
e2e_rec = e2e_tp / e2e_gt_total if e2e_gt_total > 0 else 0
e2e_f1 = 2 * e2e_prec * e2e_rec / (e2e_prec + e2e_rec) if (e2e_prec + e2e_rec) > 0 else 0

print("\\n=== PART B: SAHI + RESNET END-TO-END RESULTS ===")
print(f"Total GT Defects: {e2e_gt_total}")
print(f"Total Pred Defects: {e2e_pred_total}")
print(f"E2E TP: {e2e_tp} | E2E FP: {e2e_fp} | E2E FN: {e2e_fn}")
print(f"E2E Precision: {e2e_prec*100:.2f}%")
print(f"E2E Recall: {e2e_rec*100:.2f}%")
print(f"E2E F1-Score: {e2e_f1*100:.2f}%")

print("\\nClassification Report (on YOLO-matched boxes only):")
print(classification_report(all_true_classes, all_pred_classes, zero_division=0))
""")
]

nb['cells'] = cells
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '14_final_sahi_e2e_evaluation.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated at {notebook_path}")
