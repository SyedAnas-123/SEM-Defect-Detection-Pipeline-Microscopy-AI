import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

def c(code): return nbf.v4.new_code_cell(code)
def m(text): return nbf.v4.new_markdown_cell(text)

cells = []

cells.append(m("# FINAL PIPELINE PREDICTION: Two-Stage SEM Defect Detection System\n\nThis comprehensive notebook dynamically evaluates our final, selected models and computes all metrics LIVE. It represents the complete End-to-End (E2E) inference pipeline on untouched SEM datasets.\n\n**Best Detector:** YOLO11m (EXP-10)\n**Best Classifier:** ResNet-50 (EXP-11 RunB2)"))

cells.append(c("""\
# ==========================================
# 0. Environment Setup & Data Preparation
# ==========================================
!pip install -q ultralytics torchvision pandas matplotlib seaborn scikit-learn tabulate

import os
import cv2
import glob
import torch
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from IPython.display import display
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
import torchvision.transforms as transforms
import torchvision.models as models
from ultralytics import YOLO

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set Project Paths
PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'
os.chdir(PROJECT_ROOT)
print(f"Current working directory: {os.getcwd()}")

# Fast extraction using system unzip
if not os.path.exists('/content/classifier_crops_eval'):
    if os.path.exists('classifier_crops.zip'):
        print("Copying and Extracting classifier_crops.zip (Ultra-Fast Local Unzip)...")
        os.system("cp classifier_crops.zip /content/classifier_crops.zip")
        # Extract to a dedicated folder to avoid path mess
        os.system("unzip -q -o /content/classifier_crops.zip -d /content/classifier_crops_eval")
        os.system("rm /content/classifier_crops.zip")
        # Find the actual test folder path dynamically
        for root, dirs, files in os.walk('/content/classifier_crops_eval'):
            if 'test' in dirs:
                # Store it as a global python variable in the notebook
                print(f"Found test dataset at: {os.path.join(root, 'test')}")
                break
    else:
        print("classifier_crops.zip not found!")

if not os.path.exists('/content/test_3class'):
    if os.path.exists('test_3class.zip'):
        print("Copying and Extracting test_3class.zip (Ultra-Fast Local Unzip)...")
        os.system("cp test_3class.zip /content/test_3class.zip")
        os.system("unzip -q /content/test_3class.zip -d /content/test_3class")
        os.system("rm /content/test_3class.zip")
        # Normalize paths for linux if zip was created on Windows
        for root, dirs, files in os.walk('/content/test_3class'):
            for f in files:
                if '\\\\' in f:
                    old_path = os.path.join(root, f)
                    new_rel_path = f.replace('\\\\', '/')
                    new_full_path = os.path.join(root, new_rel_path)
                    os.makedirs(os.path.dirname(new_full_path), exist_ok=True)
                    os.rename(old_path, new_full_path)
    else:
        print("test_3class.zip not found!")

print("Environment Ready.")
"""))

cells.append(m("==================================================\n# PART 1 — VERIFY BEST MODELS & LOAD CHECKPOINTS\n=================================================="))

cells.append(c("""\
# Verified Paths from Original Training
DETECTOR_PATH = "runs/detect/runs/detect/EXP-10-YOLOv12m-640-200/weights/best.pt"
CLASSIFIER_PATH = "runs/classify/EXP-11-RunB2/best.pt"

print("YOLO BEST CHECKPOINT:")
print(f"-> {os.path.abspath(DETECTOR_PATH)}")
print(f"Exists: {os.path.exists(DETECTOR_PATH)}")

print("\\nRESNET BEST CHECKPOINT:")
print(f"-> {os.path.abspath(CLASSIFIER_PATH)}")
print(f"Exists: {os.path.exists(CLASSIFIER_PATH)}")

# Load YOLO
try:
    detector = YOLO(DETECTOR_PATH)
    print("\\nYOLO11m (EXP-10) loaded successfully.")
except Exception as e:
    print(f"Could not load detector: {e}")

# Load ResNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    classifier = models.resnet50(weights=None)
    num_ftrs = classifier.fc.in_features
    # We recreate the exact sequential layer used during training
    classifier.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_ftrs, 3)
    )
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    print("ResNet-50 (EXP-11 RunB2) loaded successfully.")
except Exception as e:
    print(f"Could not load classifier: {e}")

class_names = ['Inclusion-Particle', 'Porosity', 'Tear-Delamination']
"""))


cells.append(m("==================================================\n# PART 3 & 4 — LIVE YOLO DETECTOR EVALUATION\n=================================================="))

cells.append(c("""\
print("Running LIVE YOLO Validation Evaluation...")
val_metrics = detector.val(
    data='dataset_yolo_single_class/data.yaml', 
    split='val', 
    conf=0.15, 
    iou=0.50, 
    verbose=False
)

p_val = val_metrics.box.mp
r_val = val_metrics.box.mr
f1_val = 2 * (p_val * r_val) / (p_val + r_val) if (p_val + r_val) > 0 else 0
map50_val = val_metrics.box.map50
map95_val = val_metrics.box.map

print("\\nRunning LIVE YOLO FINAL BLIND TEST EVALUATION...")
test_metrics = detector.val(
    data='dataset_yolo_single_class/data.yaml', 
    split='test', 
    conf=0.15, 
    iou=0.50, 
    verbose=False
)

p_test = test_metrics.box.mp
r_test = test_metrics.box.mr
f1_test = 2 * (p_test * r_test) / (p_test + r_test) if (p_test + r_test) > 0 else 0
map50_test = test_metrics.box.map50
map95_test = test_metrics.box.map

print("\\n------------------------------------------------")
print("LIVE DETECTOR RESULTS (Calculated just now)")
print("------------------------------------------------")
print(f"VALIDATION -> Precision: {p_val*100:.2f}% | Recall: {r_val*100:.2f}% | F1: {f1_val*100:.2f}% | mAP50: {map50_val*100:.2f}% | mAP50-95: {map95_val*100:.2f}%")
print(f"BLIND TEST -> Precision: {p_test*100:.2f}% | Recall: {r_test*100:.2f}% | F1: {f1_test*100:.2f}% | mAP50: {map50_test*100:.2f}% | mAP50-95: {map95_test*100:.2f}%")

# Plotting Detection Metrics
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 5, figsize=(22, 5))

metrics_names = ['Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP50-95']
v_scores = [p_val*100, r_val*100, f1_val*100, map50_val*100, map95_val*100] 
t_scores = [p_test*100, r_test*100, f1_test*100, map50_test*100, map95_test*100]

for i, metric in enumerate(metrics_names):
    ax = axes[i]
    ax.bar(['Validation', 'Test'], [v_scores[i], t_scores[i]], color=['#4C72B0', '#55A868'])
    ax.set_title(metric, fontweight='bold')
    ax.set_ylim(0, 100)
    for j, v in enumerate([v_scores[i], t_scores[i]]):
        ax.text(j, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')

plt.suptitle('LIVE YOLO11m Detector Performance', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()
"""))

cells.append(m("==================================================\n# PART 5 & 6 — LIVE RESNET CLASSIFIER EVALUATION\n=================================================="))

cells.append(c("""\
from torchvision import datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_classifier(data_dir, split_name):
    ds = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = classifier(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\\n--- {split_name} CLASSIFIER RESULTS ---")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Macro F1:    {macro_f1*100:.2f}%")
    print(f"Weighted F1: {weighted_f1*100:.2f}%")
    print("\\nPer-class Metrics:")
    for i, cls in enumerate(ds.classes):
        print(f"{cls:20s}: P={p[i]*100:.2f}%, R={r[i]*100:.2f}%, F1={f1[i]*100:.2f}% (Support: {support[i]})")
        
    return acc, macro_f1, weighted_f1, p, r, f1, cm, ds.classes

# Find actual test path
import glob
test_paths = glob.glob('/content/classifier_crops_eval/**/test', recursive=True)
actual_test_dir = test_paths[0] if test_paths else '/content/classifier_crops_eval/test'
actual_val_dir = os.path.join(os.path.dirname(actual_test_dir), 'valid')

if os.path.exists(actual_val_dir):
    print("Running LIVE ResNet Validation...")
    v_acc, v_mf1, v_wf1, v_p, v_r, v_f1, v_cm, cls_names = evaluate_classifier(actual_val_dir, 'VALIDATION')
else:
    print(f"Warning: Validation folder 'valid' not found at {actual_val_dir}. Skipping Validation Evaluation.")

print("\\nRunning LIVE FINAL CLASSIFIER TEST EVALUATION...")
t_acc, t_mf1, t_wf1, t_p, t_r, t_f1, t_cm, cls_names = evaluate_classifier(actual_test_dir, 'BLIND TEST')

# Plotting Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(t_cm, annot=True, fmt='d', cmap='Greens', xticklabels=cls_names, yticklabels=cls_names, ax=ax)
ax.set_title('Test Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
plt.show()

# Plotting Per-Class F1
x = np.arange(len(cls_names))
fig, ax = plt.subplots(figsize=(8, 5))
rects2 = ax.bar(x, t_f1*100, 0.4, label='Test', color='#55A868')
ax.set_ylabel('F1-Score (%)')
ax.set_title('Classifier Per-Class F1-Scores')
ax.set_xticks(x)
ax.set_xticklabels(cls_names)
ax.legend()
plt.ylim(0, 100)
plt.show()
"""))

cells.append(m("==================================================\n# PART 7 — LIVE COMPLETE E2E PIPELINE\n=================================================="))

cells.append(c("""\
def bbox_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

def run_live_e2e(test_img_dir, test_label_dir, conf_thresh=0.15, padding_factor=0.10, iou_thresh=0.50):
    images = glob.glob(os.path.join(test_img_dir, '*.jpg'))
    if not images:
        images = glob.glob(os.path.join(test_img_dir, '*.png'))
        
    print(f"Evaluating {len(images)} TEST images END-TO-END...")
    
    total_gt = 0
    total_preds = 0
    
    # Strict E2E (IoU + Class)
    e2e_tp = 0
    class_stats = {0: {'tp':0, 'fp':0, 'fn':0}, 1: {'tp':0, 'fp':0, 'fn':0}, 2: {'tp':0, 'fp':0, 'fn':0}}
    
    # Detection Only (IoU only)
    det_tp = 0
    
    # Conditional Classification (Given YOLO found it, did ResNet get it right?)
    cond_cls_correct = 0
    cond_cls_total = 0
    
    for img_path in images:
        img_name = os.path.basename(img_path)
        lbl_path = os.path.join(test_label_dir, img_name.rsplit('.', 1)[0] + '.txt')
        
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        # Load GT
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    c, x, y, w, h = map(float, line.strip().split())
                    c = int(c)
                    x1 = (x - w/2) * img_w
                    y1 = (y - h/2) * img_h
                    x2 = (x + w/2) * img_w
                    y2 = (y + h/2) * img_h
                    gt_boxes.append({'class': c, 'box': [x1, y1, x2, y2], 'matched_e2e': False, 'matched_det': False})
                    class_stats[c]['fn'] += 1 # Initially all are FN
                    total_gt += 1
        
        # 1. Detect
        results = detector(img_path, conf=conf_thresh, iou=0.50, verbose=False)[0]
        
        # 2. Predict Subtypes
        preds = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2-x1, y2-y1
            px, py = int(w * padding_factor), int(h * padding_factor)
            cx1, cy1 = max(0, int(x1-px)), max(0, int(y1-py))
            cx2, cy2 = min(img_w, int(x2+px)), min(img_h, int(y2+py))
            
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0: continue
            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            inp = transform(pil_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = classifier(inp)
                _, pred_cls = torch.max(out, 1)
                pred_c = pred_cls.item()
            
            preds.append({'box': [x1, y1, x2, y2], 'class': pred_c})
            total_preds += 1
            class_stats[pred_c]['fp'] += 1 # Initially mark as FP
            
        # 3. Match
        for p in preds:
            best_iou = 0
            best_gt_idx = -1
            for i, gt in enumerate(gt_boxes):
                if gt['matched_det']: continue # A GT box can only be matched once per image
                iou = bbox_iou(p['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_thresh:
                gt_boxes[best_gt_idx]['matched_det'] = True
                det_tp += 1
                cond_cls_total += 1
                
                gt_c = gt_boxes[best_gt_idx]['class']
                if p['class'] == gt_c:
                    cond_cls_correct += 1
                    # TRUE POSITIVE E2E
                    e2e_tp += 1
                    gt_boxes[best_gt_idx]['matched_e2e'] = True
                    class_stats[gt_c]['tp'] += 1
                    class_stats[gt_c]['fp'] -= 1
                    class_stats[gt_c]['fn'] -= 1
                    
    # Global Metrics
    e2e_p = e2e_tp / total_preds if total_preds > 0 else 0
    e2e_r = e2e_tp / total_gt if total_gt > 0 else 0
    e2e_f1 = 2 * (e2e_p * e2e_r) / (e2e_p + e2e_r) if (e2e_p + e2e_r) > 0 else 0
    
    det_p = det_tp / total_preds if total_preds > 0 else 0
    det_r = det_tp / total_gt if total_gt > 0 else 0
    det_f1 = 2 * (det_p * det_r) / (det_p + det_r) if (det_p + det_r) > 0 else 0
    
    cond_acc = cond_cls_correct / cond_cls_total if cond_cls_total > 0 else 0
    
    print("\\n=======================================================")
    print("STAGE 1: DETECTION-ONLY METRICS (IoU >= 0.50, Ignoring Class)")
    print("=======================================================")
    print(f"Total GT Defects:    {total_gt}")
    print(f"Total Predictions:   {total_preds}")
    print(f"Detection TP:        {det_tp}")
    print(f"Detection FP:        {total_preds - det_tp}")
    print(f"Detection FN:        {total_gt - det_tp}")
    print(f"-> Detection Precision: {det_p*100:.2f}%")
    print(f"-> Detection Recall:    {det_r*100:.2f}%")
    print(f"-> Detection F1-Score:  {det_f1*100:.2f}%")
    
    print("\\n=======================================================")
    print("STAGE 2: CONDITIONAL CLASSIFICATION ACCURACY")
    print("=======================================================")
    print(f"Out of {cond_cls_total} correctly detected boxes, ResNet classified {cond_cls_correct} correctly.")
    print(f"-> Conditional Classification Accuracy: {cond_acc*100:.2f}%")
    
    print("\\n=======================================================")
    print("STAGE 3: STRICT END-TO-END METRICS (IoU >= 0.50 AND Exact Subtype)")
    print("=======================================================")
    print(f"Strict E2E TP:       {e2e_tp}")
    print(f"-> E2E Precision:    {e2e_p*100:.2f}%")
    print(f"-> E2E Recall:       {e2e_r*100:.2f}%")
    print(f"-> E2E F1-Score:     {e2e_f1*100:.2f}%")
    
    print("\\nPer-Class E2E Metrics:")
    for c_id, name in enumerate(class_names):
        ctp = class_stats[c_id]['tp']
        cfp = class_stats[c_id]['fp']
        cfn = class_stats[c_id]['fn']
        cp = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0
        cr = ctp / (ctp + cfn) if (ctp + cfn) > 0 else 0
        cf1 = 2 * (cp * cr) / (cp + cr) if (cp + cr) > 0 else 0
        print(f"{name:20s} | P: {cp*100:.2f}% | R: {cr*100:.2f}% | F1: {cf1*100:.2f}% | TP: {ctp}, FP: {cfp}, FN: {cfn}")
        
    return e2e_p, e2e_r, e2e_f1, class_stats

print("Executing LIVE E2E Pipeline on Test Set...")
test_img_dir = '/content/test_3class/test/images'
test_lbl_dir = '/content/test_3class/test/labels'

if os.path.exists(test_img_dir):
    p_e2e, r_e2e, f1_e2e, e2e_class_stats = run_live_e2e(test_img_dir, test_lbl_dir)
    
    # Plot Global E2E
    plt.figure(figsize=(6, 5))
    metrics = [p_e2e*100, r_e2e*100, f1_e2e*100]
    names = ['E2E Precision', 'E2E Recall', 'E2E F1-Score']
    ax = sns.barplot(x=names, y=metrics, palette='magma')
    plt.ylim(0, 100)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 2), ha='center', fontweight='bold')
    plt.title('LIVE End-to-End Test Performance', fontsize=14, fontweight='bold')
    plt.show()
else:
    print(f"Error: Could not find {test_img_dir}")
"""))

cells.append(m("==================================================\n# PART 10 — HISTORICAL EXPERIMENT RESULTS\n=================================================="))

cells.append(c("""\
# Verified from actual original reports/artifacts.
history = [
    {"Run": "EXP-10 YOLO11m 640", "Recall": 76.60, "Precision": 66.60, "F1": 71.25, "mAP50": 70.10},
    {"Run": "EXP-12B YOLO11l 640", "Recall": 73.91, "Precision": 68.77, "F1": 71.25, "mAP50": 71.41},
    {"Run": "EXP-13 YOLO11m 1024", "Recall": 77.17, "Precision": 68.19, "F1": 71.68, "mAP50": 67.26},
    {"Run": "EXP-15 YOLOv8l 1024", "Recall": 66.30, "Precision": 66.19, "F1": 66.24, "mAP50": 65.02},
    {"Run": "EXP-16 YOLO11m-P2", "Recall": 66.67, "Precision": 69.04, "F1": 67.83, "mAP50": 67.64},
    {"Run": "EXP-17 Copy-Paste", "Recall": 73.37, "Precision": 62.66, "F1": 67.58, "mAP50": 69.91},
    {"Run": "EXP-18 Copy-Paste (AdamW)", "Recall": 72.28, "Precision": 66.61, "F1": 69.33, "mAP50": 68.50}
]
df_hist = pd.DataFrame(history)

print("--- HISTORICAL EXPERIMENT COMPARISON ---")
print("Best-performing evaluated detector: EXP-10 YOLO11m")

plt.figure(figsize=(14, 6))
ax = sns.barplot(data=df_hist, x='Recall', y='Run', palette='crest')
plt.title('Detector Historical Recall Tracking (Validation Set)', fontweight='bold')
for p in ax.patches:
    ax.annotate(f"{p.get_width():.2f}%", (p.get_width() + 0.5, p.get_y() + p.get_height()/2.), va='center')
plt.xlim(0, 85)
plt.show()

fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=df_hist, x='Run', y='F1', marker='o', color='purple', label='F1-Score', ax=ax)
sns.lineplot(data=df_hist, x='Run', y='mAP50', marker='s', color='orange', label='mAP50', ax=ax)
plt.title('F1 & mAP50 Historical Trends', fontweight='bold')
plt.xticks(rotation=45)
plt.ylim(60, 75)
plt.grid(True)
plt.show()
"""))

cells.append(m("==================================================\n# PART 8 — LIVE IMAGE DEMONSTRATION\n==================================================\n\nUpload a completely new, raw SEM image. The pipeline will process it live."))

cells.append(c("""\
from google.colab import files
import os

print("Upload SEM Images for Demonstration:")
uploaded = files.upload()

def demo_predict(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    
    # Detect
    results = detector(img_path, conf=0.15, iou=0.50, verbose=False)[0]
    predictions = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        det_conf = box.conf[0].item()
        
        # Crop & Pad
        px, py = int((x2-x1) * 0.10), int((y2-y1) * 0.10)
        cx1, cy1 = max(0, int(x1-px)), max(0, int(y1-py))
        cx2, cy2 = min(img_w, int(x2+px)), min(img_h, int(y2+py))
        
        crop = img_rgb[cy1:cy2, cx1:cx2]
        if crop.size == 0: continue
            
        # Classify
        pil_img = Image.fromarray(crop)
        inp = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = classifier(inp)
            probs = torch.nn.functional.softmax(out, dim=1)
            cls_conf, cls_idx = torch.max(probs, 1)
            
        subtype = class_names[cls_idx.item()]
        
        predictions.append({
            'Box': f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]",
            'Det Conf': f"{det_conf*100:.1f}%",
            'Subtype': subtype,
            'Cls Conf': f"{cls_conf.item()*100:.1f}%"
        })
        
        # Draw
        cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{subtype} (D:{det_conf:.2f} C:{cls_conf.item():.2f})"
        cv2.putText(img_rgb, label, (int(x1), max(10, int(y1)-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"LIVE Predictions for {os.path.basename(img_path)}", fontweight='bold')
    plt.show()
    
    if predictions:
        from tabulate import tabulate
        print("\\nPrediction Details:")
        print(tabulate(predictions, headers='keys', tablefmt='fancy_grid'))
    else:
        print("\\nNo defects detected.")

for filename in uploaded.keys():
    print(f"\\nProcessing {filename}...")
    demo_predict(filename)
"""))

nb['cells'] = cells
nb_path = os.path.join("c:\\Users\\syed mohammad anas\\Desktop\\sir_khurram_project\\sem_defect_project\\notebooks", "FINAL_ANALYSIS_OF_THE_PIPELINE.ipynb")

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook successfully updated at {nb_path}")
