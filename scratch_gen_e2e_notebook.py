import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Markdown and code cells for the notebook
cells = [
    nbf.v4.new_markdown_cell("# Phase 6: End-to-End Evaluation on Colab\n\nThis notebook evaluates the two-stage pipeline on the untouched test set using YOLOv12 and ResNet-50 checkpoints. All pipeline code is embedded directly in this notebook so you don't need to upload local Python scripts to Drive."),
    
    nbf.v4.new_code_cell("""\
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""),

    nbf.v4.new_code_cell("""\
# Install dependencies
!pip install ultralytics"""),

    nbf.v4.new_code_cell("""\
# Set up paths
import os
import sys

PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'
os.chdir(PROJECT_ROOT)

print(f"Current working directory: {os.getcwd()}")"""),

    nbf.v4.new_code_cell("""\
# Extract the true 3-class test set to Colab's high-speed local storage
import zipfile
import shutil
import glob

zip_path = 'test_3class.zip'
local_extract_dir = '/content/test_3class'

if os.path.exists(zip_path):
    print(f"Extracting {zip_path} to {local_extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(local_extract_dir)
    print("Extracted successfully.")
elif os.path.exists('/content/test_3class.zip'):
    print(f"Extracting /content/test_3class.zip to {local_extract_dir}...")
    with zipfile.ZipFile('/content/test_3class.zip', 'r') as zip_ref:
        zip_ref.extractall(local_extract_dir)
    print("Extracted successfully.")
else:
    print("test_3class.zip not found! Please upload it.")

# Fix Windows backslash issue if extracted on Linux
import os
import shutil

for root, dirs, files in os.walk(local_extract_dir):
    for file in files:
        if '\\\\' in file:
            # The zip extracted files with backslashes in their names instead of folders
            new_path = file.replace('\\\\', '/')
            full_new_path = os.path.join(local_extract_dir, new_path)
            
            # Create directories
            os.makedirs(os.path.dirname(full_new_path), exist_ok=True)
            
            # Move file
            shutil.move(os.path.join(root, file), full_new_path)
print("Zip extraction paths normalized.")
"""),

    nbf.v4.new_markdown_cell("## 1. Define Two-Stage Pipeline (Inference Logic)"),

    nbf.v4.new_code_cell("""\
import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np

class TwoStagePipeline:
    def __init__(self, yolo_weights_path, resnet_weights_path, device=None, conf_thresh=0.25, padding_factor=0.10):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.padding_factor = padding_factor
        
        # Load YOLO
        self.detector = YOLO(yolo_weights_path)
        self.detector.to(self.device)
        
        # Load ResNet-50
        self.classifier = models.resnet50(weights=None)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, 3)
        )
        self.classifier.load_state_dict(torch.load(resnet_weights_path, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()
        
        self.class_names = ['Inclusion-Particle', 'Porosity', 'Tear-Delamination']
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        # 1. Detection
        results = self.detector.predict(image_path, conf=self.conf_thresh, verbose=False)
        boxes = results[0].boxes
        
        original_img = Image.open(image_path).convert("RGB")
        img_width, img_height = original_img.size
        
        predictions = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            yolo_conf = float(box.conf[0].cpu().numpy())
            
            # Apply 10% contextual padding
            w = x2 - x1
            h = y2 - y1
            pad_x = w * self.padding_factor
            pad_y = h * self.padding_factor
            
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(img_width, x2 + pad_x)
            crop_y2 = min(img_height, y2 + pad_y)
            
            # 2. Crop
            crop_img = original_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            
            # 3. Classification
            input_tensor = self.preprocess(crop_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.classifier(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                resnet_conf, preds = torch.max(probs, 1)
                
            pred_class = self.class_names[preds.item()]
            resnet_conf = float(resnet_conf.item())
            
            predictions.append({
                'bbox': [x1, y1, x2, y2],
                'yolo_conf': yolo_conf,
                'pred_class': pred_class,
                'resnet_conf': resnet_conf
            })
            
        return predictions, original_img

    def draw_predictions(self, image, predictions, output_path=None):
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("LiberationSans-Regular.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            
        colors = {
            'Inclusion-Particle': 'red',
            'Porosity': 'blue',
            'Tear-Delamination': 'green'
        }
        
        for p in predictions:
            x1, y1, x2, y2 = p['bbox']
            cls_name = p['pred_class']
            y_conf = p['yolo_conf']
            r_conf = p['resnet_conf']
            
            color = colors.get(cls_name, 'yellow')
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            label = f"{cls_name}\\nYOLO: {y_conf:.2f}\\nResNet: {r_conf:.2f}"
            
            # Text background
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(text_bbox, fill="black")
            draw.text((x1, y1), label, fill="white", font=font)
            
        if output_path:
            image.save(output_path)
            
        return image
"""),

    nbf.v4.new_markdown_cell("## 2. Define Evaluation Logic"),

    nbf.v4.new_code_cell("""\
import glob
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_iou(box1, box2):
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
    union_area = float(box1_area + box2_area - intersection_area)

    return intersection_area / union_area if union_area > 0 else 0.0

def load_ground_truth(label_path, img_width, img_height, class_names):
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            x1 = x_center - (width / 2)
            y1 = y_center - (height / 2)
            x2 = x_center + (width / 2)
            y2 = y_center + (height / 2)
            
            gt_boxes.append({
                'bbox': [x1, y1, x2, y2],
                'class': class_names[class_id]
            })
            
    return gt_boxes

def evaluate_e2e(test_img_dir, test_label_dir, output_dir, pipeline, iou_thresh=0.50):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'correct'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'detection_missed'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'false_positive'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'classification_wrong'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'low_confidence'), exist_ok=True)

    image_paths = glob.glob(os.path.join(test_img_dir, '*.jpg'))
    
    if not image_paths:
        print(f"Error: No images found in {test_img_dir}!")
        return
        
    all_true_classes = []
    all_pred_classes = []
    
    total_gt = 0
    total_preds = 0
    true_positives = 0
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(test_label_dir, label_name)
        
        # Run inference
        preds, img = pipeline.predict(img_path)
        img_width, img_height = img.size
        
        gt_boxes = load_ground_truth(label_path, img_width, img_height, pipeline.class_names)
        
        total_gt += len(gt_boxes)
        total_preds += len(preds)
        
        matched_gt_indices = set()
        
        for p in preds:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt_indices:
                    continue
                iou = calculate_iou(p['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
                    
            if best_iou >= iou_thresh:
                matched_gt_indices.add(best_gt_idx)
                gt_class = gt_boxes[best_gt_idx]['class']
                pred_class = p['pred_class']
                
                all_true_classes.append(gt_class)
                all_pred_classes.append(pred_class)
                
                if gt_class == pred_class:
                    true_positives += 1
                    
                    if p['yolo_conf'] < 0.5 or p['resnet_conf'] < 0.5:
                        out_path = os.path.join(output_dir, 'low_confidence', img_name)
                    else:
                        out_path = os.path.join(output_dir, 'correct', img_name)
                else:
                    out_path = os.path.join(output_dir, 'classification_wrong', img_name)
                
                pipeline.draw_predictions(img.copy(), [p], out_path)
            else:
                out_path = os.path.join(output_dir, 'false_positive', img_name)
                pipeline.draw_predictions(img.copy(), [p], out_path)
                
        # Find missed detections
        for i, gt in enumerate(gt_boxes):
            if i not in matched_gt_indices:
                out_path = os.path.join(output_dir, 'detection_missed', img_name)
                pipeline.draw_predictions(img.copy(), [], out_path)
                
    # Calculate end-to-end metrics
    e2e_precision = true_positives / total_preds if total_preds > 0 else 0
    e2e_recall = true_positives / total_gt if total_gt > 0 else 0
    e2e_f1 = 2 * (e2e_precision * e2e_recall) / (e2e_precision + e2e_recall) if (e2e_precision + e2e_recall) > 0 else 0
    
    clf_report = classification_report(all_true_classes, all_pred_classes, output_dict=True, zero_division=0)
    
    metrics = {
        'total_images': len(image_paths),
        'total_ground_truth': total_gt,
        'total_predictions': total_preds,
        'true_positives_e2e': true_positives,
        'e2e_precision': e2e_precision,
        'e2e_recall': e2e_recall,
        'e2e_f1': e2e_f1,
        'classification_report_on_matched': clf_report
    }
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("End-to-End Metrics:")
    print(f"Precision: {e2e_precision:.4f}")
    print(f"Recall: {e2e_recall:.4f}")
    print(f"F1-Score: {e2e_f1:.4f}")
    print("\\nClassification Report (on matched boxes):")
    print(classification_report(all_true_classes, all_pred_classes, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(all_true_classes, all_pred_classes, labels=pipeline.class_names)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline.class_names, yticklabels=pipeline.class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('End-to-End Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
"""),

    nbf.v4.new_markdown_cell("## 3. Execute End-to-End Run"),
    
    nbf.v4.new_code_cell("""\
# Check if the extracted directory exists
import os
import glob

# The exact absolute paths after the backslash fix
test_img_dir = '/content/test_3class/test/images'
test_label_dir = '/content/test_3class/test/labels'
output_dir = 'runs/end_to_end'

if not os.path.exists(test_img_dir):
    print(f"Directory {test_img_dir} does not exist!")
    print("Contents of /content/test_3class:")
    if os.path.exists('/content/test_3class'):
        for r, d, f in os.walk('/content/test_3class'):
            print(r, d, f[:2])
else:
    print(f"Found images directory: {test_img_dir}")
    print(f"Number of images: {len(glob.glob(os.path.join(test_img_dir, '*.jpg')))}")

# Load config
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = {}

# Extract config values
yolo_conf_thresh = config.get('end_to_end', {}).get('yolo_conf_thresh', 0.25)
iou_thresh = config.get('end_to_end', {}).get('iou_thresh', 0.50)
padding_factor = config.get('end_to_end', {}).get('padding_factor', 0.10)

# Paths for models (using the provided names from the user's run)
yolo_weights = 'runs/detect/EXP-02-YOLO11-SingleClass-Baseline/weights/best.pt'
resnet_weights = 'runs/classify/EXP-03-ResNet50/best.pt'

if not os.path.exists(yolo_weights):
    print(f"Error: YOLO weights not found at {yolo_weights}")
if not os.path.exists(resnet_weights):
    print(f"Error: ResNet weights not found at {resnet_weights}")

print("Initializing Two-Stage Pipeline...")
pipeline = TwoStagePipeline(
    yolo_weights_path=yolo_weights,
    resnet_weights_path=resnet_weights,
    conf_thresh=yolo_conf_thresh,
    padding_factor=padding_factor
)

print(f"Starting Evaluation on {test_img_dir}...")
evaluate_e2e(test_img_dir, test_label_dir, output_dir, pipeline, iou_thresh=iou_thresh)

print("Evaluation Complete. Results saved to runs/end_to_end/")
""")
]

nb['cells'] = cells

notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '05_end_to_end_evaluation.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated successfully at {notebook_path}")
