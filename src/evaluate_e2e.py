import os
import glob
import json
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.inference import TwoStagePipeline
import yaml

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
                pipeline.draw_predictions(img.copy(), [], out_path) # Just saving image for now, can be improved to draw GT box
                
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
    print("\nClassification Report (on matched boxes):")
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
    
if __name__ == "__main__":
    # Note: These paths will be overridden when running in Colab notebook
    test_img_dir = 'datasets/original_dataset/test/images'
    test_label_dir = 'datasets/original_dataset/test/labels'
    yolo_weights = 'runs/detect/EXP-02-YOLO11-SingleClass-Baseline/weights/best.pt'
    resnet_weights = 'runs/classify/EXP-03-ResNet50/best.pt'
    output_dir = 'runs/end_to_end'
    
    pipeline = TwoStagePipeline(yolo_weights, resnet_weights)
    evaluate_e2e(test_img_dir, test_label_dir, output_dir, pipeline)
