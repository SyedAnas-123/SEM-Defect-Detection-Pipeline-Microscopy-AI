import os
import torch
import time
from ultralytics import YOLO

# GPU check
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    torch.cuda.empty_cache()

data_yaml = r'c:\Users\syed mohammad anas\Desktop\sir_khurram_project\sem_defect_project\datasets\dataset_yolo_single_class\data.yaml'

print('Loading YOLO11l model...')
model = YOLO('yolo11l.pt')

print('Starting training for EXP-12B (YOLO11l)...')
start_time = time.time()

# Train the model
results = model.train(
    data=data_yaml,
    epochs=200,
    imgsz=640,
    batch=16,
    project='runs/detect',
    name='EXP-12B',
    seed=42,
    patience=50, # early stopping
    device=0
)

train_time = time.time() - start_time
print(f'\nTraining completed in {train_time/60:.2f} minutes.')

# Validate the model using best weights
print('\nRunning validation on the val split...')
val_results = model.val(
    data=data_yaml,
    split='val',
    imgsz=640,
    batch=16,
    device=0,
    conf=0.15,
    iou=0.50
)

print('\n--- EXP-12B Validation Results ---')
print(f"Precision: {val_results.results_dict.get('metrics/precision(B)', 0):.4f}")
print(f"Recall: {val_results.results_dict.get('metrics/recall(B)', 0):.4f}")
print(f"mAP50: {val_results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
print(f"mAP50-95: {val_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

# Calculate F1 score
p = val_results.results_dict.get('metrics/precision(B)', 0)
r = val_results.results_dict.get('metrics/recall(B)', 0)
f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
print(f"F1 Score: {f1:.4f}")
