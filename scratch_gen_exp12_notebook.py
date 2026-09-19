import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 9: EXP-12B YOLO11l Model Scaling\n\nThis notebook trains YOLO11l on the existing single-class dataset to determine if scaling up the architecture improves Stage-1 recall over the YOLO11m baseline."),
    
    nbf.v4.new_code_cell("""\
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""),

    nbf.v4.new_code_cell("""\
# Install dependencies
!pip install ultralytics"""),

    nbf.v4.new_code_cell("""\
import os
import sys
import torch

PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'
os.chdir(PROJECT_ROOT)
print(f"Current working directory: {os.getcwd()}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""),

    nbf.v4.new_code_cell("""\
# We will use the data directly from Google Drive.
data_yaml_path = 'dataset_yolo_single_class/data.yaml'

print(f"Using dataset configuration from: {data_yaml_path}")"""),

    nbf.v4.new_markdown_cell("## EXP-12B: YOLO11l Training (200 Epochs)"),

    nbf.v4.new_code_cell("""\
from ultralytics import YOLO

# Initialize YOLO11l
model_l = YOLO('yolo11l.pt')

# Add a callback to clearly print the epoch number to avoid Colab truncation hiding progress
def print_epoch(trainer):
    print(f"\\n>>> Completed Epoch {trainer.epoch + 1} / {trainer.epochs} <<<\\n", flush=True)

model_l.add_callback("on_train_epoch_end", print_epoch)

results_l = model_l.train(
    data=data_yaml_path,
    epochs=200,
    patience=30, # Stop after 30 epochs with no validation improvement
    save=True,   # Ensure best.pt is saved
    imgsz=640,
    batch=16,
    project='runs/detect',
    name='EXP-12B_YOLO11l',
    seed=42
)
"""),

    nbf.v4.new_markdown_cell("## Validation Metrics Extraction\nExtract and print the precise validation metrics for comparison."),

    nbf.v4.new_code_cell("""\
# Validate the model using best weights
val_results = model_l.val(
    data=data_yaml_path,
    split='val',
    imgsz=640,
    batch=16,
    device=0,
    conf=0.15,
    iou=0.50
)

print('\\n--- EXP-12B Validation Results ---')
p = val_results.results_dict.get('metrics/precision(B)', 0)
r = val_results.results_dict.get('metrics/recall(B)', 0)
f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

print(f"Precision: {p:.4f}")
print(f"Recall: {r:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"mAP50: {val_results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
print(f"mAP50-95: {val_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
print(f"Training parameters: {model_l.info()}")
""")
]

nb['cells'] = cells

notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '11_exp12_large_model_scaling.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated successfully at {notebook_path}")
