import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 7 Extension: YOLOv12 Medium Model Scaling\n\nThis notebook scales the YOLO architecture from Nano to Medium and extends training to 200 epochs to break the 66% recall ceiling. It trains purely on the Single-Class Defect Dataset on Google Drive."),
    
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

# Verify GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(torch.cuda.get_device_name(0))"""),

    nbf.v4.new_code_cell("""\
# Verify Dataset Paths
data_yaml_path = 'dataset_yolo_single_class/data.yaml'

if not os.path.exists(data_yaml_path):
    print(f"Error: Could not find {data_yaml_path}! Please ensure the dataset_yolo_single_class folder is at the root of sem_defect_project.")
else:
    print(f"Dataset config found at {data_yaml_path}")
    
import yaml
with open(data_yaml_path, 'r') as f:
    cfg = yaml.safe_load(f)
    print("Classes:", cfg.get('names'))"""),

    nbf.v4.new_markdown_cell("## Step 1: Sanity Test (2 Epochs)\nVerifies that the Medium model can load, compile, and execute properly without OOM errors before launching the 200 epoch run."),

    nbf.v4.new_code_cell("""\
from ultralytics import YOLO

print("Initializing YOLOv12m (Medium) for Sanity Test...")
try:
    model_sanity = YOLO('yolov12m.pt')
except Exception as e:
    print(f"Failed to load yolov12m.pt, using yolo11m.pt fallback. Error: {e}")
    model_sanity = YOLO('yolo11m.pt')

results_sanity = model_sanity.train(
    data=data_yaml_path,
    epochs=2,
    imgsz=640,
    batch=16, # Slightly reduced batch size for Medium model to prevent OOM
    optimizer='AdamW',
    lr0=0.001,
    project='runs/detect',
    name='EXP-10-Sanity-Test'
)

print("\\n✅ Sanity Test Complete! If this succeeded without errors, proceed to the full run below.")"""),

    nbf.v4.new_markdown_cell("## Step 2: Full EXP-10 Training (200 Epochs)\nExecutes the optimized training routine."),

    nbf.v4.new_code_cell("""\
print("Initializing YOLOv12m (Medium) for Full Training...")
try:
    model_full = YOLO('yolov12m.pt')
except Exception:
    model_full = YOLO('yolo11m.pt')

results_full = model_full.train(
    data=data_yaml_path,
    epochs=200,
    patience=50, # Generous early stopping
    imgsz=640,
    batch=16, # Keep at 16 to be safe with Medium model on a T4 GPU
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.01,
    project='runs/detect',
    name='EXP-10-YOLOv12m-640-200'
)

print("\\n✅ EXP-10 Full Training Complete!")
print("Best weights saved at: runs/detect/EXP-10-YOLOv12m-640-200/weights/best.pt")
""")
]

nb['cells'] = cells

notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '06B_yolo_model_scaling.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated successfully at {notebook_path}")
