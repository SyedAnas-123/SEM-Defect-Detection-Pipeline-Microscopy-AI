import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 17: YOLO11m + Copy-Paste Augmentation (FULL RUN)\n\nThis notebook trains the standard YOLO11m model on the augmented `dataset_yolo_single_class_cp` dataset, which artificially injects sub-0.5% area defects using Gaussian Alpha Blending. The goal is to boost Recall above the EXP-10 baseline (76.63%)."),
    
    nbf.v4.new_code_cell("""\
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""),

    nbf.v4.new_code_cell("""\
# Install dependencies
!pip install ultralytics
"""),

    nbf.v4.new_code_cell("""\
import os
import torch
from ultralytics import YOLO

PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'
os.chdir(PROJECT_ROOT)
print(f"Current working directory: {os.getcwd()}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
"""),

    nbf.v4.new_markdown_cell("## 1. Load YOLO11m (Official Weights)"),

    nbf.v4.new_code_cell("""\
# Instantiate standard YOLO11m using official pretrained weights
print("Instantiating standard YOLO11m model...")
model = YOLO('yolo11m.pt')
"""),

    nbf.v4.new_markdown_cell("## 2. FULL Training Run (200 Epochs) on Augmented Dataset"),

    nbf.v4.new_code_cell("""\
import time

print("Starting FULL Run on CP Dataset (Batch=16, 640x640, Epochs=200)...")
t0 = time.time()

results = model.train(
    data='datasets/dataset_yolo_single_class_cp/data.yaml',
    epochs=200,
    patience=30,
    imgsz=640,
    batch=16,
    name='EXP17-YOLO11m-CP-FULL',
    cache=False,
    amp=True,
    exist_ok=True
)

t1 = time.time()
print(f"\\nFull Run Complete in {(t1 - t0)/3600:.2f} hours.")
print("Best weights saved to: runs/detect/EXP17-YOLO11m-CP-FULL/weights/best.pt")
""")
]

nb['cells'] = cells
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '21_exp17_yolo11m_cp_full_run.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated at {notebook_path}")
