import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 18: YOLO11m + Copy-Paste (EXACT EXP-10 Optimizer)\\n\\nThis notebook trains the standard YOLO11m model on the augmented `dataset_yolo_single_class_cp` dataset using the EXACT same hyperparameter configuration as the EXP-10 baseline (`AdamW`, `lr0=0.001`, `weight_decay=0.01`), but with patience=30 as requested."),
    
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

    nbf.v4.new_markdown_cell("## 2. FULL Training Run (EXACT EXP-10 HYPERPARAMETERS)"),

    nbf.v4.new_code_cell("""\
import time

print("Starting FULL Run on CP Dataset (Batch=16, 640x640, Epochs=200)...")
print("Enforcing EXP-10 Exact Optimizer: AdamW, lr0=0.001, weight_decay=0.01, patience=30")

t0 = time.time()

results = model.train(
    data='datasets/dataset_yolo_single_class_cp/data.yaml',
    epochs=200,
    patience=30,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.01,
    name='EXP18-YOLO11m-CP-Exact-Opt',
    cache=False,
    amp=True,
    exist_ok=True
)

t1 = time.time()
print(f"\\nFull Run Complete in {(t1 - t0)/3600:.2f} hours.")
print("Best weights saved to: runs/detect/EXP18-YOLO11m-CP-Exact-Opt/weights/best.pt")
""")
]

nb['cells'] = cells
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '22_exp18_yolo11m_cp_exact_opt.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated at {notebook_path}")
