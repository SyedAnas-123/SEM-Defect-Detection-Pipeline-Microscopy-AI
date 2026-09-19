import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 15: YOLOv8l High-Resolution Detection (FULL RUN)\n\nThis notebook executes the full 200-epoch training for `yolov8l` at `1024x1024`. The sanity run confirmed `batch=4` used only ~2.5GB VRAM. We are increasing to `batch=8` for faster training while remaining completely safe within the 15GB T4 limit."),
    
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

    nbf.v4.new_markdown_cell("## 1. Full Run (YOLOv8l, 1024x1024, Epochs=200)"),

    nbf.v4.new_code_cell("""\
import time

# Load official COCO-pretrained YOLOv8l
model = YOLO('yolov8l.pt')

print("Starting Full Run (Batch=8)...")
t0 = time.time()

# Train for 200 epochs
results = model.train(
    data='dataset_yolo_single_class/data.yaml',
    epochs=200,
    patience=30,
    imgsz=1024,          # High-Res Target
    batch=8,             # Increased from 4 to 8 for speed
    name='EXP15-YOLOv8l-1024',
    cache=False,         # Disable RAM caching
    amp=True,            # Enable FP16 Mixed Precision
    exist_ok=True
)

t1 = time.time()
print(f"\\nFull Run Complete in {(t1 - t0)/60:.2f} minutes.")
print("Best weights saved to: runs/detect/EXP15-YOLOv8l-1024/weights/best.pt")
""")
]

nb['cells'] = cells
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '17_exp15_yolov8l_full_run.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated at {notebook_path}")
