import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 15: YOLOv8l High-Resolution Detection (Sanity Run)\n\nThis notebook tests the YOLOv8l (Large) architecture natively at `1024x1024` on a Tesla T4 to ensure we don't OOM. We are starting with `batch=4` for 3 epochs."),
    
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

    nbf.v4.new_code_cell("""\
# Verify baseline weights exist
weights_path = 'yolov8l.pt'
if not os.path.exists(weights_path):
    print("Downloading baseline yolov8l.pt...")
    YOLO(weights_path)
"""),

    nbf.v4.new_markdown_cell("## 1. Sanity Run (Batch=4, Epochs=3)"),

    nbf.v4.new_code_cell("""\
import time

# Load official COCO-pretrained YOLOv8l
model = YOLO('yolov8l.pt')

print("Starting Sanity Run (Batch=4)...")
t0 = time.time()

# Train for 3 epochs to test VRAM
results = model.train(
    data='dataset_yolo_single_class/data.yaml',
    epochs=3,
    imgsz=1024,          # High-Res Target
    batch=4,             # Testing batch=4 for 15GB VRAM
    name='EXP15-Sanity-v8l-1024',
    cache=False,         # Disable RAM caching
    amp=True,            # Enable FP16 Mixed Precision
    exist_ok=True
)

t1 = time.time()
print(f"\\nSanity Run Complete in {t1 - t0:.2f} seconds.")
"""),

    nbf.v4.new_code_cell("""\
# Print Peak VRAM Usage
print("\\n=== TESLA T4 VRAM USAGE ===")
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
""")
]

nb['cells'] = cells
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '17_exp15_yolov8l_sanity_run.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated at {notebook_path}")
