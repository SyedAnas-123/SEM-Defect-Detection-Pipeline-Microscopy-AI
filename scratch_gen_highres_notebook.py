import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 13: Full High-Resolution Training (YOLO11m @ 1024px)\n\nThis notebook executes the full 200-epoch `EXP-13` training run. The sanity check confirmed that `batch=4` safely fits within the 15GB Tesla T4 VRAM."),
    
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

    nbf.v4.new_markdown_cell("## 1. Full High-Res Training Run"),

    nbf.v4.new_code_cell("""\
# Load pre-trained weights
model = YOLO('yolo11m.pt')

print("Starting FULL Training Run (Batch=4, Epochs=200)...")

# Train for 200 epochs
results = model.train(
    data='dataset_yolo_single_class/data.yaml',
    epochs=200,
    patience=30,         # Early stopping
    imgsz=1024,          # High-Res Target
    batch=4,             # Safe batch size
    name='EXP13-YOLO11m-1024',
    cache=False,         # Disable RAM caching to prevent crash
    amp=True,            # Enable FP16 Mixed Precision
    exist_ok=True
)

print("\\nTraining Complete! Best model saved to runs/detect/EXP13-YOLO11m-1024/weights/best.pt")
""")
]

nb['cells'] = cells
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '16_exp13_high_res_training.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook updated for FULL training at {notebook_path}")
