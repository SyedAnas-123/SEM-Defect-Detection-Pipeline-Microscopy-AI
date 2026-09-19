import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 15: YOLOv8l High-Resolution (RESUME RUN)\n\nThis notebook resumes training from `last.pt`. No progress is lost!"),
    
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

    nbf.v4.new_markdown_cell("## 1. Resume Training"),

    nbf.v4.new_code_cell("""\
import time

# Load the LAST saved weights from the interrupted run
last_weights = 'runs/detect/EXP15-YOLOv8l-1024/weights/last.pt'
if not os.path.exists(last_weights):
    print(f"Error: Could not find {last_weights}")
else:
    model = YOLO(last_weights)

    print("Resuming Full Run...")
    t0 = time.time()

    # Resume training
    results = model.train(resume=True)

    t1 = time.time()
    print(f"\\nResume Run Complete in {(t1 - t0)/60:.2f} minutes.")
    print("Best weights saved to: runs/detect/EXP15-YOLOv8l-1024/weights/best.pt")
""")
]

nb['cells'] = cells
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '17_exp15_yolov8l_resume_run.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated at {notebook_path}")
