import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 7: YOLOv12 Optimization Study (Part 1 - Training)\n\nThis notebook runs multiple YOLO training experiments to optimize Stage 1 recall. It focuses on Image Resolution, Training Strategies, and Augmentation. \n\n**Note: This notebook uses ONLY the training and validation sets.**"),
    
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

PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'
os.chdir(PROJECT_ROOT)
print(f"Current working directory: {os.getcwd()}")"""),

    nbf.v4.new_code_cell("""\
# We will use the data directly from Google Drive as requested.
data_yaml_path = 'dataset_yolo_single_class/data.yaml'

print(f"Using dataset configuration from: {data_yaml_path}")"""),

    nbf.v4.new_markdown_cell("## 1. EXP-05: Resolution 640x640\nTests if a moderate increase in resolution improves small object detection."),

    nbf.v4.new_code_cell("""\
from ultralytics import YOLO

# Initialize YOLOv12 (or YOLO11 fallback)
try:
    model_640 = YOLO('yolov12n.pt')
except Exception:
    print("YOLOv12 weights not found, using YOLO11n fallback.")
    model_640 = YOLO('yolo11n.pt')

results_640 = model_640.train(
    data=data_yaml_path,
    epochs=40,
    imgsz=640,
    batch=32,
    project='runs/detect/optimization',
    name='EXP-05-Res640'
)
"""),

    nbf.v4.new_markdown_cell("## 2. EXP-06: Resolution 768x768\nTests if pushing resolution even higher yields better recall for microporosities."),

    nbf.v4.new_code_cell("""\
try:
    model_768 = YOLO('yolov12n.pt')
except Exception:
    model_768 = YOLO('yolo11n.pt')

results_768 = model_768.train(
    data=data_yaml_path,
    epochs=40,
    imgsz=768,
    batch=16, # Reduced batch size to fit in GPU memory
    project='runs/detect/optimization',
    name='EXP-06-Res768'
)
"""),

    nbf.v4.new_markdown_cell("## 3. EXP-07: Enhanced Training (100 Epochs, AdamW)\nWe use the best resolution (e.g., 640) and push training to 100 epochs with AdamW and early stopping to ensure convergence."),

    nbf.v4.new_code_cell("""\
# Note: You can change imgsz=640 to 768 if EXP-06 was vastly superior
try:
    model_opt = YOLO('yolov12n.pt')
except Exception:
    model_opt = YOLO('yolo11n.pt')

results_opt = model_opt.train(
    data=data_yaml_path,
    epochs=100,
    patience=20, # Early stopping
    imgsz=640, # Using 640 as a safe baseline, adjust if 768 was better
    batch=32,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.01,
    project='runs/detect/optimization',
    name='EXP-07-AdamW100'
)
"""),

    nbf.v4.new_markdown_cell("## 4. EXP-08: Conservative Augmentation\nWe disable aggressive augmentations (Mosaic, MixUp) to preserve SEM textures. We base this on the enhanced training setup."),

    nbf.v4.new_code_cell("""\
try:
    model_aug = YOLO('yolov12n.pt')
except Exception:
    model_aug = YOLO('yolo11n.pt')

results_aug = model_aug.train(
    data=data_yaml_path,
    epochs=100,
    patience=20,
    imgsz=640,
    batch=32,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.01,
    mosaic=0.0, # Disable Mosaic
    mixup=0.0,  # Disable Mixup
    degrees=0.0, # Handled offline
    flipud=0.0, # Handled offline
    fliplr=0.0, # Handled offline
    project='runs/detect/optimization',
    name='EXP-08-NoMosaic'
)
""")
]

nb['cells'] = cells

notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '06_yolo_optimization.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated successfully at {notebook_path}")
