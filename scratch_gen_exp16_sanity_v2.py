import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 16: YOLO11m + P2 Head (Sanity Run V2)\n\nThis notebook dynamically creates the corrected `yolo11-p2.yaml` architecture (with `m: [0.50, 1.00, 512]`), loads `yolo11m.pt` weights (verifying transfer), and runs a 3-epoch sanity training to ensure mathematically correct operation on the Tesla T4 without OOM."),
    
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
# Create Custom yolo11-p2.yaml (FIXED V2)
yaml_content = '''
# YOLO11-P2 (Stride-4 High Resolution Head)
nc: 1
scales:
  # [depth, width, max_channels]
  m: [0.50, 1.00, 512] # CORRECTED YOLO11m scale

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]] # 2
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]] # 4
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]] # 6
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]] # 8
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  # --- P2 EXTENSION ---
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]] # cat backbone P2
  - [-1, 2, C3k2, [128, False]] # 19 (P2/4-tiny)

  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]] # cat head P3
  - [-1, 2, C3k2, [256, False]] # 22 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 25 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 28 (P5/32-large)

  - [[19, 22, 25, 28], 1, Detect, [nc]] # Detect(P2, P3, P4, P5)
'''

with open('yolo11-p2.yaml', 'w') as f:
    f.write(yaml_content)

print("Created yolo11-p2.yaml successfully (V2 Corrected)!")
"""),

    nbf.v4.new_code_cell("""\
# Instantiate Model and Load Weights
print("Instantiating YOLO11m-P2 model...")
model = YOLO('yolo11-p2.yaml')

print("\\nLoading pretrained YOLO11m weights into P2 architecture...")
# Load weights, matching shapes automatically
model.load('yolo11m.pt')

# Print full architecture summary to verify P2/P3/P4/P5 shapes
model.info(detailed=True)
"""),

    nbf.v4.new_markdown_cell("## 2. Sanity Training Run V2"),

    nbf.v4.new_code_cell("""\
import time

print("Starting 3-Epoch Sanity Run V2 (Batch=16, 640x640)...")
t0 = time.time()

results = model.train(
    data='dataset_yolo_single_class/data.yaml',
    epochs=3,
    imgsz=640,
    batch=16,
    name='EXP16-SanityV2-YOLO11m-P2',
    cache=False,
    amp=True,
    exist_ok=True
)

t1 = time.time()
print(f"\\nSanity Run V2 Complete in {t1 - t0:.2f} seconds.")
"""),

    nbf.v4.new_code_cell("""\
# Print Peak VRAM Usage
print("\\n=== TESLA T4 VRAM USAGE ===")
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
""")
]

nb['cells'] = cells
notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '19_exp16_yolo11m_p2_sanity_v2.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated at {notebook_path}")
