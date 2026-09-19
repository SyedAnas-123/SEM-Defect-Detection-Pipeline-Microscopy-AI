import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 7: YOLOv12 Optimization Study (Part 2 - Threshold Sweeps)\n\nThis notebook evaluates the best trained weights from EXPs 5-8 across various Confidence and NMS IoU thresholds using ONLY the validation set."),
    
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
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'
os.chdir(PROJECT_ROOT)
print(f"Current working directory: {os.getcwd()}")"""),

    nbf.v4.new_code_cell("""\
# IMPORTANT: Point this to the best weights from your EXP-10 run.
BEST_WEIGHTS_PATH = 'runs/detect/EXP-10-YOLOv12m-640-200/weights/best.pt'
DATA_YAML = 'dataset_yolo_single_class/data.yaml'

print(f"Loading weights from: {BEST_WEIGHTS_PATH}")
model = YOLO(BEST_WEIGHTS_PATH)

# Check architecture
model_yaml = getattr(model.model, 'yaml', {})
arch_name = model_yaml.get('yaml_file', 'Unknown Architecture')
print(f"\\nLoaded Model Architecture: {arch_name}")

"""),

    nbf.v4.new_code_cell("""\
# Define sweep parameters
conf_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
fixed_iou = 0.50 # Standard NMS IoU setting

results_list = []

print(f"Starting Threshold Sweeps on Validation Set (Fixed NMS IoU={fixed_iou})...")

for conf in conf_thresholds:
    print(f"\\n--- Evaluating Conf: {conf} ---")
    
    # Run validation
    metrics = model.val(
        data=DATA_YAML,
        split='val',
        conf=conf,
        iou=fixed_iou,
        verbose=False
    )
    
    # Extract metrics
    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']
    map50 = metrics.results_dict['metrics/mAP50(B)']
    map50_95 = metrics.results_dict['metrics/mAP50-95(B)']
    
    # Calculate F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results_list.append({
        'conf': conf,
        'iou': fixed_iou,
        'precision': precision,
        'recall': recall,
        'map50': map50,
        'map50_95': map50_95,
        'f1': f1
    })

# Create a DataFrame and save to CSV
df = pd.DataFrame(results_list)
os.makedirs('runs/detect/EXP-10-YOLOv12m-640-200', exist_ok=True)
csv_path = 'runs/detect/EXP-10-YOLOv12m-640-200/EXP-10-Threshold-Sweep-Results.csv'
df.to_csv(csv_path, index=False)

print(f"\\nSweep Complete! Results saved to {csv_path}")
"""),

    nbf.v4.new_code_cell("""\
# Display the top 5 configurations sorted by F1-Score
display(df.sort_values(by='f1', ascending=False).head(5))

# Also display the top 5 configurations sorted by Recall (while maintaining precision > 0.50)
print("\\nTop configs for Recall (Precision > 0.50):")
filtered_df = df[df['precision'] > 0.50]
display(filtered_df.sort_values(by='recall', ascending=False).head(5))
""")
]

nb['cells'] = cells

notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '07_yolo_threshold_sweep.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated successfully at {notebook_path}")
