import nbformat as nbf
import os
import json

# We will reuse the TwoStagePipeline code from 08_final_evaluation_V2.ipynb for Part B
input_nb = 'c:/Users/syed mohammad anas/Desktop/sir_khurram_project/sem_defect_project/notebooks/08_final_evaluation_V2.ipynb'
output_nb = 'c:/Users/syed mohammad anas/Desktop/sir_khurram_project/sem_defect_project/notebooks/10_final_resnet_and_e2e_evaluation.ipynb'

with open(input_nb, 'r', encoding='utf-8') as f:
    old_nb = json.load(f)

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("# Phase 8 Final Evaluation: EXP-11-RunB2 ResNet-50\n\nThis notebook evaluates the optimized ResNet-50 classifier on the strictly untouched **Test Set**, and then runs the complete Two-Stage pipeline End-to-End."))

cells.append(nbf.v4.new_code_cell("""\
# Mount Google Drive
from google.colab import drive
import os
import zipfile

drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'
os.chdir(PROJECT_ROOT)
print(f"Current working directory: {os.getcwd()}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Install dependencies
!pip install ultralytics pandas pyyaml scikit-learn torch torchvision seaborn matplotlib
"""))

# PART A: Classifier Test Evaluation
cells.append(nbf.v4.new_markdown_cell("## PART A: Standalone ResNet Classifier Evaluation (Test Set)\nEvaluating the EXP-11-RunB2 model on the test split of `classifier_crops.zip`."))

cells.append(nbf.v4.new_code_cell("""\
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Extract dataset if not already done
ZIP_PATH = os.path.join(PROJECT_ROOT, 'classifier_crops.zip')
DATASET_ROOT = '/content/classifier_crops'

if not os.path.exists(DATASET_ROOT):
    print("Extracting classifier crops...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('/content')
    print("Extraction complete.")

# Setup Dataloader for Test Set
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_ds = datasets.ImageFolder(os.path.join(DATASET_ROOT, 'test'), transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
class_names = test_ds.classes

# Load EXP-11-RunB2
model_path = os.path.join(PROJECT_ROOT, 'runs/classify/EXP-11-RunB2/best.pt')
print(f"Loading weights from {model_path}...")

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 3))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average='macro')
weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
report = classification_report(all_labels, all_preds, target_names=class_names)

print("\\n=== PART A: RESNET CLASSIFIER TEST SET RESULTS ===")
print(f"Accuracy:    {acc:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print("\\nClassification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Classifier Test Set Confusion Matrix')
plt.ylabel('Ground Truth')
plt.xlabel('Prediction')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## PART B: End-to-End Evaluation (Two-Stage Pipeline)\nRunning the full pipeline using the embedded classes."))

# Find where the E2E pipeline code starts in the old notebook
# The old notebook had cells for the E2E pipeline which we need to copy.
# Let's extract cells that define the pipeline and run the E2E evaluation.
# In 08_final_evaluation_V2.ipynb, cells 4 to 9 contain the pipeline definition.
# Cell 10 contains the execution.

for i, cell in enumerate(old_nb['cells']):
    # We want the E2E pipeline cells (which start after the zip extraction cell in 08_v2).
    if i >= 4:
        source = "".join(cell['source'])
        # Globally replace EXP-03 with EXP-11-RunB2 in all copied cells
        if 'EXP-03-ResNet50' in source:
            source = source.replace('EXP-03-ResNet50', 'EXP-11-RunB2')
        if 'runs/end_to_end_EXP10' in source:
            source = source.replace('runs/end_to_end_EXP10', 'runs/end_to_end_EXP11_RunB2')
        cell['source'] = [source]
        cells.append(cell)

nb['cells'] = cells

os.makedirs(os.path.dirname(output_nb), exist_ok=True)
with open(output_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Notebook generated successfully at {output_nb}")
