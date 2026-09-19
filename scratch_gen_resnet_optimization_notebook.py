import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("# Phase 8: EXP-11 ResNet-50 Classifier Optimization Sweep\n\nThis notebook evaluates 7 different controlled configurations for ResNet-50 Stage 2 Classification, purely on the Validation Set."))

cells.append(nbf.v4.new_code_cell("""\
# 1. Setup & Environment Audit
!pip install torch torchvision pandas matplotlib seaborn scikit-learn
"""))

cells.append(nbf.v4.new_code_cell("""\
# 2. Mount Google Drive & Locate / Extract Classifier Dataset
from google.colab import drive
import os
import zipfile

drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'
os.chdir(PROJECT_ROOT)

ZIP_PATH = os.path.join(PROJECT_ROOT, 'classifier_crops.zip')
DATASET_ROOT = '/content/classifier_crops'

if not os.path.exists(DATASET_ROOT):
    print("Extracting classifier crops to local Colab storage...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('/content')
    print("Extraction complete.")
else:
    print("Dataset already extracted.")
"""))

cells.append(nbf.v4.new_code_cell("""\
# 3. Imports and Setup
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np
import copy
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# 4. Engine Functions
def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, run_name):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    model.load_state_dict(best_model_wts)
    return model, best_f1, best_epoch, time_elapsed

def evaluate_run(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    return acc, macro_f1, weighted_f1, report
"""))

cells.append(nbf.v4.new_code_cell("""\
# 5. Configurations to Run
configs = [
    # =====================================================================
    # ALREADY COMPLETED RUNS (Commented out to prevent re-training)
    # These runs successfully finished before the Colab kernel restarted.
    # Their weights are safely stored in runs/classify/
    # =====================================================================
    # {
    #     "run_name": "EXP-11-RunA1",
    #     "unfreeze": "layer4",
    #     "lr": 1e-4,
    #     "backbone_lr": 1e-4,
    #     "weight_strategy": "baseline_norm",
    #     "aug": "baseline"
    # },
    # {
    #     "run_name": "EXP-11-RunA2",
    #     "unfreeze": "layer4",
    #     "lr": 1e-4,
    #     "backbone_lr": 1e-5,
    #     "weight_strategy": "baseline_norm",
    #     "aug": "baseline"
    # },
    # {
    #     "run_name": "EXP-11-RunA3",
    #     "unfreeze": "all",
    #     "lr": 1e-4,
    #     "backbone_lr": 1e-5,
    #     "weight_strategy": "baseline_norm",
    #     "aug": "baseline"
    # },
    
    # =====================================================================
    # REMAINING RUNS TO EXECUTE
    # =====================================================================
    {
        "run_name": "EXP-11-RunB1",
        "unfreeze": "layer4",
        "lr": 1e-4,
        "backbone_lr": 1e-4,
        "weight_strategy": "baseline_norm",
        "aug": "baseline"
    },
    {
        "run_name": "EXP-11-RunB2",
        "unfreeze": "layer4",
        "lr": 1e-4,
        "backbone_lr": 1e-4,
        "weight_strategy": "inverse_freq",
        "aug": "baseline"
    },
    {
        "run_name": "EXP-11-RunC1",
        "unfreeze": "layer4",
        "lr": 1e-4,
        "backbone_lr": 1e-4,
        "weight_strategy": "baseline_norm",
        "aug": "baseline"
    },
    {
        "run_name": "EXP-11-RunC2",
        "unfreeze": "layer4",
        "lr": 1e-4,
        "backbone_lr": 1e-4,
        "weight_strategy": "baseline_norm",
        "aug": "affine"
    }
]

# Get class counts to pre-calculate weights
train_dir = os.path.join(DATASET_ROOT, 'train')
class_names = sorted(os.listdir(train_dir))
class_counts = {c: len(os.listdir(os.path.join(train_dir, c))) for c in class_names}
total_train = sum(class_counts.values())
"""))

cells.append(nbf.v4.new_code_cell("""\
# 5.5 Verify Already Completed Models
import os

print("--- VERIFYING ALREADY TRAINED MODELS ---")
completed_runs = ["EXP-11-RunA1", "EXP-11-RunA2", "EXP-11-RunA3"]
for run in completed_runs:
    weight_path = os.path.join('runs/classify', run, 'best.pt')
    if os.path.exists(weight_path):
        size_mb = os.path.getsize(weight_path) / (1024 * 1024)
        print(f"✅ Found weights for {run}: {weight_path} ({size_mb:.2f} MB)")
    else:
        print(f"❌ WARNING: Weights NOT found for {run} at {weight_path}")
print("----------------------------------------")
"""))

cells.append(nbf.v4.new_code_cell("""\
# 6. Execute Sweep
results = []
os.makedirs('runs/classify', exist_ok=True)

for conf in configs:
    print(f"\\n{'='*50}")
    print(f"STARTING RUN: {conf['run_name']}")
    print(f"Config: {conf}")
    print(f"{'='*50}")
    
    # 1. Setup Augmentations
    if conf['aug'] == 'baseline':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif conf['aug'] == 'affine':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # DataLoaders
    train_ds = datasets.ImageFolder(os.path.join(DATASET_ROOT, 'train'), transform=train_transform)
    valid_ds = datasets.ImageFolder(os.path.join(DATASET_ROOT, 'valid'), transform=val_transform)
    dataloaders = {'train': DataLoader(train_ds, batch_size=32, shuffle=True),
                   'valid': DataLoader(valid_ds, batch_size=32, shuffle=False)}
    dataset_sizes = {'train': len(train_ds), 'valid': len(valid_ds)}
    
    # 2. Setup Model Architecture
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freezing logic
    if conf['unfreeze'] == 'layer4':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif conf['unfreeze'] == 'all':
        for param in model.parameters():
            param.requires_grad = True
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 3))
    model = model.to(device)
    
    # Optimizer groups
    if conf['unfreeze'] == 'layer4':
        optimizer = optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': conf['backbone_lr']},
            {'params': model.fc.parameters(), 'lr': conf['lr']}
        ], weight_decay=1e-3)
    elif conf['unfreeze'] == 'all':
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': conf['backbone_lr']},
            {'params': model.fc.parameters(), 'lr': conf['lr']}
        ], weight_decay=1e-3)
        
    # 3. Setup Loss Function
    if conf['weight_strategy'] == 'baseline_norm':
        weights = [total_train / class_counts[c] for c in class_names]
        min_w = min(weights)
        class_weights = torch.FloatTensor([w / min_w for w in weights]).to(device)
    elif conf['weight_strategy'] == 'inverse_freq':
        # Standard scikit-learn inverse freq formula: N / (C * n_i)
        class_weights = torch.FloatTensor([total_train / (len(class_names) * class_counts[c]) for c in class_names]).to(device)
        
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 4. Train
    print("Training...")
    best_model, best_f1, best_epoch, time_elapsed = train_model(
        model, criterion, optimizer, scheduler, num_epochs=40, dataloaders=dataloaders, dataset_sizes=dataset_sizes, run_name=conf['run_name']
    )
    
    # 5. Evaluate on Validation Set
    print("Evaluating on Validation Set...")
    val_acc, val_macro_f1, val_weighted_f1, report = evaluate_run(best_model, dataloaders['valid'])
    
    # Save checkpoint securely
    save_dir = os.path.join('runs/classify', conf['run_name'])
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(save_dir, 'best.pt'))
    
    res_dict = {
        'Run': conf['run_name'],
        'Val_Acc': val_acc,
        'Val_Macro_F1': val_macro_f1,
        'Val_Weighted_F1': val_weighted_f1,
        'Best_Epoch': best_epoch,
        'Time_sec': time_elapsed,
        '0_Inclusion_F1': report['0']['f1-score'] if '0' in report else report['Inclusion-Particle']['f1-score'],
        '1_Porosity_F1': report['1']['f1-score'] if '1' in report else report['Porosity']['f1-score'],
        '2_Tear_F1': report['2']['f1-score'] if '2' in report else report['Tear-Delamination']['f1-score'],
    }
    results.append(res_dict)
    
    print(f"Results for {conf['run_name']}: Acc={val_acc:.4f}, Macro F1={val_macro_f1:.4f}")

# Final Report Generation
df = pd.DataFrame(results)
df.to_csv('runs/classify/EXP-11-Optimization-Sweep-Results.csv', index=False)
print("\\nSweep Complete! Results saved to runs/classify/EXP-11-Optimization-Sweep-Results.csv")
display(df)
"""))

nb['cells'] = cells

notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '09_resnet_optimization_sweep.ipynb')
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print(f"Notebook generated successfully at {notebook_path}")
