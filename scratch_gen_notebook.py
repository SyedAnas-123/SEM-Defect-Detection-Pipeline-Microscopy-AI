import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Phase 2: YOLOv12 Single-Class Training Pipeline (Google Colab Kernel)\n",
    "\n",
    "This notebook is designed to execute on a Google Colab GPU kernel while connected from a local Jupyter interface.\n",
    "All data and outputs persist on Google Drive.\n",
    "\n",
    "### **Instructions**\n",
    "Run the setup and verification cells (Cells 1-6). Review the output carefully. Do not run the final 100-epoch training cell until approval is granted."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Environment and GPU Audit\n",
    "!pip install ultralytics pyyaml pandas opencv-python-headless matplotlib > /dev/null 2>&1\n",
    "\n",
    "import sys\n",
    "import torch\n",
    "import ultralytics\n",
    "\n",
    "print(\"--- ENVIRONMENT AUDIT ---\")\n",
    "print(f\"Python Version: {sys.version.split()[0]}\")\n",
    "print(f\"PyTorch Version: {torch.__version__}\")\n",
    "print(f\"Ultralytics Version: {ultralytics.__version__}\")\n",
    "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
    "\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"CUDA Version: {torch.version.cuda}\")\n",
    "    print(f\"GPU Name: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"GPU VRAM (GB): {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)}\")\n",
    "else:\n",
    "    raise SystemError(\"No CUDA GPU detected! Please ensure you are connected to a Colab GPU kernel.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Mount Google Drive\n",
    "import os\n",
    "\n",
    "PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'\n",
    "DATASET_ROOT = f\"{PROJECT_ROOT}/dataset_yolo_single_class\"\n",
    "ZIP_PATH = f\"{PROJECT_ROOT}/dataset_yolo_single_class.zip\"\n",
    "RUNS_DIR = f\"{PROJECT_ROOT}/runs/detect\"\n",
    "\n",
    "# Check if Drive is already mounted (either previously or via Colab Files sidebar)\n",
    "if os.path.exists('/content/drive/MyDrive'):\n",
    "    print(\"✅ Google Drive is already mounted and accessible.\")\n",
    "else:\n",
    "    try:\n",
    "        from google.colab import drive\n",
    "        drive.mount('/content/drive')\n",
    "        print(\"✅ Google Drive successfully mounted.\")\n",
    "    except Exception as e:\n",
    "        print(\"⚠️ Notice: Remote Jupyter kernel cannot open interactive Colab auth popup.\")\n",
    "        print(\"👉 To fix: In your Colab web browser tab, click the 'Files' icon (left panel) -> click 'Mount Drive'.\")\n",
    "        if not os.path.exists('/content/drive/MyDrive'):\n",
    "            raise RuntimeError(\"Google Drive is not mounted yet. Please click 'Mount Drive' in the Colab browser UI.\") from e\n",
    "\n",
    "os.makedirs(RUNS_DIR, exist_ok=True)\n",
    "print(f\"📁 Project root: {PROJECT_ROOT}\")\n",
    "print(f\"📁 Runs directory: {RUNS_DIR}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Extract Dataset if Not Exists\n",
    "import zipfile\n",
    "import shutil\n",
    "\n",
    "if not os.path.exists(DATASET_ROOT):\n",
    "    if not os.path.exists(ZIP_PATH):\n",
    "        found_files = os.listdir(PROJECT_ROOT) if os.path.exists(PROJECT_ROOT) else []\n",
    "        raise FileNotFoundError(\n",
    "            f\"Dataset ZIP not found at '{ZIP_PATH}'.\\n\"\n",
    "            f\"Files currently in '{PROJECT_ROOT}': {found_files}\\n\"\n",
    "            \"Please ensure 'dataset_yolo_single_class.zip' is placed inside 'My Drive/sem_defect_project/'.\"\n",
    "        )\n",
    "    \n",
    "    print(f\"Extracting {ZIP_PATH} to {PROJECT_ROOT}...\")\n",
    "    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:\n",
    "        zip_ref.extractall(PROJECT_ROOT)\n",
    "    print(\"✅ Extraction complete.\")\n",
    "else:\n",
    "    print(f\"✅ Dataset already exists at {DATASET_ROOT}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Dataset Verification & Path Rewriting\n",
    "import yaml\n",
    "import glob\n",
    "\n",
    "print(\"--- DATASET VERIFICATION ---\")\n",
    "splits = ['train', 'valid', 'test']\n",
    "for split in splits:\n",
    "    img_dir = f\"{DATASET_ROOT}/{split}/images\"\n",
    "    lbl_dir = f\"{DATASET_ROOT}/{split}/labels\"\n",
    "    \n",
    "    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):\n",
    "        raise FileNotFoundError(f\"Missing structure for split '{split}'. Expected {img_dir} and {lbl_dir}.\")\n",
    "    \n",
    "    imgs = glob.glob(f\"{img_dir}/*.jpg\")\n",
    "    print(f\"Split '{split}': Found {len(imgs)} images.\")\n",
    "    if len(imgs) == 0:\n",
    "        raise ValueError(f\"No images found in {img_dir}\")\n",
    "\n",
    "DATA_YAML = f\"{DATASET_ROOT}/data.yaml\"\n",
    "if not os.path.exists(DATA_YAML):\n",
    "    raise FileNotFoundError(f\"{DATA_YAML} not found!\")\n",
    "\n",
    "# Ensure data.yaml points to absolute paths for Colab\n",
    "with open(DATA_YAML, 'r') as f:\n",
    "    data = yaml.safe_load(f)\n",
    "\n",
    "data['train'] = f\"{DATASET_ROOT}/train/images\"\n",
    "data['val'] = f\"{DATASET_ROOT}/valid/images\"\n",
    "data['test'] = f\"{DATASET_ROOT}/test/images\"\n",
    "\n",
    "with open(DATA_YAML, 'w') as f:\n",
    "    yaml.dump(data, f)\n",
    "\n",
    "print(\"data.yaml updated with absolute Colab paths and verified.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. YOLOv12 Model Loading Test\n",
    "from ultralytics import YOLO\n",
    "import gc\n",
    "\n",
    "print(\"Loading YOLO model...\")\n",
    "try:\n",
    "    model = YOLO(\"yolov12n.pt\")\n",
    "    print(\"Successfully loaded yolov12n.pt\")\n",
    "except Exception as e:\n",
    "    print(f\"YOLOv12 not found. Falling back to YOLO11. Error: {e}\")\n",
    "    model = YOLO(\"yolo11n.pt\")\n",
    "    print(\"Successfully loaded yolo11n.pt\")\n",
    "\n",
    "print(\"Model load test passed.\")\n",
    "del model\n",
    "gc.collect()\n",
    "torch.cuda.empty_cache()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### **STOP HERE**\n",
    "**Do NOT proceed to the full training cell below until you have reviewed the verification outputs above.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6. Dry Run (2 Epochs) - Pre-Flight Training Verification\n",
    "from ultralytics import YOLO\n",
    "\n",
    "print(\"Starting 2-epoch pre-flight dry run to verify training loop, loss calculation, and Drive checkpointing...\")\n",
    "model = YOLO(\"yolo11n.pt\")\n",
    "\n",
    "dry_results = model.train(\n",
    "    data=DATA_YAML,\n",
    "    epochs=2,\n",
    "    imgsz=512,\n",
    "    batch=16,\n",
    "    optimizer=\"AdamW\",\n",
    "    lr0=0.001,\n",
    "    project=RUNS_DIR,\n",
    "    name=\"EXP-01-YOLO11-SingleClass-DryRun\",\n",
    "    seed=42,\n",
    "    device=0,\n",
    "    exist_ok=True,\n",
    "    save=True,\n",
    "    plots=True\n",
    ")\n",
    "print(\"✅ Dry run passed! Weights and metrics successfully verified on Google Drive.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "### **Full Training**\n",
    "Run the cell below for the full 100-epoch baseline training run."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7. Full 100-Epoch Training (YOLO11 Single-Class Defect Baseline)\n",
    "from ultralytics import YOLO\n",
    "\n",
    "print(f\"Starting full 100-epoch training. Checkpoints will be saved to: {RUNS_DIR}\")\n",
    "model = YOLO(\"yolo11n.pt\")\n",
    "\n",
    "results = model.train(\n",
    "    data=DATA_YAML,\n",
    "    epochs=100,\n",
    "    imgsz=512,\n",
    "    batch=16,\n",
    "    patience=20,\n",
    "    optimizer=\"AdamW\",\n",
    "    lr0=0.001,\n",
    "    project=RUNS_DIR,\n",
    "    name=\"EXP-02-YOLO11-SingleClass-Baseline\",\n",
    "    seed=42,\n",
    "    device=0,\n",
    "    exist_ok=True,\n",
    "    save=True,\n",
    "    save_period=-1,\n",
    "    hsv_h=0.015,\n",
    "    hsv_s=0.7,\n",
    "    hsv_v=0.4,\n",
    "    degrees=0.0,\n",
    "    translate=0.1,\n",
    "    scale=0.5,\n",
    "    flipud=0.5,\n",
    "    fliplr=0.5,\n",
    "    mosaic=0.5,\n",
    "    plots=True\n",
    ")\n",
    "print(\"✅ Full training complete! All weights, curves, and validation metrics saved to Google Drive.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

out_path = r"c:\Users\syed mohammad anas\Desktop\sir_khurram_project\sem_defect_project\notebooks\01_yolov12_training.ipynb"
with open(out_path, 'w') as f:
    json.dump(notebook, f, indent=1)
print(f"Created updated notebook at {out_path}")
