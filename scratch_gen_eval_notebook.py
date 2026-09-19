import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Phase 3: YOLO Detector Evaluation\n",
    "\n",
    "This notebook evaluates the YOLO model trained in Phase 2 on the unseen **test set**.\n",
    "It computes performance metrics, generates PR curves and confusion matrices, and performs a confidence sweep."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Setup and Google Drive Mount\n",
    "!pip install ultralytics pandas matplotlib pyyaml > /dev/null 2>&1\n",
    "import os\n",
    "import json\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from ultralytics import YOLO\n",
    "\n",
    "PROJECT_ROOT = '/content/drive/MyDrive/sem_defect_project'\n",
    "DATA_YAML = f\"{PROJECT_ROOT}/dataset_yolo_single_class/data.yaml\"\n",
    "\n",
    "# Dynamically identify which experiment name was used during training\n",
    "BEST_PT_V12 = f\"{PROJECT_ROOT}/runs/detect/EXP-02-YOLOv12-SingleClass-Baseline/weights/best.pt\"\n",
    "BEST_PT_V11 = f\"{PROJECT_ROOT}/runs/detect/EXP-02-YOLO11-SingleClass-Baseline/weights/best.pt\"\n",
    "\n",
    "EVAL_DIR = f\"{PROJECT_ROOT}/runs/detect/EVALUATION\"\n",
    "\n",
    "if os.path.exists('/content/drive/MyDrive'):\n",
    "    print(\"✅ Google Drive is mounted.\")\n",
    "else:\n",
    "    try:\n",
    "        from google.colab import drive\n",
    "        drive.mount('/content/drive')\n",
    "        print(\"✅ Google Drive successfully mounted.\")\n",
    "    except Exception as e:\n",
    "        raise RuntimeError(\"Please mount Drive using the Colab web UI.\") from e\n",
    "\n",
    "if os.path.exists(BEST_PT_V12):\n",
    "    BEST_PT = BEST_PT_V12\n",
    "elif os.path.exists(BEST_PT_V11):\n",
    "    BEST_PT = BEST_PT_V11\n",
    "else:\n",
    "    raise FileNotFoundError(\"Trained model not found! Did Phase 2 complete successfully?\")\n",
    "\n",
    "print(f\"✅ Using trained model checkpoint: {BEST_PT}\")\n",
    "os.makedirs(EVAL_DIR, exist_ok=True)\n",
    "print(\"✅ Setup complete.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Validation Set Evaluation\n",
    "print(\"--- Evaluating on VALIDATION set ---\")\n",
    "model = YOLO(BEST_PT)\n",
    "\n",
    "val_metrics = model.val(\n",
    "    data=DATA_YAML,\n",
    "    split='val',\n",
    "    project=EVAL_DIR,\n",
    "    name=\"val_baseline\",\n",
    "    plots=True\n",
    ")\n",
    "\n",
    "print(f\"\\nValidation mAP@50: {val_metrics.box.map50}\")\n",
    "print(f\"Validation mAP@50-95: {val_metrics.box.map}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Untouched TEST Set Evaluation\n",
    "print(\"\\n--- Evaluating on TEST set ---\")\n",
    "test_metrics = model.val(\n",
    "    data=DATA_YAML,\n",
    "    split='test',\n",
    "    project=EVAL_DIR,\n",
    "    name=\"test_baseline\",\n",
    "    plots=True,\n",
    "    save_json=True\n",
    ")\n",
    "\n",
    "print(f\"\\nTest mAP@50: {test_metrics.box.map50}\")\n",
    "print(f\"Test mAP@50-95: {test_metrics.box.map}\")\n",
    "\n",
    "# Extract Precision and Recall\n",
    "test_p = test_metrics.box.mp\n",
    "test_r = test_metrics.box.mr\n",
    "test_f1 = 2 * (test_p * test_r) / (test_p + test_r + 1e-6)\n",
    "\n",
    "print(f\"Test Precision: {test_p:.4f}\")\n",
    "print(f\"Test Recall: {test_r:.4f}\")\n",
    "print(f\"Test F1-Score: {test_f1:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Confidence Threshold Analysis\n",
    "print(\"\\n--- Running Confidence Threshold Sweep on TEST set ---\")\n",
    "thresholds = [0.15, 0.25, 0.40, 0.50, 0.60]\n",
    "sweep_results = []\n",
    "\n",
    "for conf in thresholds:\n",
    "    print(f\"\\nEvaluating at conf={conf}...\")\n",
    "    res = model.val(\n",
    "        data=DATA_YAML,\n",
    "        split='test',\n",
    "        conf=conf,\n",
    "        project=EVAL_DIR,\n",
    "        name=f\"test_conf_{conf}\",\n",
    "        plots=False,\n",
    "        verbose=False\n",
    "    )\n",
    "    \n",
    "    p = res.box.mp\n",
    "    r = res.box.mr\n",
    "    f1 = 2 * (p * r) / (p + r + 1e-6)\n",
    "    sweep_results.append({\n",
    "        'conf': conf,\n",
    "        'precision': float(p),\n",
    "        'recall': float(r),\n",
    "        'f1': float(f1),\n",
    "        'mAP50': float(res.box.map50)\n",
    "    })\n",
    "\n",
    "df_sweep = pd.DataFrame(sweep_results)\n",
    "print(\"\\nThreshold Sweep Results:\")\n",
    "print(df_sweep.to_string(index=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5. Generate Predictions for Visual Analysis (Error Analysis directories)\n",
    "import shutil\n",
    "\n",
    "print(\"\\n--- Running Inference & Error Analysis Dirs ---\")\n",
    "test_images_dir = f\"{PROJECT_ROOT}/dataset_yolo_single_class/test/images\"\n",
    "\n",
    "# Run standard inference to generate bounding box predictions on test images\n",
    "preds = model.predict(\n",
    "    source=test_images_dir,\n",
    "    conf=0.25, # good balance from sweep\n",
    "    project=EVAL_DIR,\n",
    "    name=\"test_predictions\",\n",
    "    save=True,\n",
    "    save_txt=True,\n",
    "    save_conf=True\n",
    ")\n",
    "print(f\"Predictions saved visually to {EVAL_DIR}/test_predictions\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 6. Export Final Summary JSON\n",
    "summary = {\n",
    "    'val': {\n",
    "        'precision': float(val_metrics.box.mp),\n",
    "        'recall': float(val_metrics.box.mr),\n",
    "        'mAP50': float(val_metrics.box.map50),\n",
    "        'mAP50-95': float(val_metrics.box.map)\n",
    "    },\n",
    "    'test': {\n",
    "        'precision': float(test_p),\n",
    "        'recall': float(test_r),\n",
    "        'f1': float(test_f1),\n",
    "        'mAP50': float(test_metrics.box.map50),\n",
    "        'mAP50-95': float(test_metrics.box.map)\n",
    "    },\n",
    "    'sweep': sweep_results\n",
    "}\n",
    "\n",
    "summary_path = f\"{EVAL_DIR}/eval_summary.json\"\n",
    "with open(summary_path, 'w') as f:\n",
    "    json.dump(summary, f, indent=4)\n",
    "\n",
    "print(f\"\\n✅ Evaluation completed. Summary saved to {summary_path}\")\n",
    "print(\"\\n================ FINAL RESULTS ================\")\n",
    "print(json.dumps(summary, indent=4))\n",
    "print(\"\\nPlease copy these results back or review them locally to proceed with Phase 3 Report generation.\")"
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

out_path = r"c:\Users\syed mohammad anas\Desktop\sir_khurram_project\sem_defect_project\notebooks\02_yolov12_evaluation.ipynb"
with open(out_path, 'w') as f:
    json.dump(notebook, f, indent=1)
print(f"Created notebook at {out_path}")
