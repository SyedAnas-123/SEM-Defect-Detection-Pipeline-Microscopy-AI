import json
import os

input_nb = 'c:/Users/syed mohammad anas/Desktop/sir_khurram_project/sem_defect_project/notebooks/05_end_to_end_evaluation.ipynb'
output_nb = 'c:/Users/syed mohammad anas/Desktop/sir_khurram_project/sem_defect_project/notebooks/08_final_evaluation_V2.ipynb'

with open(input_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The first 8 cells of 05_end_to_end_evaluation set up the pipeline.
# We will keep cells 0 to 9, then modify cell 10 (the execution cell).

new_cells = []

# Prepend Part A (YOLO standalone evaluation)
part_a_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## PART A: Standalone YOLO Detector Evaluation (Test Set)\n",
        "Evaluating the EXP-10 YOLOv12m model on the test set using the optimal threshold of 0.15 found during validation sweeps."
    ]
}

part_a_code = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "from ultralytics import YOLO\n",
        "import os\n",
        "os.chdir('/content/drive/MyDrive/sem_defect_project')\n",
        "\n",
        "# Load the best weights from the EXP-10 training run\n",
        "model = YOLO('runs/detect/EXP-10-YOLOv12m-640-200/weights/best.pt')\n",
        "\n",
        "print(\"Evaluating EXP-10 on the untouched TEST set...\")\n",
        "\n",
        "# Run evaluation on the TEST split using the locked optimal thresholds\n",
        "metrics = model.val(\n",
        "    data='dataset_yolo_single_class/data.yaml',\n",
        "    split='test',\n",
        "    conf=0.15,\n",
        "    iou=0.50,\n",
        "    plots=True\n",
        ")\n",
        "\n",
        "print(\"\\n=== PART A: YOLO TEST SET RESULTS ===\")\n",
        "precision = metrics.results_dict['metrics/precision(B)']\n",
        "recall = metrics.results_dict['metrics/recall(B)']\n",
        "map50 = metrics.results_dict['metrics/mAP50(B)']\n",
        "map50_95 = metrics.results_dict['metrics/mAP50-95(B)']\n",
        "f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0\n",
        "\n",
        "print(f\"Precision: {precision:.4f}\")\n",
        "print(f\"Recall:    {recall:.4f}\")\n",
        "print(f\"F1-Score:  {f1:.4f}\")\n",
        "print(f\"mAP50:     {map50:.4f}\")\n",
        "print(f\"mAP50-95:  {map50_95:.4f}\")\n"
    ]
}

new_cells.extend([part_a_markdown, part_a_code])

part_b_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## PART B: End-to-End Evaluation (Two-Stage Pipeline)\n",
        "Running the full pipeline using the embedded classes."
    ]
}
new_cells.append(part_b_markdown)

# Copy the exact pipeline cells from 05 notebook
for i, cell in enumerate(nb['cells']):
    # In notebook 05, cell 0 is drive mount, cell 1 is pip, cell 2 is unzip, cell 3 is imports, cells 4-9 are pipeline code
    if i < 10:
        new_cells.append(cell)

# Now write the execution cell for E2E
exec_code = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# Check if the extracted directory exists\n",
        "import os\n",
        "import glob\n",
        "\n",
        "# The exact absolute paths for the test data\n",
        "test_img_dir = '/content/test_3class/test/images'\n",
        "test_label_dir = '/content/test_3class/test/labels'\n",
        "output_dir = '/content/drive/MyDrive/sem_defect_project/runs/end_to_end_EXP10'\n",
        "\n",
        "if not os.path.exists(test_img_dir):\n",
        "    print(f\"Directory {test_img_dir} does not exist! Please ensure you run the unzip cell above.\")\n",
        "else:\n",
        "    print(f\"Found images directory: {test_img_dir}\")\n",
        "    print(f\"Number of images: {len(glob.glob(os.path.join(test_img_dir, '*.jpg')))}\")\n",
        "\n",
        "# HARDCODED TO THE OPTIMAL SETTINGS\n",
        "yolo_conf_thresh = 0.15\n",
        "iou_thresh = 0.50\n",
        "padding_factor = 0.10\n",
        "\n",
        "# Paths for models \n",
        "yolo_weights = '/content/drive/MyDrive/sem_defect_project/runs/detect/EXP-10-YOLOv12m-640-200/weights/best.pt'\n",
        "resnet_weights = '/content/drive/MyDrive/sem_defect_project/runs/classify/EXP-03-ResNet50/best.pt'\n",
        "\n",
        "if not os.path.exists(yolo_weights):\n",
        "    print(f\"Error: YOLO weights not found at {yolo_weights}\")\n",
        "if not os.path.exists(resnet_weights):\n",
        "    print(f\"Error: ResNet weights not found at {resnet_weights}\")\n",
        "\n",
        "print(\"Initializing Two-Stage Pipeline...\")\n",
        "pipeline = TwoStagePipeline(\n",
        "    yolo_weights_path=yolo_weights,\n",
        "    resnet_weights_path=resnet_weights,\n",
        "    conf_thresh=yolo_conf_thresh,\n",
        "    padding_factor=padding_factor\n",
        ")\n",
        "\n",
        "print(f\"Starting Evaluation on {test_img_dir}...\")\n",
        "evaluate_e2e(test_img_dir, test_label_dir, output_dir, pipeline, iou_thresh=iou_thresh)\n",
        "\n",
        "print(f\"Evaluation Complete. Results saved to {output_dir}/\")\n"
    ]
}

new_cells.append(exec_code)

nb['cells'] = new_cells

with open(output_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print(f"Notebook generated successfully at {output_nb}")
