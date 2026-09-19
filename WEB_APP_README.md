# SEM Defect Detection Web Application

This repository contains the interactive research demonstration for the Scanning Electron Microscope (SEM) Defect Detection system.

## Architecture
The system consists of two sequential stages, decoupling detection and classification to avoid compounding errors in complex microscopic datasets:
1. **Stage 1 (Localization):** YOLO11m detects regions of interest (anomalies).
2. **Stage 2 (Classification):** ResNet-50 determines the exact subtype of each cropped region.

Their performances are reported independently.

## Directory Structure
```
sem_defect_project/
│
├── app.py                      # Main Streamlit Application
├── WEB_APP_README.md           # This file
│
├── models/                     # Model Checkpoints (MUST BE PLACED HERE)
│   ├── yolo_exp10_best.pt
│   └── resnet_exp11_best.pt
│
└── src/                        # Inference Logic
    ├── config.py               # Paths and Hyperparameters
    └── inference/
        ├── detector.py         # YOLO Wrapper
        ├── classifier.py       # ResNet Wrapper
        └── pipeline.py         # End-to-End Orchestrator
```

## Setup & Requirements

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install streamlit ultralytics torch torchvision opencv-python pandas numpy Pillow
   ```
3. **Important:** Place the YOLO (`best.pt`) and ResNet (`best.pt`) model files in the `models/` directory exactly as named in the directory structure above.

## Running the Web App

Launch the application by running the following command in your terminal from the `sem_defect_project` root directory:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser (usually at `http://localhost:8501`).

## Usage
1. Open the "Live Inference" tab.
2. Upload a `.jpg` or `.png` SEM image.
3. Click "Analyze Image".
4. The system will process the image through both stages and output the annotated image alongside detailed confidence metrics.
