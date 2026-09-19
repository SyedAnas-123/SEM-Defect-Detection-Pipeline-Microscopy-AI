# SEM Defect Detection Pipeline 🔬

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)

An advanced, two-stage Deep Learning pipeline designed for the automated detection and classification of manufacturing defects in Scanning Electron Microscope (SEM) imagery. 

---

## 🌟 Research Overview & Discovery
In materials science and manufacturing, identifying microscopic defects (like porosity, inclusions, or delamination) using SEM images is highly manual, time-consuming, and prone to human error. 

**Our Discovery & Approach:**
Instead of relying on a single end-to-end model which often suffers from compounding errors on complex microscopic backgrounds, we discovered that a **Decoupled Two-Stage Architecture** yields significantly higher reliability:
1. **Stage 1 (Localization):** We utilize a fine-tuned **YOLO11m** model to scan the entire SEM image and draw bounding boxes around any anomalous regions.
2. **Stage 2 (Classification):** We extract (crop) these specific regions and pass them through a robust **ResNet-50** classifier to precisely identify the defect subtype (e.g., *Inclusion-Particle*, *Porosity*, *Tear-Delamination*).

This decoupled approach ensures that the classifier only focuses on the defect itself, ignoring irrelevant microscopic background textures.

---

## 📂 Project Structure
```text
sem_defect_project/
│
├── app.py                      # Main Streamlit Interactive Web Application
├── requirements.txt            # All dependencies and strict versions for Windows compatibility
├── README.md                   # Project documentation
│
├── models/                     # Trained Model Weights
│   ├── yolo_exp10_best.pt      # Stage 1: YOLO Detection Weights
│   └── resnet_exp11_best.pt    # Stage 2: ResNet Classification Weights
│
├── notebooks/                  # Jupyter Notebooks for EDA, Training, and Evaluation
│   └── FINAL_ANALYSIS_OF_THE_PIPELINE.ipynb
│
└── src/                        # Core Inference Logic
    ├── config.py               # Hyperparameters and threshold settings
    └── inference/
        ├── detector.py         # YOLO integration
        ├── classifier.py       # ResNet integration
        └── pipeline.py         # E2E logic connecting Stage 1 and Stage 2
```

---

## 🚀 How to Run the Project (For Reviewers & Examiners)

Follow these simple steps to download the requirements and run the interactive web application on your local machine.

### 1. Set Up Virtual Environment & Install Dependencies
It is highly recommended to use a Python virtual environment to avoid conflicts. Open your terminal in the root of this project and run:

**For Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**For macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(Note: The `requirements.txt` specifically uses PyTorch `2.2.2` and NumPy `<2.0` to ensure maximum compatibility and avoid Windows DLL blocking issues).*

### 2. Run the Interactive Web App
Once the requirements are installed, launch the Streamlit app:
```bash
python -m streamlit run app.py
```
The application will automatically open in your web browser (usually at `http://localhost:8501`). 

### 3. Usage
- Go to the **Live Inference** tab.
- Upload any SEM image (`.jpg` or `.png`).
- Click **Analyze Image**.
- The system will run Stage 1 (YOLO) and Stage 2 (ResNet) sequentially and display the image with bounding boxes, predicted defect types, and confidence scores.

---

## 📊 Model Performance Highlights
- **Detector (YOLO11m):** Optimized for high recall to ensure no defect is missed.
- **Classifier (ResNet-50):** Highly accurate subtype classification on isolated defect crops.
- Detailed metrics, Precision-Recall curves, and Confusion Matrices are available in the **Model Performance** tab within the web app and the Jupyter Notebooks.

---
*Developed as a dedicated research project for automated defect analysis in microscopy.*
