import os

# Project root based on the script location
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model Paths
YOLO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'models', 'yolo_exp10_best.pt')
RESNET_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'models', 'resnet_exp11_best.pt')

# Inference Parameters
DETECTION_CONF_THRESH = 0.15
DETECTION_IOU_THRESH = 0.50
CROP_PADDING_FACTOR = 0.10

# Classes mapping
CLASS_NAMES = ['Inclusion-Particle', 'Porosity', 'Tear-Delamination']
