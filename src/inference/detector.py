import os
from ultralytics import YOLO
import sys

# Ensure ultralytics doesn't output too much to stdout in the web app
class YOLODetector:
    def __init__(self, model_path, conf_thresh, iou_thresh):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO weights not found at: {model_path}")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect(self, img_path_or_array):
        # Run inference
        results = self.model(img_path_or_array, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)
        return results[0]
