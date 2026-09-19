import cv2
import numpy as np

class DefectPipeline:
    def __init__(self, detector, classifier, class_names, padding_factor):
        self.detector = detector
        self.classifier = classifier
        self.class_names = class_names
        self.padding_factor = padding_factor

    def run(self, img_array_bgr):
        img_rgb = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_array_bgr.shape[:2]
        
        # 1. Detect
        results = self.detector.detect(img_array_bgr)
        
        predictions = []
        
        # 2. Classify each detected box
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det_conf = box.conf[0].item()
            
            # Crop & Pad
            w, h = x2 - x1, y2 - y1
            px, py = int(w * self.padding_factor), int(h * self.padding_factor)
            cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            cx2, cy2 = min(img_w, int(x2 + px)), min(img_h, int(y2 + py))
            
            crop = img_rgb[cy1:cy2, cx1:cx2]
            if crop.size == 0: 
                continue
                
            # Classify
            cls_idx, cls_conf = self.classifier.predict(crop)
            subtype = self.class_names[cls_idx]
            
            predictions.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'det_conf': det_conf,
                'class_idx': cls_idx,
                'subtype': subtype,
                'cls_conf': cls_conf
            })
            
        return predictions

    def annotate_image(self, img_array_bgr, predictions):
        annotated_img = img_array_bgr.copy()
        
        # Color mapping for classes (BGR)
        colors = {
            'Inclusion-Particle': (0, 165, 255),    # Orange
            'Porosity': (0, 255, 0),                # Green
            'Tear-Delamination': (0, 0, 255)        # Red
        }
        
        for i, pred in enumerate(predictions):
            x1, y1, x2, y2 = pred['box']
            subtype = pred['subtype']
            color = colors.get(subtype, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"#{i+1} {subtype}"
            conf_label = f"D:{pred['det_conf']:.2f} C:{pred['cls_conf']:.2f}"
            
            # Draw labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Top label (Defect Type)
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(annotated_img, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
            cv2.putText(annotated_img, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
            
            # Bottom label (Confidences)
            (cw, ch), _ = cv2.getTextSize(conf_label, font, font_scale, thickness)
            cv2.rectangle(annotated_img, (x1, y2), (x1 + cw, y2 + ch + 10), color, -1)
            cv2.putText(annotated_img, conf_label, (x1, y2 + ch + 5), font, font_scale, (0, 0, 0), thickness)
            
        return cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
