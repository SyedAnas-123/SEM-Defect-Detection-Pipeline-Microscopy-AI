import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np

class TwoStagePipeline:
    def __init__(self, yolo_weights_path, resnet_weights_path, device=None, conf_thresh=0.25, padding_factor=0.10):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.padding_factor = padding_factor
        
        # Load YOLO
        self.detector = YOLO(yolo_weights_path)
        self.detector.to(self.device)
        
        # Load ResNet-50
        self.classifier = models.resnet50(weights=None)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, 3)
        )
        self.classifier.load_state_dict(torch.load(resnet_weights_path, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()
        
        self.class_names = ['Inclusion-Particle', 'Porosity', 'Tear-Delamination']
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        # 1. Detection
        results = self.detector.predict(image_path, conf=self.conf_thresh, verbose=False)
        boxes = results[0].boxes
        
        original_img = Image.open(image_path).convert("RGB")
        img_width, img_height = original_img.size
        
        predictions = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            yolo_conf = float(box.conf[0].cpu().numpy())
            
            # Apply 10% contextual padding
            w = x2 - x1
            h = y2 - y1
            pad_x = w * self.padding_factor
            pad_y = h * self.padding_factor
            
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(img_width, x2 + pad_x)
            crop_y2 = min(img_height, y2 + pad_y)
            
            # 2. Crop
            crop_img = original_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            
            # 3. Classification
            input_tensor = self.preprocess(crop_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.classifier(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                resnet_conf, preds = torch.max(probs, 1)
                
            pred_class = self.class_names[preds.item()]
            resnet_conf = float(resnet_conf.item())
            
            predictions.append({
                'bbox': [x1, y1, x2, y2],
                'yolo_conf': yolo_conf,
                'pred_class': pred_class,
                'resnet_conf': resnet_conf
            })
            
        return predictions, original_img

    def draw_predictions(self, image, predictions, output_path=None):
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            
        colors = {
            'Inclusion-Particle': 'red',
            'Porosity': 'blue',
            'Tear-Delamination': 'green'
        }
        
        for p in predictions:
            x1, y1, x2, y2 = p['bbox']
            cls_name = p['pred_class']
            y_conf = p['yolo_conf']
            r_conf = p['resnet_conf']
            
            color = colors.get(cls_name, 'yellow')
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            label = f"{cls_name}\nYOLO: {y_conf:.2f}\nResNet: {r_conf:.2f}"
            
            # Text background
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(text_bbox, fill="black")
            draw.text((x1, y1), label, fill="white", font=font)
            
        if output_path:
            image.save(output_path)
            
        return image
