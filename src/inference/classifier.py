import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ResNetClassifier:
    def __init__(self, model_path, num_classes=3):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ResNet weights not found at: {model_path}")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model Architecture (ResNet-50)
        self.model = models.resnet50(weights=None)
        num_ftrs = self.model.fc.in_features
        # The trained weights have 'fc.1.weight' and 'fc.1.bias', indicating a Sequential block
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
        # Load Weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transformation Pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, crop_image_rgb):
        # crop_image_rgb must be a PIL Image or numpy array in RGB
        if not isinstance(crop_image_rgb, Image.Image):
            pil_img = Image.fromarray(crop_image_rgb)
        else:
            pil_img = crop_image_rgb
            
        inp = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(inp)
            probs = torch.nn.functional.softmax(out, dim=1)
            cls_conf, cls_idx = torch.max(probs, 1)
            
        return cls_idx.item(), cls_conf.item()
