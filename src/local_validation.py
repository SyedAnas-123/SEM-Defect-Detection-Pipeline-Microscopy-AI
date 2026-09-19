import os
import yaml

def check_paths():
    print("--- LOCAL VALIDATION START ---")
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    print(f"Loading config from: {config_path}")
    
    if not os.path.exists(config_path):
        print("ERROR: config.yaml not found!")
        return False
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("Config loaded successfully.")
    
    yolo_dataset = config['paths']['yolo_single_class_dataset']
    data_yaml = os.path.join(yolo_dataset, 'data.yaml')
    print(f"Checking YOLO derived dataset at: {data_yaml}")
    
    if not os.path.exists(data_yaml):
        print("ERROR: YOLO data.yaml not found!")
        return False
        
    with open(data_yaml, 'r') as f:
        yolo_config = yaml.safe_load(f)
        
    print("YOLO data.yaml loaded successfully.")
    
    if yolo_config.get('nc') != 1:
        print(f"ERROR: Expected nc: 1, got {yolo_config.get('nc')}")
        return False
        
    print("Class mapping verified: nc = 1")
    
    # Check splits (directories are named train, valid, test regardless of YAML keys)
    splits = ['train', 'valid', 'test']
    for split in splits:
        img_dir = os.path.join(yolo_dataset, split, 'images')
        lbl_dir = os.path.join(yolo_dataset, split, 'labels')
        
        if not os.path.exists(img_dir):
            print(f"WARNING: Image dir missing for split {split}: {img_dir}")
        else:
            imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
            lbls = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')] if os.path.exists(lbl_dir) else []
            print(f"Split '{split}': Found {len(imgs)} images and {len(lbls)} labels.")
            
            # Label parsing check
            if len(lbls) > 0:
                sample_lbl = os.path.join(lbl_dir, lbls[0])
                with open(sample_lbl, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        parts = lines[0].strip().split()
                        if parts[0] != '0':
                            print(f"ERROR: Found non-zero class ID in {sample_lbl}: {parts[0]}")
                            return False
                print(f"Label parsing passed for {split} split (Sample ID 0).")
                
    print("--- LOCAL VALIDATION PASSED ---")
    return True

if __name__ == "__main__":
    check_paths()
