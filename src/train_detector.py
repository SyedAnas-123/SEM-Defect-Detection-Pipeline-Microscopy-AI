import os
import yaml
import argparse
from ultralytics import YOLO

def main(dry_run=False):
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Dataset path
    data_yaml = os.path.join(config['paths']['yolo_single_class_dataset'], 'data.yaml')
    
    # Model configuration
    model_name = "yolo11n.pt" # Fallback to 11 if 12 isn't fully pulled yet, but ultralytics auto-downloads
    
    # Actually let's try to load yolov12n.pt (if ultralytics is updated enough)
    try:
        model = YOLO("yolov12n.pt")
    except Exception:
        print("YOLOv12 weights not found or unsupported in this ultralytics version. Falling back to YOLO11.")
        model = YOLO("yolo11n.pt")

    # Training parameters
    yolo_cfg = config['yolo']
    epochs = 2 if dry_run else yolo_cfg['epochs']
    
    # Run directory
    project_dir = os.path.join(config['paths']['runs_dir'], 'detect')
    name = "yolov12_single_class_dryrun" if dry_run else "EXP-02-YOLOv12-SingleClass-Baseline"
    
    print(f"Starting training pipeline...")
    print(f"Dry Run: {dry_run}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Project Dir: {project_dir}/{name}")

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=yolo_cfg['img_size'],
        batch=yolo_cfg['batch_size'],
        patience=yolo_cfg['patience'],
        optimizer=yolo_cfg['optimizer'],
        lr0=yolo_cfg['lr0'],
        project=project_dir,
        name=name,
        seed=config['seed'],
        device=0, # Auto-select CUDA device 0
        exist_ok=True, # Allow overwriting dry-run if necessary
        # Conservative online augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0
    )
    
    print("Training complete.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run a short 2-epoch sanity check')
    args = parser.parse_args()
    main(dry_run=args.dry_run)
