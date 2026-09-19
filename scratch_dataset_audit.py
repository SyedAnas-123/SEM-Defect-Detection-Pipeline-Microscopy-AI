import os
import glob
import numpy as np

def bbox_iou(box1, box2):
    # box format: [x, y, w, h] normalized
    x1_min, y1_min = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x1_max, y1_max = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x2_min, y2_min = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_max, y2_max = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def analyze_split(split_dir):
    label_files = glob.glob(os.path.join(split_dir, 'labels', '*.txt'))
    
    total_boxes = 0
    areas = []
    widths = []
    heights = []
    
    bins = {'<0.1%': 0, '0.1-0.25%': 0, '0.25-0.5%': 0, '0.5-1%': 0, '>1%': 0}
    
    duplicates = 0
    overlapping = 0
    touching_boundary = 0
    
    for lbl_path in label_files:
        boxes = []
        if not os.path.exists(lbl_path): continue
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x, y, w, h = map(float, parts[1:5])
                    boxes.append([x, y, w, h])
                    
        total_boxes += len(boxes)
        for i, box in enumerate(boxes):
            x, y, w, h = box
            area_pct = (w * h) * 100
            areas.append(area_pct)
            widths.append(w * 100)
            heights.append(h * 100)
            
            if area_pct < 0.1: bins['<0.1%'] += 1
            elif area_pct < 0.25: bins['0.1-0.25%'] += 1
            elif area_pct < 0.5: bins['0.25-0.5%'] += 1
            elif area_pct < 1.0: bins['0.5-1%'] += 1
            else: bins['>1%'] += 1
                
            x_min, y_min = x - w/2, y - h/2
            x_max, y_max = x + w/2, y + h/2
            
            if x_min < 0.01 or y_min < 0.01 or x_max > 0.99 or y_max > 0.99:
                touching_boundary += 1
                
            for j, other_box in enumerate(boxes):
                if i >= j: continue
                iou = bbox_iou(box, other_box)
                if iou > 0.9: duplicates += 1
                elif iou > 0.5: overlapping += 1
                
    return {
        'total': total_boxes,
        'areas': areas,
        'widths': widths,
        'heights': heights,
        'bins': bins,
        'duplicates': duplicates,
        'overlapping': overlapping,
        'touching_boundary': touching_boundary
    }

base_dir = r"c:\Users\syed mohammad anas\Desktop\sir_khurram_project\sem_defect_project\datasets\dataset_yolo_single_class"

splits = ['train', 'valid', 'test']
results = {}

for split in splits:
    split_path = os.path.join(base_dir, split)
    print(f"Analyzing {split}...")
    results[split] = analyze_split(split_path)

print("\\n=== DATASET AUDIT RESULTS ===")
for split in splits:
    r = results[split]
    tot = r['total']
    if tot == 0: continue
    print(f"\\n--- {split.upper()} SET ---")
    print(f"Total Annotations: {tot}")
    areas = r['areas']
    print(f"Area: Mean={np.mean(areas):.3f}%, Median={np.median(areas):.3f}%, Min={np.min(areas):.3f}%, Max={np.max(areas):.3f}%")
    print(f"Widths: Mean={np.mean(r['widths']):.2f}%, Heights: Mean={np.mean(r['heights']):.2f}%")
    print("Bins:")
    for k, v in r['bins'].items():
        print(f"  {k}: {v} ({v/tot*100:.1f}%)")
    print(f"Duplicates (IoU>0.9): {r['duplicates']} ({r['duplicates']/tot*100:.1f}%)")
    print(f"Overlapping (0.5<IoU<0.9): {r['overlapping']} ({r['overlapping']/tot*100:.1f}%)")
    print(f"Touching Boundary: {r['touching_boundary']} ({r['touching_boundary']/tot*100:.1f}%)")
