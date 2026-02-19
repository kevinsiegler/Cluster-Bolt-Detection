r"""
C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\preprocess.py
"""
import os
import numpy as np
from tqdm import tqdm
from utils import load_config, load_yolo_labels, ensure_dir

def preprocess():
    # Load config automatically from script directory
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(config_path)
    
    # Setup Output Dirs
    out_root = cfg['paths']['output_root']
    train_out = os.path.join(out_root, "preprocessing", "train")
    val_out_input = os.path.join(out_root, "preprocessing", "val_input") # Only existing bolts
    val_out_gt = os.path.join(out_root, "preprocessing", "val_gt")       # Only missing bolts (for eval)
    
    ensure_dir(train_out)
    ensure_dir(val_out_input)
    ensure_dir(val_out_gt)
    
    print("--- Preprocessing Training Data ---")
    train_dir = cfg['paths']['train_labels']
    files = [f for f in os.listdir(train_dir) if f.endswith('.txt')]
    
    processed_train_count = 0
    for f in tqdm(files):
        labels = load_yolo_labels(os.path.join(train_dir, f))
        if len(labels) == 0: continue
        
        # Training Logic: Treat Class 1 (missing) as Class 0 (present)
        # We want to learn the COMPLETE layout.
        # We need x, y, w, h for later steps
        points = labels[:, 1:5] # x, y, w, h
        
        # Save as numpy binary for fast loading
        np.save(os.path.join(train_out, f.replace('.txt', '.npy')), points)
        processed_train_count += 1
        
    print(f"Processed {processed_train_count} training samples.")
    
    print("\n--- Preprocessing Validation Data ---")
    val_dir = cfg['paths']['val_labels']
    files = [f for f in os.listdir(val_dir) if f.endswith('.txt')]
    
    for f in tqdm(files):
        labels = load_yolo_labels(os.path.join(val_dir, f))
        if len(labels) == 0: continue
        
        # Validation Logic: Split into Input (Present) and GT (Missing)
        
        # Input: Class 0
        input_mask = labels[:, 0] == 0
        input_points = labels[input_mask][:, 1:5] # x, y, w, h
        
        # GT: Class 1
        gt_mask = labels[:, 0] == 1
        gt_points = labels[gt_mask][:, 1:5] # x, y, w, h
        
        # Save
        np.save(os.path.join(val_out_input, f.replace('.txt', '.npy')), input_points)
        np.save(os.path.join(val_out_gt, f.replace('.txt', '.npy')), gt_points) # Also save w,h for evaluation
        
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess()
