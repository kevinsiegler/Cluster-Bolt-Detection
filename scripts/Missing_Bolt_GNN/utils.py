r"""
c:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Missing_Bolt_GNN\utils.py
"""
import os
import numpy as np
import yaml
import shutil

# --- CONFIGURATION ---
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load config globally
CONFIG = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_run_directories(config, mode):
    """
    Creates the folder structure based on the mode (training/inference/evaluation).
    mode: 'training', 'inference', or 'evaluation'
    """
    run_name = config[mode].get('run_name') if mode != 'evaluation' else config['evaluation'].get('inference_run')
    
    # Structure: output_root / mode / run_name
    base_dir = os.path.join(config['paths']['output_root'], mode, run_name)
    ensure_dir(base_dir)
    
    # Save a copy of the config for reproducibility (only for training and inference)
    if mode in ['training', 'inference']:
        shutil.copy(os.path.join(os.path.dirname(__file__), "config.yaml"), os.path.join(base_dir, "config_copy.yaml"))
    
    return base_dir

def load_yolo_labels(path):
    """
    Loads YOLO labels. Returns Nx5 numpy array: [class, x, y, w, h].
    """
    if not os.path.exists(path):
        return np.empty((0, 5))
    
    labels = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # class, x, y, w, h
                labels.append([float(x) for x in parts[:5]])
    
    return np.array(labels) if labels else np.empty((0, 5))

def save_yolo_labels(path, labels):
    """
    Saves Nx5 numpy array to file.
    """
    with open(path, 'w') as f:
        for row in labels:
            cls = int(row[0])
            f.write(f"{cls} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")