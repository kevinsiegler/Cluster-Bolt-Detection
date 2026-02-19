r"""
C:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Cluster\evaluate.py
"""
import os
import numpy as np
from scipy.spatial.distance import cdist
from utils import load_config, load_yolo_labels, ensure_dir

def evaluate():
    # Load config automatically from script directory
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(config_path)
    run_name = cfg['inference'].get('run_name', 'default_run')
    
    pred_dir = os.path.join(cfg['paths']['output_root'], "inference", run_name)
    gt_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_gt") # Contains only missing bolts
    
    if not os.path.exists(pred_dir):
        print("Prediction directory not found.")
        return

    files = [f for f in os.listdir(gt_dir) if f.endswith('.npy')]
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    total_dist_error = 0
    
    dist_thresh = cfg['evaluation']['dist_threshold']
    
    print(f"Evaluating {len(files)} files against Ground Truth...")
    
    for filename in files:
        # Load GT Missing (Points)
        gt_pts = np.load(os.path.join(gt_dir, filename))
        
        # Fix: GT now has 4 columns (x,y,w,h) from preprocessing, but we evaluate on x,y
        if gt_pts.ndim == 2 and gt_pts.shape[1] > 2:
            gt_pts = gt_pts[:, :2]
        
        # Load Prediction (YOLO txt)
        pred_path = os.path.join(pred_dir, filename.replace('.npy', '.txt'))
        pred_labels = load_yolo_labels(pred_path)
        
        # Filter for Class 1 (Predicted Missing)
        if len(pred_labels) > 0:
            pred_pts = pred_labels[pred_labels[:, 0] == 1][:, 1:3]
        else:
            pred_pts = np.empty((0, 2))
            
        # Matching Logic
        n_gt = len(gt_pts)
        n_pred = len(pred_pts)
        
        if n_gt == 0 and n_pred == 0:
            continue
            
        if n_gt == 0:
            fp_total += n_pred
            continue
            
        if n_pred == 0:
            fn_total += n_gt
            continue
            
        # Distance Matrix
        dists = cdist(gt_pts, pred_pts)
        
        # Greedy Matching
        matched_gt = set()
        matched_pred = set()
        local_tp = 0
        local_dist_error = 0
        
        # Iterate GTs, find closest Pred
        for i in range(n_gt):
            best_match_idx = np.argmin(dists[i])
            min_dist = dists[i][best_match_idx]
            
            if min_dist < dist_thresh:
                # Check if this pred is already used (simple greedy)
                # Ideally we use Hungarian algorithm, but greedy is fine for sparse points
                if best_match_idx not in matched_pred:
                    matched_pred.add(best_match_idx)
                    matched_gt.add(i)
                    local_tp += 1
                    local_dist_error += min_dist
        
        local_fn = n_gt - len(matched_gt)
        local_fp = n_pred - len(matched_pred)
        
        tp_total += local_tp
        fn_total += local_fn
        fp_total += local_fp
        total_dist_error += local_dist_error

    # Metrics
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_error = total_dist_error / tp_total if tp_total > 0 else 0
    
    print("\n" + "="*30)
    print(f"EVALUATION REPORT: {run_name}")
    print("="*30)
    print(f"Correct Predictions (TP): {tp_total}")
    print(f"Wrong Predictions   (FP): {fp_total}")
    print(f"Missed Bolts        (FN): {fn_total}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate()
