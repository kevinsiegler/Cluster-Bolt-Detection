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
    gt_existing_dir = os.path.join(cfg['paths']['output_root'], "preprocessing", "val_input") # Contains existing bolts (GT)
    inference_input_dir = cfg['paths']['inference_input_dir'] # YOLO Input (Class 0)
    
    if not os.path.exists(pred_dir):
        print(f"Prediction directory not found: {pred_dir}")
        return

    # Evaluate only files present in the prediction directory (subset)
    files = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    total_dist_error = 0
    fn_due_to_yolo_occlusion = 0 # New metric: Missing bolts missed because YOLO said "present"
    fp_on_existing_bolt = 0 # New metric: Predicted "missing" but actually "existing" (YOLO FN -> Cluster FP)
    
    dist_thresh = cfg['evaluation']['dist_threshold']
    
    print(f"Evaluating {len(files)} files (subset) against Ground Truth...")
    
    for filename in files:
        # filename is .txt from pred_dir
        
        # Load GT Missing (Points)
        gt_filename = filename.replace('.txt', '.npy')
        gt_path = os.path.join(gt_dir, gt_filename)
        
        if not os.path.exists(gt_path):
            continue
            
        gt_pts = np.load(gt_path)
        
        # Fix: GT now has 4 columns (x,y,w,h) from preprocessing, but we evaluate on x,y
        if gt_pts.ndim == 2 and gt_pts.shape[1] > 2:
            gt_pts = gt_pts[:, :2]
        
        # Load GT Existing (Points) to check for FPs on existing bolts
        gt_existing_path = os.path.join(gt_existing_dir, gt_filename)
        gt_existing_pts = np.empty((0, 2))
        if os.path.exists(gt_existing_path):
            gt_existing_data = np.load(gt_existing_path)
            # Fix: GT usually has 4 columns (x,y,w,h)
            if gt_existing_data.ndim == 2 and gt_existing_data.shape[1] >= 2:
                gt_existing_pts = gt_existing_data[:, :2]

        # Load Prediction (YOLO txt)
        pred_path = os.path.join(pred_dir, filename)
        pred_labels = load_yolo_labels(pred_path)
        
        # Filter for Class 1 (Predicted Missing)
        if len(pred_labels) > 0:
            pred_pts = pred_labels[pred_labels[:, 0] == 1][:, 1:3]
        else:
            pred_pts = np.empty((0, 2))
            
        # Load YOLO Input (Class 0) to check for occlusions
        yolo_input_path = os.path.join(inference_input_dir, filename)
        yolo_labels = load_yolo_labels(yolo_input_path)
        yolo_pts = np.empty((0, 2))
        if len(yolo_labels) > 0:
            yolo_pts = yolo_labels[yolo_labels[:, 0] == 0][:, 1:3]

        # Matching Logic
        n_gt = len(gt_pts)
        n_pred = len(pred_pts)
        
        if n_gt == 0 and n_pred == 0:
            continue
            
        if n_pred == 0:
            fn_total += n_gt
            # Check if these FNs were caused by YOLO false positives
            if len(yolo_pts) > 0:
                dists_yolo = cdist(gt_pts, yolo_pts)
                min_dists_yolo = np.min(dists_yolo, axis=1)
                # If a GT missing bolt is close to a YOLO present bolt, Cluster couldn't predict it
                fn_due_to_yolo_occlusion += np.sum(min_dists_yolo < dist_thresh)
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
        
        # Analyze False Positives (Predictions that are NOT missing bolts)
        unmatched_pred_indices = [i for i in range(n_pred) if i not in matched_pred]
        
        if len(unmatched_pred_indices) > 0 and len(gt_existing_pts) > 0:
            unmatched_preds = pred_pts[unmatched_pred_indices]
            dists_existing = cdist(unmatched_preds, gt_existing_pts)
            # If a predicted missing bolt is close to an EXISTING bolt, count it
            fp_on_existing_bolt += np.sum(np.min(dists_existing, axis=1) < dist_thresh)

        # Analyze False Negatives
        unmatched_gt_indices = [i for i in range(n_gt) if i not in matched_gt]
        local_fn = len(unmatched_gt_indices)
        
        if local_fn > 0 and len(yolo_pts) > 0:
            unmatched_gt_pts = gt_pts[unmatched_gt_indices]
            dists_yolo = cdist(unmatched_gt_pts, yolo_pts)
            fn_due_to_yolo_occlusion += np.sum(np.min(dists_yolo, axis=1) < dist_thresh)

        local_fp = n_pred - len(matched_pred)
        
        tp_total += local_tp
        fn_total += local_fn
        fp_total += local_fp
        total_dist_error += local_dist_error

    # --- Calculate Pure Cluster Performance (Hypothetical Perfect YOLO) ---
    # Logic:
    # 1. FN due to YOLO occlusion: If YOLO hadn't reported a ghost bolt, Cluster would have likely found the missing bolt (FN -> TP).
    # 2. FP on existing bolt: If YOLO hadn't missed the bolt, Cluster wouldn't have filled the gap (FP -> TN/Removed).
    
    pure_tp = tp_total + fn_due_to_yolo_occlusion
    pure_fp = max(0, fp_total - fp_on_existing_bolt)
    pure_fn = max(0, fn_total - fn_due_to_yolo_occlusion)

    # Metrics
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_error = total_dist_error / tp_total if tp_total > 0 else 0
    
    # Pure Metrics
    pure_precision = pure_tp / (pure_tp + pure_fp) if (pure_tp + pure_fp) > 0 else 0
    pure_recall = pure_tp / (pure_tp + pure_fn) if (pure_tp + pure_fn) > 0 else 0
    pure_f1 = 2 * (pure_precision * pure_recall) / (pure_precision + pure_recall) if (pure_precision + pure_recall) > 0 else 0
    
    report_lines = [
        "="*30,
        f"EVALUATION REPORT: {run_name}",
        "="*30,
        f"Correct Predictions (TP): {tp_total}",
        f"Wrong Predictions   (FP): {fp_total}",
        f"  -> On Existing Bolt : {fp_on_existing_bolt} (Cluster filled gap, but bolt was actually there)",
        f"Missed Bolts        (FN): {fn_total}",
        f"  -> Due to YOLO FP : {fn_due_to_yolo_occlusion} (Cluster blocked by wrong YOLO detection)",
        "-" * 30,
        f"Precision: {precision:.4f}",
        f"Recall:    {recall:.4f}",
        f"F1 Score:  {f1:.4f}",
        f"Avg Error: {avg_error:.6f}",
        "="*30,
        f"PURE CLUSTER PERFORMANCE (Hypothetical Perfect YOLO)",
        f"Assumption: YOLO errors caused specific Cluster errors.",
        "-" * 30,
        f"Pure TP:   {pure_tp} (Original TP + Blocked by YOLO)",
        f"Pure FP:   {pure_fp} (Original FP - YOLO Missed Bolt)",
        f"Pure FN:   {pure_fn} (Original FN - Blocked by YOLO)",
        f"Pure Prec: {pure_precision:.4f}",
        f"Pure Rec:  {pure_recall:.4f}",
        f"Pure F1:   {pure_f1:.4f}",
        "="*30
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save report to file
    ratings_dir = os.path.join(cfg['paths']['output_root'], "performance_ratings")
    ensure_dir(ratings_dir)
    
    save_path = os.path.join(ratings_dir, f"{run_name}.txt")
    with open(save_path, 'w') as f:
        f.write(report_text)
        
    print(f"Report saved to {save_path}")

if __name__ == "__main__":
    evaluate()
