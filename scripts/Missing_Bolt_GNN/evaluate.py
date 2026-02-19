r"""
c:\Users\Kevin\Clustererkennung\bolt_detection\scripts\Missing_Bolt_GNN\evaluate.py
"""
import os
import numpy as np
import torch
from scipy.spatial.distance import cdist
from utils import CONFIG, load_yolo_labels, setup_run_directories

def calculate_metrics(gt_points, pred_points, threshold=0.05):
    """
    Calculates Precision, Recall, F1 based on Euclidean distance matching.
    Args:
        gt_points: (N, 2) normalized
        pred_points: (M, 2) normalized
        threshold: normalized distance threshold
    """
    tp = 0
    fp = 0
    fn = 0
    
    if len(pred_points) == 0:
        return 0, 0, len(gt_points), 0.0, 0.0, 0.0, 0.0

    if len(gt_points) == 0:
        return 0, len(pred_points), 0, 0.0, 0.0, 0.0, 0.0

    # Distance matrix (N_gt, M_pred)
    dists = cdist(gt_points, pred_points)
    
    # Simple greedy matching
    # Find closest prediction for each GT
    matched_gt = set()
    matched_pred = set()
    total_pos_error = 0
    
    # Iterate through GT points to find matches
    for i in range(len(gt_points)):
        # Find closest prediction
        min_idx = np.argmin(dists[i])
        min_dist = dists[i][min_idx]
        
        if min_dist < threshold:
            if min_idx not in matched_pred:
                tp += 1
                matched_pred.add(min_idx)
                matched_gt.add(i)
                total_pos_error += min_dist
            else:
                # Prediction already matched to another GT?
                # In a strict Hungarian algorithm we would optimize global cost.
                # Here we assume sparse points, so greedy is fine.
                # If multiple GT map to same Pred, only one is TP, others FN.
                pass
    
    fn = len(gt_points) - len(matched_gt)
    fp = len(pred_points) - len(matched_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_error = total_pos_error / tp if tp > 0 else 0
    
    return tp, fp, fn, precision, recall, f1, avg_error

def evaluate():
    # Setup output directory: output/evaluation/<inference_run_name>
    eval_output_dir = setup_run_directories(CONFIG, 'evaluation')
    
    inference_run_name = CONFIG['evaluation']['inference_run']
    
    # Paths
    gt_dir = CONFIG['paths']['val_labels']
    pred_dir = os.path.join(CONFIG['paths']['output_root'], "inference", inference_run_name)
    
    print(f"Evaluating Inference Run: {inference_run_name}")
    print(f"Ground Truth: {gt_dir}")
    print(f"Predictions: {pred_dir}")
    
    if not os.path.exists(pred_dir):
        print("Prediction directory not found. Run inference.py first.")
        return

    files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_error = 0
    count_tp_images = 0
    
    results_log = []
    
    for filename in files:
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)
        
        if not os.path.exists(pred_path):
            continue
            
        # Load Labels
        gt_labels = load_yolo_labels(gt_path)
        pred_labels = load_yolo_labels(pred_path)
        
        # Filter:
        # GT: We want to see if the model found the MISSING bolts.
        # In the original dataset, Class 1 = Missing Bolt.
        gt_missing = gt_labels[gt_labels[:, 0] == 1][:, 1:3]
        
        # Pred: The model outputs Class 1 for predicted missing bolts.
        pred_missing = pred_labels[pred_labels[:, 0] == 1][:, 1:3]
        
        tp, fp, fn, prec, rec, f1, err = calculate_metrics(
            gt_missing, 
            pred_missing, 
            threshold=CONFIG['evaluation']['distance_threshold']
        )
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_error += (err * tp) # Weighted sum
        if tp > 0:
            count_tp_images += tp
            
        results_log.append(f"{filename}: TP={tp}, FP={fp}, FN={fn}, F1={f1:.2f}")

    # Global Metrics
    precision_global = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall_global = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_global = 2 * (precision_global * recall_global) / (precision_global + recall_global) if (precision_global + recall_global) > 0 else 0
    avg_pos_error = total_error / total_tp if total_tp > 0 else 0
    
    # Output Report
    report_path = os.path.join(eval_output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"EVALUATION REPORT - Inference Run: {inference_run_name}\n")
        f.write("========================================\n")
        f.write(f"Global Precision: {precision_global:.4f}\n")
        f.write(f"Global Recall:    {recall_global:.4f}\n")
        f.write(f"Global F1 Score:  {f1_global:.4f}\n")
        f.write(f"Avg Position Err: {avg_pos_error:.6f} (normalized)\n")
        f.write("----------------------------------------\n")
        f.write(f"Total True Positives:  {total_tp}\n")
        f.write(f"Total False Positives: {total_fp}\n")
        f.write(f"Total False Negatives: {total_fn}\n")
        f.write("========================================\n")
        f.write("Detailed per Image:\n")
        for line in results_log:
            f.write(line + "\n")
            
    print("\nEvaluation Complete.")
    print(f"Precision: {precision_global:.4f}, Recall: {recall_global:.4f}, F1: {f1_global:.4f}")
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    evaluate()
