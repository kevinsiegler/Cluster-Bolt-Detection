import os
import yaml
import numpy as np
from tqdm import tqdm
from utils import load_yolo_labels

def box_iou_xywh(box1, box2):
    """
    Berechnet die Intersection over Union (IoU) von zwei Bounding Boxes im YOLO-Format (xc, yc, w, h).
    """
    # Konvertiere (xc, yc, w, h) zu (x1, y1, x2, y2)
    def to_corners(box):
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        x2 = box[0] + box[2] / 2
        y2 = box[1] + box[3] / 2
        return np.array([x1, y1, x2, y2])

    box1_corners = to_corners(box1)
    box2_corners = to_corners(box2)

    # Berechne die Koordinaten der Schnittfläche
    x_left = max(box1_corners[0], box2_corners[0])
    y_top = max(box1_corners[1], box2_corners[1])
    x_right = min(box1_corners[2], box2_corners[2])
    y_bottom = min(box1_corners[3], box2_corners[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def calculate_metrics(gt_labels, pred_labels, iou_threshold):
    """
    Berechnet True Positives, False Positives und False Negatives
    nach der Logik des Dashboards (Spatial Matching first).
    """
    gt_boxes = gt_labels[:, 1:5]
    pred_boxes = pred_labels[:, 1:5]
    gt_classes = gt_labels[:, 0]
    pred_classes = pred_labels[:, 0]

    # Zähler analog zum Dashboard
    tp_correct = 0      # Match + Klasse korrekt
    wrong_class = 0     # Match + Klasse falsch
    missed = 0          # Kein Match für GT
    
    matched_pred_indices = set()

    # 1. Iteriere über GTs und suche besten Match (Greedy)
    for i, gt_box in enumerate(gt_boxes):
        gt_class = gt_classes[i]
        best_iou = 0
        best_pred_idx = -1

        for j, pred_box in enumerate(pred_boxes):
            if j in matched_pred_indices:
                continue
            
            # IoU berechnen (Reihenfolge der Argumente beachten, hier aber symmetrisch)
            iou = box_iou_xywh(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = j
        
        if best_iou >= iou_threshold and best_pred_idx != -1:
            matched_pred_indices.add(best_pred_idx)
            pred_class = pred_classes[best_pred_idx]
            
            if gt_class == pred_class:
                tp_correct += 1
            else:
                wrong_class += 1
        else:
            missed += 1
            
    # 2. Übrige Predictions sind False Positives
    false_positives = len(pred_boxes) - len(matched_pred_indices)
    
    # Mapping auf Standard-Metriken:
    # Precision = tp_correct / (tp_correct + wrong_class + false_positives)
    # Recall    = tp_correct / (tp_correct + wrong_class + missed)
    
    tp = tp_correct
    fp = wrong_class + false_positives
    fn = wrong_class + missed

    return tp, fp, fn

def print_report(name, tp, fp, fn):
    """Druckt einen formatierten Bericht."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n--- {name} vs. Ground Truth ---")
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  ---------------------------------")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    return precision, recall, f1

def main():
    """
    Vergleicht die Performance von rohen YOLO-Labels und GNN-validierten Labels
    mit den Ground-Truth-Labels.
    """
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # --- Pfade definieren ---
    gt_dir = cfg["paths"]["val_labels"]
    yolo_dir = cfg["paths"]["yolo_inference"]
    inference_run_name = cfg["inference"]["run_name"]
    gnn_dir = os.path.join(cfg["paths"]["output_root"], "validated_labels", inference_run_name)
    iou_threshold = cfg["evaluation"]["iou_threshold"]

    print("--- Performance Evaluation ---")
    print(f"Comparing against Ground Truth validation labels from: {gt_dir}")
    print(f"Raw YOLO labels from: {yolo_dir}")
    print(f"GNN validated labels from: {gnn_dir}")
    print(f"Using IoU threshold: {iou_threshold}")

    # --- Sammle alle Ground-Truth-Dateien ---
    gt_files = {os.path.splitext(f)[0]: os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(".txt")}

    # --- Initialisiere Metriken ---
    yolo_tp_total, yolo_fp_total, yolo_fn_total = 0, 0, 0
    gnn_tp_total, gnn_fp_total, gnn_fn_total = 0, 0, 0

    for image_id, gt_path in tqdm(gt_files.items(), desc="Evaluating files"):
        yolo_labels = load_yolo_labels(os.path.join(yolo_dir, f"{image_id}.txt"), with_confidence=True)
        gnn_labels = load_yolo_labels(os.path.join(gnn_dir, f"{image_id}.txt"), with_confidence=True)
        gt_labels = load_yolo_labels(gt_path, with_confidence=False)

        tp, fp, fn = calculate_metrics(gt_labels, yolo_labels, iou_threshold); yolo_tp_total+=tp; yolo_fp_total+=fp; yolo_fn_total+=fn
        tp, fp, fn = calculate_metrics(gt_labels, gnn_labels, iou_threshold); gnn_tp_total+=tp; gnn_fp_total+=fp; gnn_fn_total+=fn

    # --- Ergebnisse ausgeben ---
    yolo_precision, yolo_recall, _ = print_report("Raw YOLO", yolo_tp_total, yolo_fp_total, yolo_fn_total)
    gnn_precision, gnn_recall, _ = print_report("GNN Validated", gnn_tp_total, gnn_fp_total, gnn_fn_total)

    print("\n--- Summary ---")
    print(f"GNN validation changed Precision by: {gnn_precision - yolo_precision:+.4f}")
    print(f"GNN validation changed Recall by:    {gnn_recall - yolo_recall:+.4f}")

if __name__ == "__main__":
    main()