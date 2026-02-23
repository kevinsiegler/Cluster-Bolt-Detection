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

    # Verhindert Division durch Null, wenn eine Box keine Fläche hat.
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area

def calculate_metrics_per_class(gt_labels, pred_labels, iou_threshold):
    gt_boxes = gt_labels[:, 1:5]
    pred_boxes = pred_labels[:, 1:5]
    gt_classes = gt_labels[:, 0].astype(int)
    pred_classes = pred_labels[:, 0].astype(int)

    stats = {
        0: {"tp": 0, "wrong": 0, "missed": 0, "fp": 0},
        1: {"tp": 0, "wrong": 0, "missed": 0, "fp": 0}
    }

    matched_pred_indices = set()

    # --- 1. GT durchgehen ---
    for i, gt_box in enumerate(gt_boxes):
        gt_class = gt_classes[i]
        best_iou = 0
        best_pred_idx = -1

        for j, pred_box in enumerate(pred_boxes):
            if j in matched_pred_indices:
                continue

            iou = box_iou_xywh(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = j

        if best_iou >= iou_threshold and best_pred_idx != -1:
            matched_pred_indices.add(best_pred_idx)
            pred_class = pred_classes[best_pred_idx]

            if gt_class == pred_class:
                stats[gt_class]["tp"] += 1
            else:
                stats[gt_class]["wrong"] += 1
                stats[pred_class]["fp"] += 1
        else:
            stats[gt_class]["missed"] += 1

    # --- 2. Übrige Predictions = Überflüssige Boxen ---
    for j in range(len(pred_boxes)):
        if j not in matched_pred_indices:
            pred_class = pred_classes[j]
            stats[pred_class]["fp"] += 1

    return stats



def compute_metrics(stats):
    tp = stats["tp"]
    wrong = stats["wrong"]
    missed = stats["missed"]
    fp = stats["fp"]

    precision = tp / (tp + wrong + fp) if (tp + wrong + fp) > 0 else 0
    recall = tp / (tp + wrong + missed) if (tp + wrong + missed) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": tp,
        "Wrong": wrong,
        "Missed": missed,
        "FP": fp,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

# 2️⃣ print_table
def print_table(title, yolo_stats, gnn_stats, log_func=print):
    """
    Zeigt Metriken im Tabellenformat, alle Prozentwerte 0–100.
    Berechnet F1 direkt aus globalen Summen.
    """
    log_func(f"\n\n===== {title} =====")

    def to_metrics(stats, is_global=False):
        TP = stats["tp"]
        Wrong = stats["wrong"]
        Missed = stats["missed"]
        FP = stats["fp"]

        if is_global:
            # Dashboard-Logik für F1: direkt aus den Summen
            Precision = TP / (TP + Wrong + FP) if (TP + Wrong + FP) > 0 else 0
            Recall = TP / (TP + Wrong + Missed) if (TP + Wrong + Missed) > 0 else 0
            F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0
        else:
            # Klassische Berechnung identisch
            Precision = TP / (TP + Wrong + FP) if (TP + Wrong + FP) > 0 else 0
            Recall = TP / (TP + Wrong + Missed) if (TP + Wrong + Missed) > 0 else 0
            F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0

        return {
            "TP": TP,
            "Wrong": Wrong,
            "Missed": Missed,
            "FP": FP,
            "Precision": Precision*100,
            "Recall": Recall*100,
            "F1": F1*100
        }

    # Global-Tabelle bekommt is_global=True
    is_global = (title == "GLOBAL")
    yolo_metrics = to_metrics(yolo_stats, is_global=is_global)
    gnn_metrics = to_metrics(gnn_stats, is_global=is_global)

    # Tabellenheader
    header = f"{'Metric':<12} | {'Raw YOLO':>12} | {'GNN':>12} | {'Δ (GNN-YOLO)':>14}"
    log_func(header)
    log_func("-" * len(header))

    # Zahlen
    for key in ["TP", "Wrong", "Missed", "FP"]:
        y = yolo_metrics[key]
        g = gnn_metrics[key]
        diff = g - y
        log_func(f"{key:<12} | {y:12d} | {g:12d} | {diff:14d}")

    # Prozentwerte
    for key in ["Precision", "Recall", "F1"]:
        y = yolo_metrics[key]
        g = gnn_metrics[key]
        diff = g - y
        log_func(f"{key:<12} | {y:12.2f} % | {g:12.2f} % | {diff:14.2f} %")





def main():
    """
    Vergleicht die Performance von rohen YOLO-Labels und GNN-validierten Labels
    mit den Ground-Truth-Labels.
    """
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    gt_dir = cfg["paths"]["val_labels"]
    yolo_dir = cfg["paths"]["yolo_inference"]
    inference_run_name = cfg["inference"]["run_name"]
    gnn_dir = os.path.join(cfg["paths"]["output_root"], "validated_labels", inference_run_name)
    iou_threshold = cfg["evaluation"]["iou_threshold"]

    gt_files = {
        os.path.splitext(f)[0]: os.path.join(gt_dir, f)
        for f in os.listdir(gt_dir)
        if f.endswith(".txt")
    }

    # Aggregation initialisieren
    yolo_agg = {
        "total": {"tp":0,"wrong":0,"missed":0,"fp":0},
        0: {"tp":0,"wrong":0,"missed":0,"fp":0},
        1: {"tp":0,"wrong":0,"missed":0,"fp":0}
    }

    gnn_agg = {
        "total": {"tp":0,"wrong":0,"missed":0,"fp":0},
        0: {"tp":0,"wrong":0,"missed":0,"fp":0},
        1: {"tp":0,"wrong":0,"missed":0,"fp":0}
    }

    def aggregate(agg, new_stats):
        for key in [0,1]:   # nur über die Klassen iterieren
            agg[key]["tp"] += new_stats[key]["tp"]
            agg[key]["wrong"] += new_stats[key]["wrong"]
            agg[key]["missed"] += new_stats[key]["missed"]
            agg[key]["fp"] += new_stats[key]["fp"]


    for image_id, gt_path in tqdm(gt_files.items(), desc="Evaluating files"):
        yolo_labels = load_yolo_labels(os.path.join(yolo_dir, f"{image_id}.txt"), with_confidence=True)
        gnn_labels = load_yolo_labels(os.path.join(gnn_dir, f"{image_id}.txt"), with_confidence=True)
        gt_labels = load_yolo_labels(gt_path, with_confidence=False)

        yolo_stats = calculate_metrics_per_class(gt_labels, yolo_labels, iou_threshold)
        gnn_stats = calculate_metrics_per_class(gt_labels, gnn_labels, iou_threshold)

        aggregate(yolo_agg, yolo_stats)
        aggregate(gnn_agg, gnn_stats)


    # Output file setup
    perf_dir = os.path.join(cfg["paths"]["output_root"], "performance_ratings")
    os.makedirs(perf_dir, exist_ok=True)
    log_path = os.path.join(perf_dir, f"{inference_run_name}.txt")

    with open(log_path, "w", encoding="utf-8") as f:
        def log_func(msg):
            print(msg)
            f.write(msg + "\n")
        
        # Globale Metriken hier berechnen, direkt vor der Ausgabe
        global_yolo_agg = {
            "tp": yolo_agg[0]["tp"] + yolo_agg[1]["tp"],
            "wrong": yolo_agg[0]["wrong"] + yolo_agg[1]["wrong"],
            "missed": yolo_agg[0]["missed"] + yolo_agg[1]["missed"],
            "fp": yolo_agg[0]["fp"] + yolo_agg[1]["fp"]
        }

        global_gnn_agg = {
            "tp": gnn_agg[0]["tp"] + gnn_agg[1]["tp"],
            "wrong": gnn_agg[0]["wrong"] + gnn_agg[1]["wrong"],
            "missed": gnn_agg[0]["missed"] + gnn_agg[1]["missed"],
            "fp": gnn_agg[0]["fp"] + gnn_agg[1]["fp"]
        }

        print_table("CLASS 0 – BOLT", yolo_agg[0], gnn_agg[0], log_func=log_func)
        print_table("CLASS 1 – MISSING BOLT", yolo_agg[1], gnn_agg[1], log_func=log_func)
        print_table("GLOBAL", global_yolo_agg, global_gnn_agg, log_func=log_func)
        
        log_func(f"\nReport saved to: {log_path}")



if __name__ == "__main__":
    main()
